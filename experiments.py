"""
Experiments runner for multiparty DP clustering.
Handles experiment configuration, execution, and result collection.
"""

import itertools
import json
import os
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from functools import partial
from pathlib import Path
from timeit import default_timer as timer
from typing import List, Dict, Any, Generator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from configs import Params, exp_parameter_dict, num_clusters
from data_io import shuffle_and_split, unscale, load_txt, normalize
from utils import evaluate, mean_confidence_interval, plot_clusters


class ExperimentRunner:
    """Handles the execution and management of clustering experiments."""

    def __init__(
            self,
            protocol: callable,
            k: int,
            dataset: str,
            values: np.ndarray,
            params_list: Dict[str, Any],
            exp_type: str,
            results_folder: str,
            plot: bool = False,
            with_comm: bool = False
    ):
        """
        Initialize the experiment runner.

        Args:
            protocol: The clustering protocol to run
            k: Number of clusters
            dataset: Name of the dataset
            values: Input data values
            params_list: Dictionary of experiment parameters
            exp_type: Type of experiment
            results_folder: Output directory for results
            plot: Whether to generate plots
            with_comm: Whether to use communication metrics
        """
        self.results_folder = results_folder
        self.protocol = protocol
        self.k = k
        self.values = values
        self.dataset = dataset
        self.plot = plot
        self.with_comm = with_comm
        self.comm = None
        if with_comm:
            from data_io.comm import comm
            self.comm = comm
        self.params_list = params_list
        self.exp_type = exp_type

        self.results_df = None
        self.failed_experiments = []

    def run_single_protocol(self, params: Params) -> Dict[str, float]:
        """
        Run a single instance of the clustering protocol.

        Args:
            params: Parameters for this protocol run

        Returns:
            Dictionary of evaluation metrics
        """
        # Prepare data
        proportions = np.ones(params.num_clients) / params.num_clients
        value_lists = shuffle_and_split(self.values, params.num_clients, proportions)

        # Run protocol and time it
        start = timer()
        centroids, unassigned = self.protocol(value_lists, params)
        elapsed_time = timer() - start

        # Handle scaling
        values_unscaled = unscale(self.values.copy()) if params.fixed else self.values
        centroids_final = unscale(centroids) if params.fixed else centroids

        # Evaluate results
        metrics = evaluate(centroids_final, values_unscaled)
        metrics["elapsed"] = elapsed_time
        metrics["unassigned"] = unassigned

        # Generate plots if requested
        if self.plot:
            self._generate_plot(centroids_final, values_unscaled, params)

        return metrics

    def _generate_plot(self, centroids: np.ndarray, values: np.ndarray, params: Params) -> None:
        """Generate and save clustering visualization."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = (f"{timestamp}_{params.method}_{params.dp}_{params.eps}_"
                    f"[{params.post}-{params.alpha}]_{params.seed}")

        plot_clusters(centroids, values)
        plt.title(filename)

        folder = Path("results") / self.dataset / self.protocol.__name__
        folder.mkdir(parents=True, exist_ok=True)

        plt.savefig(folder / f"{filename}.png")
        plt.close()

    def _get_parameter_combinations(self) -> Generator[Params, None, None]:
        """Generate all parameter combinations for experiments."""
        params_order = ["methods", "posts", "delays", "dps"]
        dimension, data_size = self.values.shape[1], self.values.shape[0]

        for method, post, delay, dp in itertools.product(
                *[self.params_list[key] for key in params_order]
        ):
            for eps_budget in self._get_eps_budgets(dp):
                params = Params(
                    num_clients=self.params_list["num_clients"],
                    k=self.k,
                    dim=dimension,
                    data_size=data_size,
                    dp=dp,
                    eps=eps_budget,
                    method=method,
                    post=post,
                    delay=delay,
                )

                if method == "none":
                    params.alpha = 0
                    yield params
                else:
                    for alpha in self.params_list["alphas"]:
                        params.alpha = alpha
                        yield params

    def _get_eps_budgets(self, dp: str) -> List[float]:
        """Get epsilon budgets based on privacy setting."""
        return [0] if dp == "none" else self.params_list["eps_budgets"]

    def run_experiment(self, params: Params) -> None:
        """Run experiment with given parameters multiple times."""
        params.calculate_iters()
        total_metrics = defaultdict(list)
        successful_experiments = experiment_count = 0

        # Run multiple times with different seeds
        for seed in self.params_list["seeds"]:
            params.seed = seed
            try:
                metrics = self.run_single_protocol(params)

                # Validate metrics
                if any(np.isnan(value) for value in metrics.values()):
                    raise ValueError("NAN values in experiment results")

                # Collect metrics
                for metric, value in metrics.items():
                    total_metrics[metric].append(value)

                successful_experiments += metrics["Empty Clusters"] == 0
                experiment_count += 1

            except Exception as e:
                print(f"Experiment failed: {str(e)}")
                self.failed_experiments.append(vars(params))
                self._save_results()

        # Process and save results
        self._process_and_save_results(
            params, total_metrics, successful_experiments, experiment_count
        )

    def _process_and_save_results(
            self,
            params: Params,
            total_metrics: Dict[str, List[float]],
            successful_experiments: int,
            experiment_count: int
    ) -> None:
        """Process experiment results and save to DataFrame."""
        # Calculate statistics
        metric_stats = {
            metric: mean_confidence_interval(values)
            for metric, values in total_metrics.items()
        }

        # Prepare results dictionary
        result = {
            **{attr: getattr(params, attr) for attr in vars(params)},
            "successes": successful_experiments,
            "experiments": experiment_count,
            "post_method": params.post,
            **{metric: stats[0] for metric, stats in metric_stats.items()},
            **{f"{metric}_h": stats[1] for metric, stats in metric_stats.items()}
        }

        # Remove unnecessary attributes
        result.pop("attributes", None)

        # Add communication stats if needed
        if self.with_comm:
            result.update(self.comm.get_comm_stats())

        # Update DataFrame
        new_results = pd.DataFrame([result])
        self.results_df = (
            pd.concat([self.results_df, new_results], ignore_index=True)
            if self.results_df is not None
            else new_results
        )

        self._save_results()

    def _save_results(self) -> None:
        """Save results to files."""
        folder = Path(self.results_folder) / self.exp_type / self.dataset
        folder.mkdir(parents=True, exist_ok=True)
        print(f"Saving results to {folder}")
        # Sort and save results
        if self.results_df is not None:
            self.results_df = self.results_df.sort_values("Normalized Intra-cluster Variance (NICV)")

            # Determine filename based on communication setting
            filename = "variances.csv"
            if self.with_comm:
                rank_str = f"_{self.comm.rank}" if self.comm.world_size > 1 else ""
                filename = f"variances{rank_str}.csv"

            self.results_df.to_csv(folder / filename)

        # Save failed experiments
        with open(folder / "failed.json", "w") as f:
            json.dump(self.failed_experiments, f)

    def run(self) -> None:
        """Run all experiments with different parameter combinations."""
        for params in self._get_parameter_combinations():
            self.run_experiment(params)


def parse_args() -> Namespace:
    """Parse command line arguments."""
    parser = ArgumentParser(description="Run experiments for multiparty DP clustering")
    parser.add_argument("--exp_type", default="test", help="type of experiment")
    parser.add_argument("--datasets", nargs="+", default=["mnist"], help="datasets to run")
    parser.add_argument("--plot", action="store_true", help="plot clusters")
    parser.add_argument("--num_runs", default=100, type=int, help="number of runs")
    parser.add_argument(
        "--method",
        default="diagonal_then_frac",
        choices=["none", "diagonal_then_frac", "stay_frac"],
        help="maxdist method"
    )
    parser.add_argument("--alpha", default=0.8, type=float, help="max distance alpha")
    parser.add_argument(
        "--post",
        default="fold",
        choices=["none", "truncate", "fold"],
        help="centroid post-processing method"
    )
    parser.add_argument(
        "--results_folder",
        default="submission",
        help="folder for results"
    )
    return parser.parse_args()


def process_dataset(
        dataset: str,
        proto: callable,
        params_list: Dict[str, Any],
        fixed: bool,
        exp_type: str,
        results_folder: str,
        plot: bool,
        with_comm: bool
) -> None:
    """Process a single dataset with given parameters."""
    # Determine number of clusters
    k = int(dataset.split("_")[1]) if "synth" in dataset.lower() else num_clusters[dataset]

    # Load and prepare dataset
    dataset_file = Path("data") / f"{dataset}.txt"
    if not dataset_file.is_file():
        return

    values = load_txt(str(dataset_file))
    values = normalize(values, fixed)

    # Run experiments
    experiment = ExperimentRunner(
        proto, k, dataset, values, params_list,
        exp_type, results_folder, plot, with_comm
    )
    experiment.run()


def main() -> None:
    """Main entry point for running experiments."""
    args = parse_args()
    fixed = True  # Always use fixed-point
    exp_type = args.exp_type

    # Set up default parameters
    params_list = {
        "num_runs": args.num_runs,
        "seeds": range(args.num_runs),
        "posts": [args.post],
        "methods": [args.method],
        "alphas": [args.alpha],
        "datasets": args.datasets,
        "dps": ["none", "gaussiananalytic"],
        "delays": [0],
        "fixed": fixed,
    }

    # Override parameters if needed
    if exp_type in exp_parameter_dict:
        params_list.update(exp_parameter_dict[exp_type])

    # Set up protocol and communication
    if "timing" in exp_type:
        from data_io.comm import comm
        from utils.protocols import mpi_proto
        proto = mpi_proto
        with_comm = True
        num_clients = comm.world_size - 1
        params_list["num_clients"] = num_clients
        exp_type = f"timing_{num_clients}"
    else:
        from utils.protocols import local_proto
        proto = local_proto
        with_comm = False
        params_list["num_clients"] = 2

    # Run experiments in parallel
    max_processes = min(os.cpu_count() or 1, len(params_list["datasets"]))
    if "timing" in exp_type:
        max_processes = 1

    with ProcessPoolExecutor(max_workers=max_processes) as executor:
        partial_fn = partial(
            process_dataset,
            proto=proto,
            params_list=params_list,
            fixed=fixed,
            exp_type=exp_type,
            results_folder=args.results_folder,
            plot=args.plot,
            with_comm=with_comm
        )
        executor.map(partial_fn, params_list["datasets"])


if __name__ == "__main__":
    main()
