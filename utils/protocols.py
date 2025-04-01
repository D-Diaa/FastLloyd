import numpy as np
from tqdm import tqdm

from parties import MaskedClient, UnmaskedClient, Server
from data_io import to_fixed
from configs import Params
from utils import set_seed


def mpi_proto(value_lists, params: Params, method="masked"):
    """Implements the MPI-based protocol for federated clustering.
    
    This protocol uses Message Passing Interface (MPI) for distributed computation,
    enabling communication between multiple processes representing different clients
    and a central server. It supports both masked (privacy-preserving) and unmasked
    operations.
    
    The protocol follows these steps in each iteration:
    1. Clients compute local statistics (totals and counts)
    2. These statistics are gathered at the server
    3. Server processes the aggregated statistics (possibly adding DP noise)
    4. Results are broadcast back to all clients
    5. Clients update their local centroids
    
    Args:
        value_lists (list): List of numpy arrays, where each array contains the data
                           points for one client
        params (Params): Configuration parameters for the clustering algorithm
        method (str, optional): Either "masked" for privacy-preserving computation
                              or "unmasked" for standard computation. Defaults to "masked"
    
    Returns:
        tuple: A tuple containing:
            - np.ndarray: Final cluster centroids after all iterations
            - int: Always 0 for MPI protocol (kept for compatibility with local_proto)
            
    Note:
        - The protocol uses the `data_io.comm` module for MPI operations
        - Progress is shown only on the server process using tqdm
        - Communication statistics are tracked and printed at the end
    """
    from data_io.comm import comm, fail_together
    set_seed(params.seed)
    comm.reset_comm_stats()
    comm.set_delay(params.delay)
    server_process = (comm.rank == comm.root)

    def initialize_server():
        """Initialize the server process with given parameters."""
        return Server(params)

    def initialize_client():
        """Initialize a client (masked or unmasked) with appropriate data."""
        cls = MaskedClient if method == "masked" else UnmaskedClient
        return cls(comm.rank - 1, value_lists[comm.rank - 1], params)

    if server_process:
        server = fail_together(initialize_server, "Server Initialization Failure")
    else:
        client = fail_together(initialize_client, "Client Initialization Failure")
    pbar = tqdm(range(params.iters)) if server_process else range(params.iters)

    for i in pbar:
        params.update_maxdist(i)
        if not server_process:
            # Client-side computation
            total, count, _ = client.step(params)

            # Pack statistics into a single array for efficient communication
            total_count = np.concatenate((total.flatten(), count.flatten()))
            comm.gather_delay(total_count, root=comm.root)

            # Receive and unpack aggregated statistics from server
            total_count = comm.bcast_delay(None, root=comm.root)

            total, count = np.split(total_count, [params.k * params.dim])
            total = total.reshape((params.k, params.dim))
            count = count.reshape(params.k)
            client.update(total, count)

        if server_process:
            # Server-side computation
            total_counts = comm.gather_delay(None, root=comm.root)
            total_counts = [np.split(tc, [params.k * params.dim]) for tc in total_counts[1:]]
            totals, counts = zip(*[(total.reshape((params.k, params.dim)), count.reshape(params.k))
                                   for total, count in total_counts])
            total, count = server.step(totals, counts, params)

            # Pack and broadcast updated statistics
            total_count = np.concatenate((total.flatten(), count.flatten()))
            comm.bcast_delay(total_count, root=comm.root)

        # Synchronize centroids across all processes
        centroids = comm.bcast(client.centroids if not server_process else None, root=1)

    comm.print_comm_stats()
    return to_fixed(centroids), 0


def local_proto(value_lists, params: Params, method="masked"):
    """Implements the local protocol for federated clustering.
    
    This protocol simulates federated clustering in a single process, useful for
    testing and development. It maintains separate client and server instances
    in memory and simulates their interaction. Like the MPI protocol, it supports
    both masked and unmasked computation.
    
    The protocol follows these steps in each iteration:
    1. Each client computes local statistics
    2. The server aggregates these statistics
    3. Clients update their centroids using the aggregated statistics
    4. Progress is tracked through centroid movement
    
    The implementation also tracks the number of unassigned points (points too
    far from any centroid)
    
    Args:
        value_lists (list): List of numpy arrays, where each array contains the data
                           points for one client
        params (Params): Configuration parameters for the clustering algorithm
        method (str, optional): Either "masked" for privacy-preserving computation
                              or "unmasked" for standard computation. Defaults to "masked"
    
    Returns:
        tuple: A tuple containing:
            - np.ndarray: Final cluster centroids after all iterations
            - int: Number of points not assigned to any cluster in the final iteration
            
    Note:
        - Progress bar shows the Euclidean norm of centroid movement between iterations
        - All clients maintain identical centroids due to synchronized updates
        - A history of centroids is maintained but not returned
    """
    set_seed(params.seed)
    cls = MaskedClient if method == "masked" else UnmaskedClient
    clients = [
        cls(client, value_lists[client], params)
        for client in range(params.num_clients)
    ]
    centroids = clients[0].centroids
    centroid_history = [centroids]
    server = Server(params)
    pbar = tqdm(range(params.iters))
    unassigned_last_iter = 0

    for i in pbar:
        params.update_maxdist(i)
        # Collect statistics from all clients
        totals = []
        counts = []
        unassigneds = []
        for client in clients:
            total, count, unassigned = client.step(params)
            totals.append(total)
            counts.append(count)
            unassigneds.append(unassigned)
        unassigned_last_iter = sum(unassigneds)

        # Server processes aggregated statistics
        total, count = server.step(totals, counts, params)

        # Update all clients
        for client in clients:
            client.update(total, count)

        # Track progress through centroid movement
        err = np.linalg.norm(clients[0].centroids - centroids)
        pbar.set_description(str(err))
        centroids = clients[0].centroids
        centroid_history.append(centroids)

    return to_fixed(centroids), unassigned_last_iter
