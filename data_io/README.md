# Data I/O and Communication Utilities

This package provides essential utilities for data handling, communication, and numerical representation in distributed computing applications, with particular emphasis on privacy-preserving federated learning. The package implements robust data loading, efficient communication protocols, and precise numerical representations necessary for secure distributed computations.

## Core Components

### MPI Communication (`comm.py`)

The communication module provides a high-level interface for distributed computing using the Message Passing Interface (MPI). It includes sophisticated features for simulating real-world network conditions and tracking communication performance.

```python
from data_io.comm import Communicator

# Initialize communication with custom delay
comm = Communicator(delay=0.05)  # 50ms network delay

# Perform distributed operations
data = comm.gather_delay(local_data, root=0)
comm.bcast_delay(processed_data, root=0)

# Get communication statistics
stats = comm.get_comm_stats()
print(f"Communication rounds: {stats['num_comm_rounds']}")
print(f"Total data transferred: {stats['comm_size']} bytes")
```

Key Features:
- Simulated network delays for realistic testing
- Communication statistics tracking
- Separate client-only communication channels
- Robust error handling across distributed processes
- Support for both point-to-point and collective operations

### Data Processing (`data_handler.py`)

This module handles data loading, partitioning, and normalization, making it easy to prepare data for distributed computing applications.

```python
from data_io.data_handler import load_txt, shuffle_and_split, normalize

# Load and prepare data
raw_data = load_txt("dataset.txt")
normalized_data = normalize(raw_data, fixed=True)

# Split data among clients
client_data = shuffle_and_split(
    normalized_data,
    clients=4,
    proportions=[0.25, 0.25, 0.25, 0.25]
)
```

### Fixed-Point Arithmetic (`fixed.py`)

The fixed-point module enables precise numerical computations by providing utilities for fixed-point arithmetic, which is crucial for privacy-preserving computations.

```python
from data_io.fixed import to_fixed, unscale, MOD

# Convert floating-point values to fixed-point representation
fixed_values = to_fixed(floating_values)

# Perform computations in fixed-point arithmetic
result = (fixed_values * fixed_multiplier) % MOD

# Convert back to floating-point for final results
float_result = unscale(result)
```

Important Features:
- 16-bit precision for fractional parts
- 32-bit modular arithmetic support
- Seamless conversion between representations

## Advanced Usage Examples

### Secure Multi-Party Communication

Here's how to implement secure communication between multiple parties:

```python
from data_io.comm import Communicator, fail_together
from data_io.fixed import to_fixed

def secure_aggregation():
    comm = Communicator(delay=0.025)  # WAN-like delay
    
    # Secure function execution with failure handling
    def compute_local_stats():
        local_data = to_fixed(compute_statistics())
        return local_data
    
    # Ensure all parties succeed or fail together
    local_result = fail_together(
        compute_local_stats,
        "Secure aggregation failed"
    )
    
    # Gather results with network delay simulation
    aggregated_results = comm.gather_delay(
        local_result,
        root=0
    )
    
    return aggregated_results

```

### Data Preprocessing Pipeline

Example of a complete data preprocessing pipeline:

```python
from data_io.data_handler import load_txt, normalize, shuffle_and_split
from data_io.fixed import to_fixed

def prepare_federated_data(data_path, num_clients):
    # Load and normalize data
    raw_data = load_txt(data_path)
    normalized_data = normalize(raw_data)
    
    # Convert to fixed-point representation
    fixed_data = to_fixed(normalized_data)
    
    # Split data among clients
    client_data = shuffle_and_split(
        fixed_data,
        clients=num_clients
    )
    
    return client_data
```

## Network Simulation Features

The communication module provides sophisticated network simulation capabilities:

1. Configurable Delays:
   - LAN-like conditions (sub-millisecond delays)
   - WAN-like conditions (tens of milliseconds)
   - Custom delay profiles

2. Performance Monitoring:
   - Communication round counting
   - Data volume tracking
   - Bandwidth utilization analysis

3. Error Handling:
   - Synchronized error recovery
   - Distributed failure detection
   - Graceful degradation support

## Installation Notes

1. Install MPI implementation:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install libopenmpi-dev
   
   # CentOS/RHEL
   sudo yum install openmpi-devel
   ```

2. Install Python dependencies:
   ```bash
   pip install numpy mpi4py
   ```