# Privacy-Preserving Federated Clustering Parties

This package implements the core components for privacy-preserving federated clustering, enabling secure and privacy-aware distributed machine learning. It provides implementations for both the server and client parties in a federated learning system, with strong privacy guarantees through differential privacy and secure aggregation.

## Architecture

The package consists of three main classes that work together to enable privacy-preserving federated clustering:

1. `Server`: Coordinates the federated learning process and handles privacy-preserving aggregation
2. `UnmaskedClient`: Basic client implementation for federated clustering
3. `MaskedClient`: Enhanced client implementation with privacy-preserving features through masking

## Core Features

### Privacy Mechanisms

The system supports multiple differential privacy approaches:
- Laplace mechanism (ε-differential privacy)
- Gaussian analytic mechanism (ε,δ-differential privacy)
- Dynamic privacy budget allocation between count and sum queries
- Automatic sensitivity calculation based on data dimensionality

### Secure Aggregation

The `MaskedClient` implementation provides secure aggregation through:
- Random masking across clients, with sum of masks stored for decryption
- Separate masks for sums and counts
- Fixed-point arithmetic support for secure computations
- Seed-based mask generation for reproducibility and agreement between clients

### Bounded Updates

The system implements various constraints to ensure stability:
- Maximum distance bounds for centroid updates
- Truncation and folding mechanisms for bounded data
- Dynamic constraint adjustment based on iteration progress

## Usage Example

Here's a basic example of setting up a federated clustering system:

```python
from parties import Server, MaskedClient
from configs import Params

# Initialize server with privacy parameters
params = Params(
    dp="gaussiananalytic",
    eps=0.1,
    num_clients=3,
    k=5,  # number of clusters
    dim=2  # data dimensionality
)
server = Server(params)

# Create privacy-preserving clients
clients = []
for i in range(params.num_clients):
    client = MaskedClient(
        index=i,
        values=client_data[i],  # local data for each client
        params=params
    )
    clients.append(client)

# Run federated clustering
for iteration in range(params.iters):
    # Client-side computations
    masked_sums = []
    masked_counts = []
    for client in clients:
        sum_i, count_i, _ = client.step()
        masked_sums.append(sum_i)
        masked_counts.append(count_i)
    
    # Server-side aggregation
    agg_sums, agg_counts = server.step(masked_sums, masked_counts)
    
    # Update clients with aggregated results
    for client in clients:
        client.update(agg_sums, agg_counts)
```

## Implementation Details

### Server Class
- Implements differential privacy mechanisms
- Handles privacy budget allocation
- Performs secure aggregation

### UnmaskedClient Class
- Performs local clustering computations
- Manages centroid updates
- Implements bounded update constraints
- Handles data preprocessing and centroid post-processing

### MaskedClient Class
- Extends UnmaskedClient with privacy features
- Implements secure masking protocols
- Manages mask generation and application

## Security and Privacy Guarantees

The package provides several key security and privacy guarantees:

1. Differential Privacy
   - Formal ε or (ε,δ) privacy guarantees
   - Automatic sensitivity scaling
   - Dynamic privacy budget allocation

2. Secure Aggregation
   - Zero-knowledge protocol through masking
   - Information-theoretic security guarantees
   - Protection against curious servers

3. Data Protection
   - No raw data leaves client devices
   - Bounded updates prevent information leakage
   - Secure random number generation