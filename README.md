# ParallelMMIC

A comprehensive implementation comparing serial and parallel approaches (OpenMP, Pthreads, MPI, and Hybrid combinations) for death rate prediction using logistic regression on MIMIC clinical dataset.

## Overview

This project implements a Logistic Regression model to predict patient mortality based on:
- Ethnicity
- Gender
- ICD9 Primary Diagnosis Code

## Implementations

### Basic Parallel Implementations
1. **Serial** - Baseline single-threaded implementation
2. **OpenMP** - Shared-memory parallel processing using compiler directives
3. **Pthreads** - POSIX threads for fine-grained control
4. **MPI** - Distributed memory parallelization across processes
5. **CUDA** - GPU-accelerated implementation using NVIDIA CUDA for massively parallel computing

### Hybrid Parallel Implementations
5. **OpenMP + MPI Hybrid** - Combines distributed memory (MPI) across nodes with shared memory (OpenMP) within each node
   - **Use Case**: Industry-standard approach for HPC clusters
   - **Benefits**: Maximum efficiency on multi-node systems with multi-core processors
   
6. **Pthread + MPI Hybrid** - Combines MPI with fine-grained thread control using Pthreads
   - **Use Case**: When you need more control over thread behavior than OpenMP provides
   - **Benefits**: Explicit thread management and scheduling control within MPI processes
   
7. **OpenMP + Pthread Hybrid** - Nested parallelism combining OpenMP's ease-of-use with Pthread's fine-grained control
   - **Use Case**: When different parts of the algorithm benefit from different threading models
   - **Benefits**: Combines high-level parallel constructs with low-level thread control
   
8. **Triple Hybrid (MPI + OpenMP + Pthread)** - Three-level parallelism combining all techniques
   - **Use Case**: Maximum parallelism extraction from hierarchical hardware (cluster → node → core)
   - **Benefits**: Demonstrates advanced parallel programming concepts and maximum resource utilization

## Project Structure

```
deathPrediction/
├── serial_death_pred.cpp                    # Serial implementation
├── openmp_death_pred.cpp                    # OpenMP parallel implementation
├── pthread_death_pred.cpp                   # Pthreads parallel implementation
├── mpi_death_pred.cpp                       # MPI distributed implementation
├── cuda_death_pred.cu                       # CUDA GPU-accelerated implementation
├── hybrid_openmp_mpi_death_pred.cpp         # OpenMP+MPI hybrid
├── hybrid_pthread_mpi_death_pred.cpp        # Pthread+MPI hybrid
├── hybrid_openmp_pthread_death_pred.cpp     # OpenMP+Pthread hybrid
├── hybrid_mpi_openmp_pthread_death_pred.cpp # Triple hybrid (MPI+OpenMP+Pthread)
├── comparison_runner.cpp                    # Performance comparison script
├── makefile                                 # Build automation
└── mimic_data.csv                          # Sample dataset
```

## Requirements

- **Compiler**: g++ with C++11 support
- **OpenMP**: Usually included with g++
- **Pthreads**: POSIX threads library (pthread)
- **MPI**: OpenMPI or MPICH implementation
- **CUDA** (optional): NVIDIA CUDA Toolkit with nvcc compiler for GPU acceleration

## Installation

### Install MPI (if not already installed)

**Ubuntu/Debian:**
```bash
sudo apt-get install openmpi-bin libopenmpi-dev
```

**macOS:**
```bash
brew install open-mpi
```

**CentOS/RHEL:**
```bash
sudo yum install openmpi openmpi-devel
```

### Install CUDA (optional, for GPU acceleration)

**Ubuntu/Debian:**
```bash
# Install NVIDIA drivers first, then CUDA Toolkit
# Visit https://developer.nvidia.com/cuda-downloads for specific instructions
```

**Note:** CUDA requires an NVIDIA GPU with compute capability 3.0 or higher.

## Building

Build all implementations:
```bash
cd deathPrediction
make all
```

Build individual implementations:
```bash
make serial              # Build serial version
make openmp              # Build OpenMP version
make pthread             # Build Pthread version
make mpi                 # Build MPI version
make cuda                # Build CUDA version (requires CUDA Toolkit)
make hybrid_omp_mpi      # Build OpenMP+MPI hybrid
make hybrid_pthread_mpi  # Build Pthread+MPI hybrid
make hybrid_omp_pthread  # Build OpenMP+Pthread hybrid
make hybrid_triple       # Build triple hybrid
```

Clean build artifacts:
```bash
make clean
```

## Running

### Basic Implementations

**Serial:**
```bash
./serial_death_pred mimic_data.csv
```

**OpenMP:**
```bash
./openmp_death_pred mimic_data.csv
# Set number of threads (optional)
export OMP_NUM_THREADS=4
./openmp_death_pred mimic_data.csv
```

**Pthreads:**
```bash
./pthread_death_pred mimic_data.csv 4  # 4 = number of threads
```

**MPI:**
```bash
mpirun -np 4 ./mpi_death_pred mimic_data.csv  # 4 = number of processes
```

**CUDA:**
```bash
./cuda_death_pred mimic_data.csv  # Requires NVIDIA GPU with CUDA support
```

### Hybrid Implementations

**OpenMP + MPI Hybrid:**
```bash
# 2 MPI processes, each using OpenMP threads (controlled by OMP_NUM_THREADS)
export OMP_NUM_THREADS=4
mpirun -np 2 ./hybrid_openmp_mpi_death_pred mimic_data.csv
```

**Pthread + MPI Hybrid:**
```bash
# 2 MPI processes, each with 4 Pthreads
mpirun -np 2 ./hybrid_pthread_mpi_death_pred mimic_data.csv 4
```

**OpenMP + Pthread Hybrid:**
```bash
# OpenMP threads (controlled by OMP_NUM_THREADS), each using 2 Pthreads
export OMP_NUM_THREADS=4
./hybrid_openmp_pthread_death_pred mimic_data.csv 2
```

**Triple Hybrid (MPI + OpenMP + Pthread):**
```bash
# 2 MPI processes, each using OpenMP threads, each OpenMP thread using 2 Pthreads
export OMP_NUM_THREADS=2
mpirun -np 2 ./hybrid_mpi_openmp_pthread_death_pred mimic_data.csv 2
```

### Performance Comparison

Run all implementations and compare performance:
```bash
./comparison_runner mimic_data.csv 4
```

This will generate:
- Individual timing files (`timing_*.txt`)
- Comprehensive performance report (`performance_report.txt`)

### Hybrid Implementations Results

For detailed performance analysis of all hybrid implementations, see:
- **[HYBRID_EXECUTION_SUMMARY.md](HYBRID_EXECUTION_SUMMARY.md)** - Quick summary and reproduction steps
- **[deathPrediction/HYBRID_RESULTS.md](deathPrediction/HYBRID_RESULTS.md)** - Comprehensive performance analysis

To run all hybrid implementations:
```bash
cd deathPrediction
./run_hybrid_tests.sh
```

## Performance Characteristics

### Expected Speedup Patterns

1. **Serial**: Baseline performance (1x speedup)
2. **OpenMP**: Near-linear speedup up to the number of cores (e.g., 3.5x on 4 cores)
3. **Pthreads**: Similar to OpenMP but with slight overhead for manual thread management
4. **MPI**: Good speedup for distributed workloads, best on multiple nodes
5. **CUDA**: Exceptional speedup on large datasets (10x-100x on suitable GPUs for >100K records)
6. **Hybrid implementations**: Superior performance on multi-node, multi-core systems

### When to Use Each Implementation

| Implementation | Best Use Case |
|---------------|---------------|
| Serial | Small datasets, debugging, baseline comparison |
| OpenMP | Multi-core single node, easy parallelization |
| Pthreads | Fine-grained control needed, specific thread scheduling |
| MPI | Multiple nodes, distributed memory systems |
| CUDA | Large datasets (>100K records), NVIDIA GPU available |
| OpenMP+MPI | HPC clusters with multi-core nodes (most common hybrid) |
| Pthread+MPI | Need explicit thread control in distributed system |
| OpenMP+Pthread | Different algorithm phases need different threading models |
| Triple Hybrid | Maximum parallelism on complex hierarchical systems |

## Technical Considerations

### Thread Safety
- All hybrid implementations ensure proper synchronization between threading models
- Critical sections are protected with appropriate locks (pthread mutexes, OpenMP critical sections)
- MPI communication occurs at synchronization points

### Avoiding Oversubscription
To avoid creating too many threads:
```bash
# For hybrid implementations, balance MPI processes and threads
# Total threads = MPI_processes × OpenMP_threads × Pthreads
# Should not exceed: number_of_physical_cores × 2 (if hyperthreading)

# Example for 8-core system:
export OMP_NUM_THREADS=2
mpirun -np 2 ./hybrid_mpi_openmp_pthread_death_pred mimic_data.csv 2
# Total: 2 × 2 × 2 = 8 threads
```

### Performance Tuning

**OpenMP:**
- Adjust `OMP_NUM_THREADS` to match available cores
- Consider `OMP_SCHEDULE` for load balancing

**MPI:**
- Balance number of processes with data size
- Use process binding for NUMA systems: `mpirun --bind-to core -np 4 ...`

**Hybrid:**
- Start with fewer MPI processes and more threads per process
- Profile to find optimal balance for your hardware

## Comparison Metrics

The comparison runner provides:
- **Load Time**: Time to read and parse the CSV file
- **Training Time**: Time for gradient descent optimization
- **Evaluation Time**: Time to compute accuracy and death rate predictions
- **Total Time**: Overall execution time
- **Speedup**: Serial time / Parallel time
- **Efficiency**: (Speedup / Number of threads) × 100%
- **Accuracy**: Model prediction accuracy on test set
- **Death Rate**: Predicted mortality rate

## Dataset

The provided `mimic_data.csv` is a synthetic dataset generated based on the actual MIMIC clinical dataset. The original dataset is not available due to privacy reasons.

**Format:**
```
SUBJECT_ID,HADM_ID,HOSPITAL,EXPIRE_FLAG,ADMITTIME,ETHNICITY,GENDER,DOB,AGE_AT_ADMISSION,ICD9_CODE_1
```

## Example Output

```
Hybrid OpenMP+MPI Implementation - Logistic Regression
=======================================================
MPI Processes: 2
OpenMP Threads per process: 4
Data loaded: 10000 patients
Load time: 0.0523 seconds
Training time: 1.2341 seconds
Evaluation time: 0.0156 seconds
Total execution time: 1.3020 seconds

Results:
Accuracy: 85.43%
Predicted Death Rate: 12.67%
```

## Troubleshooting

**MPI not found:**
```bash
# Check if MPI is installed
which mpirun mpic++

# If not, install it (Ubuntu):
sudo apt-get install openmpi-bin libopenmpi-dev
```

**OpenMP not working:**
```bash
# Verify OpenMP support
echo "#include <omp.h>" | g++ -fopenmp -x c++ -
```

**Permission denied:**
```bash
chmod +x serial_death_pred openmp_death_pred pthread_death_pred mpi_death_pred
```

## Contributing

This is an educational project demonstrating various parallel computing paradigms. Contributions are welcome!

## License

Educational use only.

## References

- OpenMP: https://www.openmp.org/
- MPI: https://www.mpi-forum.org/
- POSIX Threads: https://pubs.opengroup.org/onlinepubs/9699919799/
- MIMIC Dataset: https://mimic.physionet.org/
