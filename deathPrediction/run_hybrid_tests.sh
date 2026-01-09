#!/bin/bash
# Script to run all hybrid implementations and collect results
# Usage: ./run_hybrid_tests.sh

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "  Hybrid Implementations Test Runner"
echo "=========================================="
echo ""

# Check if data file exists
if [ ! -f "mimic_data.csv" ]; then
    echo "ERROR: mimic_data.csv not found!"
    exit 1
fi

# Check if executables exist
MISSING=0
for exe in hybrid_openmp_mpi_death_pred hybrid_pthread_mpi_death_pred hybrid_openmp_pthread_death_pred hybrid_mpi_openmp_pthread_death_pred; do
    if [ ! -f "$exe" ]; then
        echo "WARNING: $exe not found"
        MISSING=1
    fi
done

if [ $MISSING -eq 1 ]; then
    echo ""
    echo "Building all implementations..."
    make all
    echo ""
fi

# Set OpenMP thread count
export OMP_NUM_THREADS=2

echo "Running hybrid implementations..."
echo "Dataset: mimic_data.csv"
echo "MPI Processes: 2"
echo "OpenMP Threads: $OMP_NUM_THREADS"
echo "=========================================="
echo ""

# 1. OpenMP + MPI Hybrid
echo "[1/5] Running Serial Implementation (baseline)..."
./serial_death_pred mimic_data.csv
echo ""

# 2. OpenMP + MPI Hybrid
echo "[2/5] Running OpenMP + MPI Hybrid..."
mpirun --allow-run-as-root --oversubscribe -np 2 ./hybrid_openmp_mpi_death_pred mimic_data.csv
echo ""

# 3. Pthread + MPI Hybrid
echo "[3/5] Running Pthread + MPI Hybrid..."
mpirun --allow-run-as-root --oversubscribe -np 2 ./hybrid_pthread_mpi_death_pred mimic_data.csv 4
echo ""

# 4. OpenMP + Pthread Hybrid
echo "[4/5] Running OpenMP + Pthread Hybrid..."
./hybrid_openmp_pthread_death_pred mimic_data.csv 2
echo ""

# 5. Triple Hybrid
echo "[5/5] Running Triple Hybrid (MPI + OpenMP + Pthread)..."
mpirun --allow-run-as-root --oversubscribe -np 2 ./hybrid_mpi_openmp_pthread_death_pred mimic_data.csv 2
echo ""

echo "=========================================="
echo "  All tests completed successfully!"
echo "=========================================="
echo ""
echo "Timing files generated:"
ls -lh timing_*.txt 2>/dev/null || echo "No timing files found"
echo ""
echo "For detailed analysis, see: HYBRID_RESULTS.md"
echo "=========================================="
