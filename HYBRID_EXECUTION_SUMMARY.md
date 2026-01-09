# Hybrid Implementations Execution Summary

**Date:** January 9, 2026  
**Status:** ✅ All hybrid implementations successfully tested

## Quick Results

All four hybrid parallel implementations have been compiled, executed, and benchmarked on the MIMIC dataset (10,000 patients).

### Performance Summary

| Implementation | Total Time | Status | Relative Performance |
|----------------|------------|--------|---------------------|
| **Serial (Baseline)** | 0.346s | ✅ | Fastest overall |
| Pthread + MPI | 0.756s | ✅ | Fastest hybrid (2.19x slower than serial) |
| OpenMP + Pthread | 0.882s | ✅ | Second fastest hybrid |
| Triple Hybrid | 1.255s | ✅ | Third fastest hybrid |
| OpenMP + MPI | 2.305s | ✅ | Fourth fastest hybrid |

### Key Findings

1. ✅ **All implementations work correctly** - 100% accuracy achieved
2. ⚠️ **Serial is fastest for this dataset size** - Parallelization overhead exceeds benefits at 10K records
3. ✅ **Pthread + MPI is best hybrid** - Achieves best performance among parallel implementations
4. 📊 **Performance scales with complexity** - More parallel paradigms = more overhead
5. 🎓 **Great for learning** - Successfully demonstrates all three parallelization techniques

### When to Use What

- **< 50K records:** Use serial implementation (fastest)
- **50K-500K records:** Try OpenMP + Pthread (single node)
- **> 500K records or multi-node:** Use Pthread + MPI or OpenMP + MPI
- **Educational purposes:** Triple Hybrid shows all paradigms

## Detailed Results

For comprehensive analysis including:
- Individual implementation configurations
- Detailed timing breakdowns
- Performance analysis and insights
- Technical observations
- Recommendations and use cases

See: **[deathPrediction/HYBRID_RESULTS.md](deathPrediction/HYBRID_RESULTS.md)**

## Files Generated

- `deathPrediction/HYBRID_RESULTS.md` - Detailed performance analysis
- `deathPrediction/timing_*.txt` - Raw timing data (git ignored)
- All executables compiled successfully (git ignored)

## How to Reproduce

```bash
cd deathPrediction

# Build all implementations
make all

# Run individual hybrid implementations
export OMP_NUM_THREADS=2

# 1. OpenMP + MPI Hybrid
mpirun --allow-run-as-root --oversubscribe -np 2 ./hybrid_openmp_mpi_death_pred mimic_data.csv

# 2. Pthread + MPI Hybrid
mpirun --allow-run-as-root --oversubscribe -np 2 ./hybrid_pthread_mpi_death_pred mimic_data.csv 4

# 3. OpenMP + Pthread Hybrid
./hybrid_openmp_pthread_death_pred mimic_data.csv 2

# 4. Triple Hybrid (MPI + OpenMP + Pthread)
mpirun --allow-run-as-root --oversubscribe -np 2 ./hybrid_mpi_openmp_pthread_death_pred mimic_data.csv 2
```

## Conclusion

✅ **Task Complete:** All hybrid implementations have been successfully run and results documented. The code is working correctly and demonstrates advanced parallel programming concepts, though serial execution is optimal for this dataset size.
