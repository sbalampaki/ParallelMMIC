# Makefile for Death Rate Prediction - Parallel Implementations
# Usage: make all

CXX = g++
MPICXX = mpic++
NVCC = nvcc
CXXFLAGS = -std=c++11 -O3 -Wall
OMPFLAGS = -fopenmp
PTHREADFLAGS = -pthread
CUDAFLAGS = -std=c++11 -O3

# Executable names
SERIAL = serial_death_pred
OPENMP = openmp_death_pred
PTHREAD = pthread_death_pred
MPI = mpi_death_pred
CUDA = cuda_death_pred
HYBRID_OMP_MPI = hybrid_openmp_mpi_death_pred
HYBRID_PTHREAD_MPI = hybrid_pthread_mpi_death_pred
HYBRID_OMP_PTHREAD = hybrid_openmp_pthread_death_pred
HYBRID_TRIPLE = hybrid_mpi_openmp_pthread_death_pred
RUNNER = comparison_runner

# Decision Tree executables
DT_SERIAL = serial_decision_tree_death_pred

# Gradient Boosting executables
GB_SERIAL = serial_gradient_boosting_death_pred
GB_OPENMP = openmp_gradient_boosting_death_pred

# Default target (excluding CUDA due to NVIDIA GPU hardware dependency)
all: $(SERIAL) $(OPENMP) $(PTHREAD) $(MPI) $(HYBRID_OMP_MPI) $(HYBRID_PTHREAD_MPI) $(HYBRID_OMP_PTHREAD) $(HYBRID_TRIPLE) $(RUNNER) $(DT_SERIAL) $(GB_SERIAL) $(GB_OPENMP)

# Target to build all including CUDA (only if nvcc is available)
all_with_cuda: all
	@if command -v nvcc >/dev/null 2>&1; then \
		$(MAKE) $(CUDA); \
	else \
		echo "CUDA compiler (nvcc) not found. Skipping CUDA build."; \
		echo "Install CUDA Toolkit to enable CUDA implementation."; \
	fi
	@echo ""
	@echo "=========================================="
	@echo "  All implementations compiled successfully!"
	@echo "=========================================="
	@echo ""
	@echo "To run the complete comparison:"
	@echo "  ./$(RUNNER) <data_file.csv> [num_threads]"
	@echo ""
	@echo "Example:"
	@echo "  ./$(RUNNER) mimic_data.csv 4"
	@echo ""
	@echo "Or run individually:"
	@echo "  ./$(SERIAL) mimic_data.csv"
	@echo "  ./$(OPENMP) mimic_data.csv"
	@echo "  ./$(PTHREAD) mimic_data.csv 4"
	@echo "  mpirun -np 4 ./$(MPI) mimic_data.csv"
	@echo ""
	@echo "CUDA implementation (requires CUDA Toolkit):"
	@echo "  make cuda           # Build CUDA implementation"
	@echo "  ./$(CUDA) mimic_data.csv"
	@echo ""
	@echo "Hybrid implementations:"
	@echo "  mpirun -np 2 ./$(HYBRID_OMP_MPI) mimic_data.csv"
	@echo "  mpirun -np 2 ./$(HYBRID_PTHREAD_MPI) mimic_data.csv 4"
	@echo "  ./$(HYBRID_OMP_PTHREAD) mimic_data.csv 2"
	@echo "  mpirun -np 2 ./$(HYBRID_TRIPLE) mimic_data.csv 2"
	@echo "=========================================="

# Serial implementation
$(SERIAL): serial_death_pred.cpp
	@echo "Compiling serial implementation..."
	$(CXX) $(CXXFLAGS) -o $(SERIAL) serial_death_pred.cpp

# OpenMP implementation
$(OPENMP): openmp_death_pred.cpp
	@echo "Compiling OpenMP implementation..."
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o $(OPENMP) openmp_death_pred.cpp

# Pthreads implementation
$(PTHREAD): pthread_death_pred.cpp
	@echo "Compiling Pthreads implementation..."
	$(CXX) $(CXXFLAGS) $(PTHREADFLAGS) -o $(PTHREAD) pthread_death_pred.cpp

# MPI implementation
$(MPI): mpi_death_pred.cpp
	@echo "Compiling MPI implementation..."
	$(MPICXX) $(CXXFLAGS) -o $(MPI) mpi_death_pred.cpp

# CUDA implementation
$(CUDA): cuda_death_pred.cu
	@echo "Compiling CUDA implementation..."
	@if command -v nvcc >/dev/null 2>&1; then \
		$(NVCC) $(CUDAFLAGS) -o $(CUDA) cuda_death_pred.cu; \
	else \
		echo "Error: CUDA compiler (nvcc) not found."; \
		echo "Please install CUDA Toolkit to build CUDA implementation."; \
		exit 1; \
	fi

# Hybrid OpenMP+MPI implementation
$(HYBRID_OMP_MPI): hybrid_openmp_mpi_death_pred.cpp
	@echo "Compiling Hybrid OpenMP+MPI implementation..."
	$(MPICXX) $(CXXFLAGS) $(OMPFLAGS) -o $(HYBRID_OMP_MPI) hybrid_openmp_mpi_death_pred.cpp

# Hybrid Pthread+MPI implementation
$(HYBRID_PTHREAD_MPI): hybrid_pthread_mpi_death_pred.cpp
	@echo "Compiling Hybrid Pthread+MPI implementation..."
	$(MPICXX) $(CXXFLAGS) $(PTHREADFLAGS) -o $(HYBRID_PTHREAD_MPI) hybrid_pthread_mpi_death_pred.cpp

# Hybrid OpenMP+Pthread implementation
$(HYBRID_OMP_PTHREAD): hybrid_openmp_pthread_death_pred.cpp
	@echo "Compiling Hybrid OpenMP+Pthread implementation..."
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $(PTHREADFLAGS) -o $(HYBRID_OMP_PTHREAD) hybrid_openmp_pthread_death_pred.cpp

# Triple Hybrid MPI+OpenMP+Pthread implementation
$(HYBRID_TRIPLE): hybrid_mpi_openmp_pthread_death_pred.cpp
	@echo "Compiling Triple Hybrid MPI+OpenMP+Pthread implementation..."
	$(MPICXX) $(CXXFLAGS) $(OMPFLAGS) $(PTHREADFLAGS) -o $(HYBRID_TRIPLE) hybrid_mpi_openmp_pthread_death_pred.cpp

# Comparison runner
$(RUNNER): comparison_runner.cpp
	@echo "Compiling comparison runner..."
	$(CXX) $(CXXFLAGS) -o $(RUNNER) comparison_runner.cpp

# Decision Tree - Serial implementation
$(DT_SERIAL): serial_decision_tree_death_pred.cpp
	@echo "Compiling Decision Tree serial implementation..."
	$(CXX) $(CXXFLAGS) -o $(DT_SERIAL) serial_decision_tree_death_pred.cpp

# Gradient Boosting - Serial implementation
$(GB_SERIAL): serial_gradient_boosting_death_pred.cpp
	@echo "Compiling Gradient Boosting serial implementation..."
	$(CXX) $(CXXFLAGS) -o $(GB_SERIAL) serial_gradient_boosting_death_pred.cpp

# Gradient Boosting - OpenMP implementation
$(GB_OPENMP): openmp_gradient_boosting_death_pred.cpp
	@echo "Compiling Gradient Boosting OpenMP implementation..."
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o $(GB_OPENMP) openmp_gradient_boosting_death_pred.cpp

# Individual targets
serial: $(SERIAL)
openmp: $(OPENMP)
pthread: $(PTHREAD)
mpi: $(MPI)
cuda: $(CUDA)
hybrid_omp_mpi: $(HYBRID_OMP_MPI)
hybrid_pthread_mpi: $(HYBRID_PTHREAD_MPI)
hybrid_omp_pthread: $(HYBRID_OMP_PTHREAD)
hybrid_triple: $(HYBRID_TRIPLE)
runner: $(RUNNER)
dt_serial: $(DT_SERIAL)
gb_serial: $(GB_SERIAL)
gb_openmp: $(GB_OPENMP)

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -f $(SERIAL) $(OPENMP) $(PTHREAD) $(MPI) $(CUDA) $(HYBRID_OMP_MPI) $(HYBRID_PTHREAD_MPI) $(HYBRID_OMP_PTHREAD) $(HYBRID_TRIPLE) $(RUNNER) $(DT_SERIAL) $(GB_SERIAL) $(GB_OPENMP)
	rm -f timing_*.txt performance_report.txt
	@echo "Clean complete."

# Run tests
test: all
	@echo "Testing with sample data..."
	@if [ ! -f mimic_data.csv ]; then \
		echo "Error: mimic_data.csv not found!"; \
		echo "Please prepare your CSV file with the following columns:"; \
		echo "SUBJECT_ID,HADM_ID,HOSPITAL,EXPIRE_FLAG,ADMITTIME,ETHNICITY,GENDER,DOB,AGE_AT_ADMISSION,ICD9_CODE_1"; \
		exit 1; \
	fi
	./$(RUNNER) mimic_data.csv 4

# Help target
help:
	@echo "Death Rate Prediction - Parallel Implementation Build System"
	@echo ""
	@echo "Available targets:"
	@echo "  make all                    - Build all implementations and runner"
	@echo "  make serial                 - Build only serial implementation"
	@echo "  make openmp                 - Build only OpenMP implementation"
	@echo "  make pthread                - Build only Pthreads implementation"
	@echo "  make mpi                    - Build only MPI implementation"
	@echo "  make cuda                   - Build only CUDA implementation (requires CUDA Toolkit)"
	@echo "  make hybrid_omp_mpi         - Build only OpenMP+MPI hybrid"
	@echo "  make hybrid_pthread_mpi     - Build only Pthread+MPI hybrid"
	@echo "  make hybrid_omp_pthread     - Build only OpenMP+Pthread hybrid"
	@echo "  make hybrid_triple          - Build only MPI+OpenMP+Pthread triple hybrid"
	@echo "  make runner                 - Build only comparison runner"
	@echo "  make dt_serial              - Build only Decision Tree serial implementation"
	@echo "  make gb_serial              - Build only Gradient Boosting serial implementation"
	@echo "  make gb_openmp              - Build only Gradient Boosting OpenMP implementation"
	@echo "  make clean                  - Remove all build artifacts"
	@echo "  make test                   - Build and run tests with mimic_data.csv"
	@echo "  make help                   - Show this help message"
	@echo ""
	@echo "Requirements:"
	@echo "  - g++ compiler with C++11 support"
	@echo "  - OpenMP support (usually included with g++)"
	@echo "  - POSIX threads library (pthread)"
	@echo "  - MPI implementation (OpenMPI or MPICH)"
	@echo "  - CUDA Toolkit with nvcc (optional, for CUDA implementation)"
	@echo "  - Python 3 + PyTorch (for transformer: python transformer_death_pred.py)"
	@echo ""
	@echo "Data file format (CSV):"
	@echo "  SUBJECT_ID,HADM_ID,HOSPITAL,EXPIRE_FLAG,ADMITTIME,ETHNICITY,GENDER,DOB,AGE_AT_ADMISSION,ICD9_CODE_1"

.PHONY: all all_with_cuda serial openmp pthread mpi cuda hybrid_omp_mpi hybrid_pthread_mpi hybrid_omp_pthread hybrid_triple runner dt_serial gb_serial gb_openmp clean test help