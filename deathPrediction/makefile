# Makefile for Death Rate Prediction - Parallel Implementations
# Usage: make all

CXX = g++
MPICXX = mpic++
CXXFLAGS = -std=c++11 -O3 -Wall
OMPFLAGS = -fopenmp
PTHREADFLAGS = -pthread

# Executable names
SERIAL = serial_death_pred
OPENMP = openmp_death_pred
PTHREAD = pthread_death_pred
MPI = mpi_death_pred
RUNNER = comparison_runner

# Default target
all: $(SERIAL) $(OPENMP) $(PTHREAD) $(MPI) $(RUNNER)
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

# Comparison runner
$(RUNNER): comparison_runner.cpp
	@echo "Compiling comparison runner..."
	$(CXX) $(CXXFLAGS) -o $(RUNNER) comparison_runner.cpp

# Individual targets
serial: $(SERIAL)
openmp: $(OPENMP)
pthread: $(PTHREAD)
mpi: $(MPI)
runner: $(RUNNER)

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -f $(SERIAL) $(OPENMP) $(PTHREAD) $(MPI) $(RUNNER)
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
	@echo "  make all       - Build all implementations and runner"
	@echo "  make serial    - Build only serial implementation"
	@echo "  make openmp    - Build only OpenMP implementation"
	@echo "  make pthread   - Build only Pthreads implementation"
	@echo "  make mpi       - Build only MPI implementation"
	@echo "  make runner    - Build only comparison runner"
	@echo "  make clean     - Remove all build artifacts"
	@echo "  make test      - Build and run tests with mimic_data.csv"
	@echo "  make help      - Show this help message"
	@echo ""
	@echo "Requirements:"
	@echo "  - g++ compiler with C++11 support"
	@echo "  - OpenMP support (usually included with g++)"
	@echo "  - POSIX threads library (pthread)"
	@echo "  - MPI implementation (OpenMPI or MPICH)"
	@echo ""
	@echo "Data file format (CSV):"
	@echo "  SUBJECT_ID,HADM_ID,HOSPITAL,EXPIRE_FLAG,ADMITTIME,ETHNICITY,GENDER,DOB,AGE_AT_ADMISSION,ICD9_CODE_1"

.PHONY: all serial openmp pthread mpi runner clean test help