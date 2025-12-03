# ParallelMMIC
A comprehensive implementation comparing serial and parallel approaches (OpenMP, Pthreads, MPI) for death
rate prediction using logistic regression on MIMIC clinical dataset.
This project implements a Logistic Regression model to predict patient mortality based on:
- Ethnicity
- Gender
- ICD9 Primary Diagnosis Code
The implementation includes:
1. Serial - Baseline single-threaded implementation
2. OpenMP - Shared-memory parallel processing
3. Pthreads - POSIX threads for fine-grained control
4. MPI - Distributed memory parallelization
Project Structure:
├── serial_death_pred.cpp # Serial implementation
├── openmp_death_pred.cpp # OpenMP parallel implementation
├── pthread_death_pred.cpp # Pthreads parallel implementation
├── mpi_death_pred.cpp # MPI distributed implementation
├── comparison_runner.cpp # Master comparison script
├── Makefile # Build automation
Steps to run:
1. Download all the files on your system.
2. Run command on terminal: make all
3. Run command on terminal: ./serial_death_pred mimic_data.csv
4. Run command on terminal: ./openmp_death_pred mimic_data.csv
5. Run command on terminal: ./pthread_death_pred mimic_data.csv 8
6. Run command on terminal: mpirun -np 4 ./mpi_death_pred mimic_data.csv
7. Run command on terminal: ./comparison_runner mimic_data.csv 4
Once the above steps are done, performance.txt file will be generated with all the comparisons on performance.

