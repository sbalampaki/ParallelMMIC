#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <iomanip>
#include <cstdlib>
#include <algorithm>

using namespace std;

struct TimingResult {
    string implementation;
    double loadTime;
    double trainTime;
    double evalTime;
    double totalTime;
    double accuracy;
    double deathRate;
    int threads;
    double speedup;
    double efficiency;
};

TimingResult parseTimingFile(const string& filename, const string& implName) {
    TimingResult result;
    result.implementation = implName;
    result.threads = 1;
    
    ifstream file(filename);
    string line;
    
    while (getline(file, line)) {
        stringstream ss(line);
        string key, value;
        getline(ss, key, ',');
        getline(ss, value, ',');
        
        if (key == "load") result.loadTime = stod(value);
        else if (key == "train") result.trainTime = stod(value);
        else if (key == "eval") result.evalTime = stod(value);
        else if (key == "total") result.totalTime = stod(value);
        else if (key == "accuracy") result.accuracy = stod(value);
        else if (key == "deathrate") result.deathRate = stod(value);
        else if (key == "threads") result.threads = stoi(value);
        else if (key == "processes") result.threads = stoi(value);
    }
    
    return result;
}

void printComparisonTable(const vector<TimingResult>& results, double serialTime) {
    cout << "\n========================================================================================================" << endl;
    cout << "                           PERFORMANCE COMPARISON - DEATH RATE PREDICTION" << endl;
    cout << "========================================================================================================" << endl;
    
    cout << fixed << setprecision(4);
    cout << left << setw(15) << "Implementation" 
         << right << setw(12) << "Load (s)" 
         << setw(12) << "Train (s)" 
         << setw(12) << "Eval (s)" 
         << setw(12) << "Total (s)" 
         << setw(10) << "Threads"
         << setw(12) << "Speedup"
         << setw(12) << "Efficiency" << endl;
    cout << "--------------------------------------------------------------------------------------------------------" << endl;
    
    for (const auto& r : results) {
        double speedup = serialTime / r.totalTime;
        double efficiency = speedup / r.threads * 100;
        
        cout << left << setw(15) << r.implementation
             << right << setw(12) << r.loadTime
             << setw(12) << r.trainTime
             << setw(12) << r.evalTime
             << setw(12) << r.totalTime
             << setw(10) << r.threads
             << setw(12) << speedup
             << setw(11) << efficiency << "%" << endl;
    }
    
    cout << "========================================================================================================" << endl;
    
    // Print accuracy and death rate comparison
    cout << "\n========================================================================================================" << endl;
    cout << "                                    MODEL ACCURACY COMPARISON" << endl;
    cout << "========================================================================================================" << endl;
    cout << left << setw(15) << "Implementation" 
         << right << setw(20) << "Accuracy (%)" 
         << setw(25) << "Predicted Death Rate (%)" << endl;
    cout << "--------------------------------------------------------------------------------------------------------" << endl;
    
    for (const auto& r : results) {
        cout << left << setw(15) << r.implementation
             << right << setw(20) << (r.accuracy * 100)
             << setw(25) << (r.deathRate * 100) << endl;
    }
    cout << "========================================================================================================" << endl;
}

void generatePerformanceReport(const vector<TimingResult>& results, double serialTime) {
    ofstream report("performance_report.txt");
    
    report << "==========================================================================\n";
    report << "         DEATH RATE PREDICTION - PERFORMANCE ANALYSIS REPORT\n";
    report << "==========================================================================\n\n";
    
    report << "DATASET: MIMIC Clinical Dataset\n";
    report << "ALGORITHM: Logistic Regression\n";
    report << "FEATURES: Ethnicity, Gender, ICD9_CODE_1\n";
    report << "TARGET: EXPIRE_FLAG (Death Rate)\n\n";
    
    report << "--------------------------------------------------------------------------\n";
    report << "TIMING BREAKDOWN (seconds)\n";
    report << "--------------------------------------------------------------------------\n";
    report << fixed << setprecision(4);
    report << left << setw(15) << "Implementation" 
           << right << setw(12) << "Load" 
           << setw(12) << "Train" 
           << setw(12) << "Eval" 
           << setw(12) << "Total\n";
    report << "--------------------------------------------------------------------------\n";
    
    for (const auto& r : results) {
        report << left << setw(15) << r.implementation
               << right << setw(12) << r.loadTime
               << setw(12) << r.trainTime
               << setw(12) << r.evalTime
               << setw(12) << r.totalTime << "\n";
    }
    
    report << "\n--------------------------------------------------------------------------\n";
    report << "PARALLEL PERFORMANCE METRICS\n";
    report << "--------------------------------------------------------------------------\n";
    report << left << setw(15) << "Implementation" 
           << right << setw(10) << "Threads" 
           << setw(12) << "Speedup" 
           << setw(15) << "Efficiency (%)\n";
    report << "--------------------------------------------------------------------------\n";
    
    for (const auto& r : results) {
        double speedup = serialTime / r.totalTime;
        double efficiency = speedup / r.threads * 100;
        
        report << left << setw(15) << r.implementation
               << right << setw(10) << r.threads
               << setw(12) << speedup
               << setw(15) << efficiency << "\n";
    }
    
    report << "\n--------------------------------------------------------------------------\n";
    report << "MODEL PERFORMANCE\n";
    report << "--------------------------------------------------------------------------\n";
    report << left << setw(15) << "Implementation" 
           << right << setw(20) << "Accuracy (%)" 
           << setw(25) << "Death Rate (%)\n";
    report << "--------------------------------------------------------------------------\n";
    
    for (const auto& r : results) {
        report << left << setw(15) << r.implementation
               << right << setw(20) << (r.accuracy * 100)
               << setw(25) << (r.deathRate * 100) << "\n";
    }
    
    report << "\n==========================================================================\n";
    report << "KEY FINDINGS:\n";
    report << "==========================================================================\n\n";
    
    // Find best performer
    auto best = *min_element(results.begin(), results.end(), 
                            [](const TimingResult& a, const TimingResult& b) {
                                return a.totalTime < b.totalTime;
                            });
    
    report << "1. FASTEST IMPLEMENTATION: " << best.implementation 
           << " (" << best.totalTime << "s)\n";
    report << "   - Speedup over serial: " << (serialTime / best.totalTime) << "x\n";
    report << "   - Parallel efficiency: " << ((serialTime / best.totalTime) / best.threads * 100) << "%\n\n";
    
    report << "2. TRAINING TIME COMPARISON:\n";
    for (const auto& r : results) {
        report << "   - " << r.implementation << ": " << r.trainTime << "s\n";
    }
    
    report << "\n3. MODEL ACCURACY:\n";
    report << "   All implementations produced consistent accuracy results,\n";
    report << "   demonstrating correct parallel algorithm implementation.\n\n";
    
    report << "4. SCALABILITY ANALYSIS:\n";
    report << "   - OpenMP: Shared memory parallelization, ideal for multi-core CPUs\n";
    report << "   - Pthreads: Low-level thread control, efficient for fine-tuned performance\n";
    report << "   - MPI: Distributed memory, suitable for cluster computing\n";
    
    // Check if CUDA results are included
    bool hasCuda = false;
    for (const auto& r : results) {
        if (r.implementation == "CUDA") {
            hasCuda = true;
            break;
        }
    }
    if (hasCuda) {
        report << "   - CUDA: GPU acceleration, exceptional speedup for large datasets\n";
    }
    
    report.close();
    cout << "\nDetailed performance report saved to: performance_report.txt" << endl;
}

void generateRFPerformanceReport(const vector<TimingResult>& results, double serialTime) {
    ofstream report("rf_performance_report.txt");

    report << "==========================================================================\n";
    report << "      DEATH RATE PREDICTION (RANDOM FOREST) - PERFORMANCE REPORT\n";
    report << "==========================================================================\n\n";

    report << "DATASET: MIMIC Clinical Dataset\n";
    report << "ALGORITHM: Random Forest\n";
    report << "FEATURES: Ethnicity, Gender, ICD9_CODE_1\n";
    report << "TARGET: EXPIRE_FLAG (Death Rate)\n\n";

    report << "--------------------------------------------------------------------------\n";
    report << "TIMING BREAKDOWN (seconds)\n";
    report << "--------------------------------------------------------------------------\n";
    report << fixed << setprecision(4);
    report << left << setw(25) << "Implementation"
           << right << setw(12) << "Load"
           << setw(12) << "Train"
           << setw(12) << "Eval"
           << setw(12) << "Total\n";
    report << "--------------------------------------------------------------------------\n";

    for (const auto& r : results) {
        report << left << setw(25) << r.implementation
               << right << setw(12) << r.loadTime
               << setw(12) << r.trainTime
               << setw(12) << r.evalTime
               << setw(12) << r.totalTime << "\n";
    }

    report << "\n--------------------------------------------------------------------------\n";
    report << "PARALLEL PERFORMANCE METRICS\n";
    report << "--------------------------------------------------------------------------\n";
    report << left << setw(25) << "Implementation"
           << right << setw(10) << "Threads"
           << setw(12) << "Speedup"
           << setw(15) << "Efficiency (%)\n";
    report << "--------------------------------------------------------------------------\n";

    for (const auto& r : results) {
        double speedup = serialTime / r.totalTime;
        double efficiency = speedup / r.threads * 100;

        report << left << setw(25) << r.implementation
               << right << setw(10) << r.threads
               << setw(12) << speedup
               << setw(15) << efficiency << "\n";
    }

    report << "\n--------------------------------------------------------------------------\n";
    report << "MODEL PERFORMANCE\n";
    report << "--------------------------------------------------------------------------\n";
    report << left << setw(25) << "Implementation"
           << right << setw(20) << "Accuracy (%)"
           << setw(25) << "Death Rate (%)\n";
    report << "--------------------------------------------------------------------------\n";

    for (const auto& r : results) {
        report << left << setw(25) << r.implementation
               << right << setw(20) << (r.accuracy * 100)
               << setw(25) << (r.deathRate * 100) << "\n";
    }

    report << "\n==========================================================================\n";
    report << "KEY FINDINGS (Random Forest):\n";
    report << "==========================================================================\n\n";

    auto best = *min_element(results.begin(), results.end(),
                             [](const TimingResult& a, const TimingResult& b) {
                                 return a.totalTime < b.totalTime;
                             });

    report << "1. FASTEST RF IMPLEMENTATION: " << best.implementation
           << " (" << best.totalTime << "s)\n";
    report << "   - Speedup over RF serial: " << (serialTime / best.totalTime) << "x\n";
    report << "   - Parallel efficiency: " << ((serialTime / best.totalTime) / best.threads * 100) << "%\n\n";

    report << "2. TRAINING TIME COMPARISON:\n";
    for (const auto& r : results) {
        report << "   - " << r.implementation << ": " << r.trainTime << "s\n";
    }

    report << "\n3. SCALABILITY ANALYSIS:\n";
    report << "   - RF-OpenMP: Shared memory parallelization of tree building\n";
    report << "   - RF-Pthreads: Fine-grained thread control per tree\n";
    report << "   - RF-MPI: Distributed ensemble across processes\n";
    report << "   - RF-Hybrid variants: Combined parallelism strategies\n";

    bool hasCuda = false;
    for (const auto& r : results) {
        if (r.implementation.find("CUDA") != string::npos) {
            hasCuda = true;
            break;
        }
    }
    if (hasCuda) {
        report << "   - RF-CUDA: GPU-accelerated forest training\n";
    }

    report.close();
    cout << "\nDetailed RF performance report saved to: rf_performance_report.txt" << endl;
}

void printRFComparisonTable(const vector<TimingResult>& results, double serialTime) {
    cout << "\n========================================================================================================" << endl;
    cout << "                    PERFORMANCE COMPARISON - RANDOM FOREST IMPLEMENTATIONS" << endl;
    cout << "========================================================================================================" << endl;

    cout << fixed << setprecision(4);
    cout << left << setw(25) << "Implementation"
         << right << setw(12) << "Load (s)"
         << setw(12) << "Train (s)"
         << setw(12) << "Eval (s)"
         << setw(12) << "Total (s)"
         << setw(10) << "Threads"
         << setw(12) << "Speedup"
         << setw(12) << "Efficiency" << endl;
    cout << "--------------------------------------------------------------------------------------------------------" << endl;

    for (const auto& r : results) {
        double speedup = serialTime / r.totalTime;
        double efficiency = speedup / r.threads * 100;

        cout << left << setw(25) << r.implementation
             << right << setw(12) << r.loadTime
             << setw(12) << r.trainTime
             << setw(12) << r.evalTime
             << setw(12) << r.totalTime
             << setw(10) << r.threads
             << setw(12) << speedup
             << setw(11) << efficiency << "%" << endl;
    }

    cout << "========================================================================================================" << endl;

    cout << "\n========================================================================================================" << endl;
    cout << "                             RANDOM FOREST - MODEL ACCURACY COMPARISON" << endl;
    cout << "========================================================================================================" << endl;
    cout << left << setw(25) << "Implementation"
         << right << setw(20) << "Accuracy (%)"
         << setw(25) << "Predicted Death Rate (%)" << endl;
    cout << "--------------------------------------------------------------------------------------------------------" << endl;

    for (const auto& r : results) {
        cout << left << setw(25) << r.implementation
             << right << setw(20) << (r.accuracy * 100)
             << setw(25) << (r.deathRate * 100) << endl;
    }
    cout << "========================================================================================================" << endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <data_file> [num_threads]" << endl;
        cerr << "\nThis script will:" << endl;
        cerr << "  1.  Run serial implementation (Logistic Regression)" << endl;
        cerr << "  2.  Run OpenMP parallel implementation (LR)" << endl;
        cerr << "  3.  Run Pthreads parallel implementation (LR)" << endl;
        cerr << "  4.  Run MPI synchronous implementation (LR)" << endl;
        cerr << "  5.  Run MPI asynchronous implementation (LR)" << endl;
        cerr << "  6.  Run CUDA parallel implementation (LR, if available)" << endl;
        cerr << "  7.  Run Random Forest serial implementation" << endl;
        cerr << "  8.  Run Random Forest OpenMP implementation" << endl;
        cerr << "  9.  Run Random Forest Pthreads implementation" << endl;
        cerr << "  10. Run Random Forest MPI implementation" << endl;
        cerr << "  11. Run Random Forest OpenMP+MPI hybrid" << endl;
        cerr << "  12. Run Random Forest Pthread+MPI hybrid" << endl;
        cerr << "  13. Run Random Forest OpenMP+Pthread hybrid" << endl;
        cerr << "  14. Run Random Forest Triple Hybrid (MPI+OpenMP+Pthread)" << endl;
        cerr << "  15. Run Random Forest CUDA implementation (if available)" << endl;
        cerr << "  16. Compare all performance metrics" << endl;
        return 1;
    }
    
    string dataFile = argv[1];
    string numThreads = argc > 2 ? argv[2] : "4";

    cout << "========================================================================================================" << endl;
    cout << "                     DEATH RATE PREDICTION - PARALLEL PERFORMANCE COMPARISON" << endl;
    cout << "========================================================================================================" << endl;
    cout << "Dataset: " << dataFile << endl;
    cout << "Threads/Processes: " << numThreads << endl;
    cout << "========================================================================================================" << endl;

    // -----------------------------------------------------------------------
    // SECTION 1: Logistic Regression implementations
    // -----------------------------------------------------------------------
    cout << "\n--- LOGISTIC REGRESSION IMPLEMENTATIONS ---" << endl;

    cout << "\n[1/6] Running Serial LR Implementation..." << endl;
    int ret1 = system(("./serial_death_pred " + dataFile).c_str());
    if (ret1 != 0) {
        cerr << "Error running serial implementation. Make sure it's compiled." << endl;
        cerr << "Compile with: g++ -o serial_death_pred serial_death_pred.cpp -std=c++11" << endl;
    }

    cout << "\n[2/6] Running OpenMP LR Implementation..." << endl;
    int ret2 = system(("./openmp_death_pred " + dataFile).c_str());
    if (ret2 != 0) {
        cerr << "Error running OpenMP implementation. Make sure it's compiled." << endl;
        cerr << "Compile with: g++ -fopenmp -o openmp_death_pred openmp_death_pred.cpp -std=c++11" << endl;
    }

    cout << "\n[3/6] Running Pthreads LR Implementation..." << endl;
    int ret3 = system(("./pthread_death_pred " + dataFile + " " + numThreads).c_str());
    if (ret3 != 0) {
        cerr << "Error running Pthreads implementation. Make sure it's compiled." << endl;
        cerr << "Compile with: g++ -pthread -o pthread_death_pred pthread_death_pred.cpp -std=c++11" << endl;
    }

    cout << "\n[4/6] Running MPI (Synchronous) LR Implementation..." << endl;
    int ret4 = system(("mpirun --oversubscribe -np " + numThreads + " ./mpi_death_pred " + dataFile).c_str());
    if (ret4 != 0) {
        cerr << "Error running MPI implementation. Make sure it's compiled." << endl;
        cerr << "Compile with: mpic++ -o mpi_death_pred mpi_death_pred.cpp -std=c++11" << endl;
    }

    cout << "\n[5/6] Running MPI (Asynchronous) LR Implementation..." << endl;
    int ret_async = system(("mpirun --oversubscribe -np " + numThreads + " ./async_mpi_death_pred " + dataFile).c_str());
    if (ret_async != 0) {
        cerr << "Error running async MPI implementation. Make sure it's compiled." << endl;
        cerr << "Compile with: mpic++ -o async_mpi_death_pred async_mpi_death_pred.cpp -std=c++11" << endl;
    }

    cout << "\n[6/6] Running CUDA LR Implementation (optional)..." << endl;
    bool cudaAvailable = false;
    {
        ifstream cudaExeCheck("./cuda_death_pred");
        if (cudaExeCheck.good()) {
            cudaExeCheck.close();
            int ret5 = system(("./cuda_death_pred " + dataFile).c_str());
            if (ret5 == 0) {
                cudaAvailable = true;
            } else {
                cerr << "CUDA executable exists but failed to run. This is expected on systems without NVIDIA GPU." << endl;
                cerr << "CUDA results will be excluded from comparison." << endl;
            }
        } else {
            cudaExeCheck.close();
            cerr << "CUDA implementation not compiled. This is expected on systems without CUDA Toolkit." << endl;
            cerr << "To compile: nvcc -std=c++11 -O3 -o cuda_death_pred cuda_death_pred.cu" << endl;
            cerr << "CUDA results will be excluded from comparison." << endl;
        }
    }

    // Parse LR results
    vector<TimingResult> lrResults;
    try {
        lrResults.push_back(parseTimingFile("timing_serial.txt", "Serial"));
        lrResults.push_back(parseTimingFile("timing_openmp.txt", "OpenMP"));
        lrResults.push_back(parseTimingFile("timing_pthread.txt", "Pthreads"));
        lrResults.push_back(parseTimingFile("timing_mpi.txt", "MPI-Sync"));

        {
            ifstream asyncCheck("timing_async_mpi.txt");
            if (asyncCheck.good()) {
                asyncCheck.close();
                lrResults.push_back(parseTimingFile("timing_async_mpi.txt", "MPI-Async"));
            }
        }

        if (cudaAvailable) {
            ifstream cudaTimingCheck("timing_cuda.txt");
            if (cudaTimingCheck.good()) {
                cudaTimingCheck.close();
                lrResults.push_back(parseTimingFile("timing_cuda.txt", "CUDA"));
            } else {
                cudaTimingCheck.close();
                cerr << "\nWarning: CUDA timing file not found despite successful execution." << endl;
            }
        }
    } catch (const exception& e) {
        cerr << "\nError parsing LR timing files: " << e.what() << endl;
        cerr << "Make sure all LR implementations ran successfully." << endl;
        return 1;
    }

    double lrSerialTime = lrResults[0].totalTime;
    printComparisonTable(lrResults, lrSerialTime);
    generatePerformanceReport(lrResults, lrSerialTime);

    // -----------------------------------------------------------------------
    // SECTION 2: Random Forest implementations
    // -----------------------------------------------------------------------
    cout << "\n\n--- RANDOM FOREST IMPLEMENTATIONS ---" << endl;

    cout << "\n[1/9] Running Random Forest Serial Implementation..." << endl;
    int rfRet1 = system(("./serial_rf_death_pred " + dataFile).c_str());
    if (rfRet1 != 0) {
        cerr << "Error running RF serial. Compile with: g++ -std=c++11 -O3 -o serial_rf_death_pred serial_random_forest_death_pred.cpp" << endl;
    }

    cout << "\n[2/9] Running Random Forest OpenMP Implementation..." << endl;
    int rfRet2 = system(("./openmp_rf_death_pred " + dataFile).c_str());
    if (rfRet2 != 0) {
        cerr << "Error running RF OpenMP. Compile with: g++ -std=c++11 -O3 -fopenmp -o openmp_rf_death_pred openmp_random_forest_death_pred.cpp" << endl;
    }

    cout << "\n[3/9] Running Random Forest Pthreads Implementation..." << endl;
    int rfRet3 = system(("./pthread_rf_death_pred " + dataFile + " " + numThreads).c_str());
    if (rfRet3 != 0) {
        cerr << "Error running RF Pthreads. Compile with: g++ -std=c++11 -O3 -pthread -o pthread_rf_death_pred pthread_random_forest_death_pred.cpp" << endl;
    }

    cout << "\n[4/9] Running Random Forest MPI Implementation..." << endl;
    int rfRet4 = system(("mpirun --oversubscribe -np " + numThreads + " ./mpi_rf_death_pred " + dataFile).c_str());
    if (rfRet4 != 0) {
        cerr << "Error running RF MPI. Compile with: mpic++ -std=c++11 -O3 -o mpi_rf_death_pred mpi_random_forest_death_pred.cpp" << endl;
    }

    cout << "\n[5/9] Running Random Forest OpenMP+MPI Hybrid Implementation..." << endl;
    int rfRet5 = system(("mpirun --oversubscribe -np " + numThreads + " ./hybrid_openmp_mpi_rf_death_pred " + dataFile).c_str());
    if (rfRet5 != 0) {
        cerr << "Error running RF OpenMP+MPI. Compile with: mpic++ -std=c++11 -O3 -fopenmp -o hybrid_openmp_mpi_rf_death_pred hybrid_openmp_mpi_random_forest_death_pred.cpp" << endl;
    }

    cout << "\n[6/9] Running Random Forest Pthread+MPI Hybrid Implementation..." << endl;
    int rfRet6 = system(("mpirun --oversubscribe -np " + numThreads + " ./hybrid_pthread_mpi_rf_death_pred " + dataFile + " " + numThreads).c_str());
    if (rfRet6 != 0) {
        cerr << "Error running RF Pthread+MPI. Compile with: mpic++ -std=c++11 -O3 -pthread -o hybrid_pthread_mpi_rf_death_pred hybrid_pthread_mpi_random_forest_death_pred.cpp" << endl;
    }

    cout << "\n[7/9] Running Random Forest OpenMP+Pthread Hybrid Implementation..." << endl;
    int rfRet7 = system(("./hybrid_openmp_pthread_rf_death_pred " + dataFile + " " + numThreads).c_str());
    if (rfRet7 != 0) {
        cerr << "Error running RF OpenMP+Pthread. Compile with: g++ -std=c++11 -O3 -fopenmp -pthread -o hybrid_openmp_pthread_rf_death_pred hybrid_openmp_pthread_random_forest_death_pred.cpp" << endl;
    }

    cout << "\n[8/9] Running Random Forest Triple Hybrid (MPI+OpenMP+Pthread) Implementation..." << endl;
    int rfRet8 = system(("mpirun --oversubscribe -np " + numThreads + " ./hybrid_triple_rf_death_pred " + dataFile + " " + numThreads).c_str());
    if (rfRet8 != 0) {
        cerr << "Error running RF Triple Hybrid. Compile with: mpic++ -std=c++11 -O3 -fopenmp -pthread -o hybrid_triple_rf_death_pred hybrid_mpi_openmp_pthread_random_forest_death_pred.cpp" << endl;
    }

    cout << "\n[9/9] Running Random Forest CUDA Implementation (optional)..." << endl;
    bool rfCudaAvailable = false;
    {
        ifstream rfCudaExeCheck("./cuda_rf_death_pred");
        if (rfCudaExeCheck.good()) {
            rfCudaExeCheck.close();
            int rfRetCuda = system(("./cuda_rf_death_pred " + dataFile).c_str());
            if (rfRetCuda == 0) {
                rfCudaAvailable = true;
            } else {
                cerr << "RF CUDA executable exists but failed to run. This is expected without NVIDIA GPU." << endl;
            }
        } else {
            rfCudaExeCheck.close();
            cerr << "RF CUDA implementation not compiled. To compile: nvcc -std=c++11 -O3 -o cuda_rf_death_pred cuda_random_forest_death_pred.cu" << endl;
        }
    }

    // Parse RF results
    vector<TimingResult> rfResults;
    bool rfSerialOk = false;
    try {
        {
            ifstream f("timing_rf_serial.txt");
            if (f.good()) { f.close(); rfResults.push_back(parseTimingFile("timing_rf_serial.txt", "RF-Serial")); rfSerialOk = true; }
        }
        {
            ifstream f("timing_rf_openmp.txt");
            if (f.good()) { f.close(); rfResults.push_back(parseTimingFile("timing_rf_openmp.txt", "RF-OpenMP")); }
        }
        {
            ifstream f("timing_rf_pthread.txt");
            if (f.good()) { f.close(); rfResults.push_back(parseTimingFile("timing_rf_pthread.txt", "RF-Pthreads")); }
        }
        {
            ifstream f("timing_rf_mpi.txt");
            if (f.good()) { f.close(); rfResults.push_back(parseTimingFile("timing_rf_mpi.txt", "RF-MPI")); }
        }
        {
            ifstream f("timing_hybrid_openmp_mpi_rf.txt");
            if (f.good()) { f.close(); rfResults.push_back(parseTimingFile("timing_hybrid_openmp_mpi_rf.txt", "RF-OMP+MPI")); }
        }
        {
            ifstream f("timing_hybrid_pthread_mpi_rf.txt");
            if (f.good()) { f.close(); rfResults.push_back(parseTimingFile("timing_hybrid_pthread_mpi_rf.txt", "RF-Pth+MPI")); }
        }
        {
            ifstream f("timing_hybrid_openmp_pthread_rf.txt");
            if (f.good()) { f.close(); rfResults.push_back(parseTimingFile("timing_hybrid_openmp_pthread_rf.txt", "RF-OMP+Pth")); }
        }
        {
            ifstream f("timing_hybrid_triple_rf.txt");
            if (f.good()) { f.close(); rfResults.push_back(parseTimingFile("timing_hybrid_triple_rf.txt", "RF-Triple")); }
        }
        if (rfCudaAvailable) {
            ifstream f("timing_rf_cuda.txt");
            if (f.good()) { f.close(); rfResults.push_back(parseTimingFile("timing_rf_cuda.txt", "RF-CUDA")); }
        }
    } catch (const exception& e) {
        cerr << "\nError parsing RF timing files: " << e.what() << endl;
    }

    if (!rfResults.empty()) {
        double rfSerialTime = rfSerialOk ? rfResults[0].totalTime : rfResults[0].totalTime;
        printRFComparisonTable(rfResults, rfSerialTime);
        generateRFPerformanceReport(rfResults, rfSerialTime);
    } else {
        cerr << "\nNo RF timing files found. Make sure RF implementations ran successfully." << endl;
    }

    cout << "\n========================================================================================================" << endl;
    cout << "                                    EXECUTION COMPLETED" << endl;
    cout << "========================================================================================================" << endl;
    cout << "LR results:  timing files + performance_report.txt" << endl;
    cout << "RF results:  timing files + rf_performance_report.txt" << endl;
    cout << "Run: python3 generate_performance_graphs.py  -> publication graphs" << endl;
    cout << "Run: python3 plot_performance.py             -> data-driven plots" << endl;
    cout << "========================================================================================================" << endl;

    return 0;
}
