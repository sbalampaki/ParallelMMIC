#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <limits>
#include <random>
#include <mpi.h>
#include <omp.h>
#include <pthread.h>

using namespace std;

struct Patient {
    int subjectId;
    string ethnicity;
    string gender;
    int icd9Code1;
    int expireFlag;
};

struct TreeNode {
    bool isLeaf;
    int prediction;

    string splitFeature;
    string splitValueStr;
    int splitValueInt;

    TreeNode* left;
    TreeNode* right;

    TreeNode() : isLeaf(false), prediction(0), splitValueInt(0), left(nullptr), right(nullptr) {}
    ~TreeNode() {
        delete left;
        delete right;
    }
};

class DecisionTree {
private:
    TreeNode* root;
    int maxDepth;
    int minSamplesSplit;
    mt19937* rng;

    double calculateEntropy(const vector<Patient>& data) {
        if (data.empty()) return 0.0;
        int positive = 0;
        for (const auto& p : data) {
            if (p.expireFlag == 1) positive++;
        }
        if (positive == 0 || positive == (int)data.size()) return 0.0;
        double p1 = (double)positive / data.size();
        double p0 = 1.0 - p1;
        return -p1 * log2(p1) - p0 * log2(p0);
    }

    double calculateInformationGain(const vector<Patient>& data,
                                     const vector<Patient>& left,
                                     const vector<Patient>& right) {
        double parentEntropy = calculateEntropy(data);
        double leftWeight  = (double)left.size()  / data.size();
        double rightWeight = (double)right.size() / data.size();
        return parentEntropy - (leftWeight  * calculateEntropy(left) +
                                rightWeight * calculateEntropy(right));
    }

    vector<string> sampleFeatures() {
        vector<string> all = {"ethnicity", "gender", "icd9"};
        if (rng == nullptr) return all;
        shuffle(all.begin(), all.end(), *rng);
        return vector<string>(all.begin(), all.begin() + 2);
    }

    void findBestSplit(const vector<Patient>& data,
                       string& bestFeature,
                       string& bestValueStr,
                       int& bestValueInt,
                       double& bestGain) {
        bestGain    = 0.0;
        bestFeature = "";

        vector<string> featuresToTry = sampleFeatures();

        for (const auto& feat : featuresToTry) {
            if (feat == "ethnicity") {
                map<string, int> groups;
                for (const auto& p : data) groups[p.ethnicity]++;
                for (const auto& kv : groups) {
                    vector<Patient> left, right;
                    for (const auto& p : data) {
                        if (p.ethnicity == kv.first) left.push_back(p);
                        else                          right.push_back(p);
                    }
                    if (!left.empty() && !right.empty()) {
                        double gain = calculateInformationGain(data, left, right);
                        if (gain > bestGain) {
                            bestGain     = gain;
                            bestFeature  = "ethnicity";
                            bestValueStr = kv.first;
                        }
                    }
                }
            } else if (feat == "gender") {
                map<string, int> groups;
                for (const auto& p : data) groups[p.gender]++;
                for (const auto& kv : groups) {
                    vector<Patient> left, right;
                    for (const auto& p : data) {
                        if (p.gender == kv.first) left.push_back(p);
                        else                       right.push_back(p);
                    }
                    if (!left.empty() && !right.empty()) {
                        double gain = calculateInformationGain(data, left, right);
                        if (gain > bestGain) {
                            bestGain     = gain;
                            bestFeature  = "gender";
                            bestValueStr = kv.first;
                        }
                    }
                }
            } else {
                map<int, int> groups;
                for (const auto& p : data) groups[p.icd9Code1]++;
                for (const auto& kv : groups) {
                    vector<Patient> left, right;
                    for (const auto& p : data) {
                        if (p.icd9Code1 == kv.first) left.push_back(p);
                        else                          right.push_back(p);
                    }
                    if (!left.empty() && !right.empty()) {
                        double gain = calculateInformationGain(data, left, right);
                        if (gain > bestGain) {
                            bestGain     = gain;
                            bestFeature  = "icd9";
                            bestValueInt = kv.first;
                        }
                    }
                }
            }
        }
    }

    TreeNode* buildTree(const vector<Patient>& data, int depth) {
        TreeNode* node = new TreeNode();

        if (data.empty()) {
            node->isLeaf     = true;
            node->prediction = 0;
            return node;
        }

        int positive = 0;
        for (const auto& p : data) {
            if (p.expireFlag == 1) positive++;
        }

        if (positive == 0 || positive == (int)data.size() ||
            depth >= maxDepth || (int)data.size() < minSamplesSplit) {
            node->isLeaf     = true;
            node->prediction = (positive > (int)data.size() / 2) ? 1 : 0;
            return node;
        }

        string bestFeature, bestValueStr;
        int    bestValueInt = 0;
        double bestGain     = 0.0;
        findBestSplit(data, bestFeature, bestValueStr, bestValueInt, bestGain);

        if (bestGain <= 0.0 || bestFeature.empty()) {
            node->isLeaf     = true;
            node->prediction = (positive > (int)data.size() / 2) ? 1 : 0;
            return node;
        }

        vector<Patient> left, right;
        if (bestFeature == "ethnicity") {
            for (const auto& p : data) {
                if (p.ethnicity == bestValueStr) left.push_back(p);
                else                              right.push_back(p);
            }
            node->splitFeature  = "ethnicity";
            node->splitValueStr = bestValueStr;
        } else if (bestFeature == "gender") {
            for (const auto& p : data) {
                if (p.gender == bestValueStr) left.push_back(p);
                else                           right.push_back(p);
            }
            node->splitFeature  = "gender";
            node->splitValueStr = bestValueStr;
        } else {
            for (const auto& p : data) {
                if (p.icd9Code1 == bestValueInt) left.push_back(p);
                else                              right.push_back(p);
            }
            node->splitFeature  = "icd9";
            node->splitValueInt = bestValueInt;
        }

        node->left  = buildTree(left,  depth + 1);
        node->right = buildTree(right, depth + 1);
        return node;
    }

    int predictOne(TreeNode* node, const Patient& p) const {
        if (node->isLeaf) return node->prediction;
        bool goLeft = false;
        if (node->splitFeature == "ethnicity") goLeft = (p.ethnicity == node->splitValueStr);
        else if (node->splitFeature == "gender") goLeft = (p.gender == node->splitValueStr);
        else                                     goLeft = (p.icd9Code1 == node->splitValueInt);
        return goLeft ? predictOne(node->left, p) : predictOne(node->right, p);
    }

public:
    DecisionTree(int maxD = 10, int minSamples = 5)
        : root(nullptr), maxDepth(maxD), minSamplesSplit(minSamples), rng(nullptr) {}

    ~DecisionTree() { delete root; }

    void train(const vector<Patient>& data, mt19937* featureRng = nullptr) {
        rng = featureRng;
        delete root;
        root = buildTree(data, 0);
        rng = nullptr;
    }

    int predict(const Patient& p) const {
        if (!root) return 0;
        return predictOne(root, p);
    }
};

// ─── Pthread thread data ───────────────────────────────────────────────────────

struct TrainThreadData {
    const vector<Patient>* trainData;
    vector<DecisionTree*>* localTrees;
    int treeStart;   // global tree index (used as seed key)
    int treeEnd;
    int localOffset; // offset into localTrees
    int maxDepth;
    int minSamplesSplit;
    unsigned int seed;
};

void* trainTreeRangePthread(void* arg) {
    TrainThreadData* td = (TrainThreadData*)arg;
    int n = (int)td->trainData->size();

    for (int t = td->treeStart; t < td->treeEnd; t++) {
        mt19937 treeRng(td->seed + (unsigned int)t * 1000003u);
        uniform_int_distribution<int> dist(0, n - 1);

        vector<Patient> bootstrap;
        bootstrap.reserve(n);
        for (int i = 0; i < n; i++) {
            bootstrap.push_back((*td->trainData)[dist(treeRng)]);
        }

        int li = td->localOffset + (t - td->treeStart);
        DecisionTree* tree = new DecisionTree(td->maxDepth, td->minSamplesSplit);
        tree->train(bootstrap, &treeRng);
        (*td->localTrees)[li] = tree;
    }
    return NULL;
}

// ─── Triple Hybrid (MPI + OpenMP + Pthread) Random Forest ────────────────────
//
// Parallelism strategy:
//   Level 1 – MPI:     Each MPI rank is assigned a disjoint slice of trees.
//   Level 2 – OpenMP:  Within each rank, OpenMP divides the rank's slice into
//                      per-thread coarse blocks.
//   Level 3 – Pthreads: Each OpenMP thread launches numPThreads Pthreads that
//                       further sub-divide its block.
//   Evaluation: local votes reduced with MPI_Allreduce; test-set iteration is
//               parallelised with OpenMP reductions inside each rank.
// ─────────────────────────────────────────────────────────────────────────────
class RandomForestTripleHybrid {
private:
    vector<DecisionTree*> localTrees;
    int numTrees;
    int maxDepth;
    int minSamplesSplit;
    int rank;
    int mpiSize;
    int numPThreads;

    int mpiTreeStart;
    int mpiTreeEnd;

public:
    RandomForestTripleHybrid(int nTrees, int maxD, int minSamples, int r, int s, int nPthreads = 2)
        : numTrees(nTrees), maxDepth(maxD), minSamplesSplit(minSamples),
          rank(r), mpiSize(s), numPThreads(nPthreads > 0 ? nPthreads : 1) {
        int treesPerProc = numTrees / mpiSize;
        mpiTreeStart     = rank * treesPerProc;
        mpiTreeEnd       = (rank == mpiSize - 1) ? numTrees : (rank + 1) * treesPerProc;
        localTrees.resize(mpiTreeEnd - mpiTreeStart, nullptr);
    }

    ~RandomForestTripleHybrid() {
        for (auto* t : localTrees) delete t;
    }

    void train(const vector<Patient>& data, unsigned int seed = 42) {
        int localCount = mpiTreeEnd - mpiTreeStart;
        int ompThreads = omp_get_max_threads();
        int ompChunk   = (localCount + ompThreads - 1) / ompThreads;

        // ── OpenMP: divide local trees into coarse blocks ──────────────────
        #pragma omp parallel
        {
            int tid       = omp_get_thread_num();
            int ompStart  = tid * ompChunk;
            int ompEnd    = min(ompStart + ompChunk, localCount);
            if (ompStart >= localCount) { /* idle thread */ }
            else {
                int ompCount  = ompEnd - ompStart;
                int ptChunk   = (ompCount + numPThreads - 1) / numPThreads;

                vector<pthread_t> pthreads(numPThreads);
                vector<TrainThreadData> tdArray(numPThreads);

                for (int pt = 0; pt < numPThreads; pt++) {
                    int ptLocalStart = pt * ptChunk;
                    int ptLocalEnd   = min(ptLocalStart + ptChunk, ompCount);

                    tdArray[pt].trainData       = &data;
                    tdArray[pt].localTrees      = &localTrees;
                    // Global tree index (for consistent seeding across all ranks/threads)
                    tdArray[pt].treeStart       = mpiTreeStart + ompStart + ptLocalStart;
                    tdArray[pt].treeEnd         = mpiTreeStart + ompStart + ptLocalEnd;
                    tdArray[pt].localOffset     = ompStart + ptLocalStart;
                    tdArray[pt].maxDepth        = maxDepth;
                    tdArray[pt].minSamplesSplit = minSamplesSplit;
                    tdArray[pt].seed            = seed;

                    pthread_create(&pthreads[pt], NULL, trainTreeRangePthread, &tdArray[pt]);
                }

                for (int pt = 0; pt < numPThreads; pt++) {
                    pthread_join(pthreads[pt], NULL);
                }
            }
        }
    }

    double calculateAccuracy(const vector<Patient>& testData) {
        int n = (int)testData.size();

        // OpenMP: accumulate local votes in parallel within this rank
        vector<int> localVotes(n, 0);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) {
            int v = 0;
            for (const auto* t : localTrees) if (t) v += t->predict(testData[i]);
            localVotes[i] = v;
        }

        // MPI: sum votes across all ranks
        vector<int> globalVotes(n, 0);
        MPI_Allreduce(localVotes.data(), globalVotes.data(), n,
                      MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        int correct = 0;
        for (int i = 0; i < n; i++) {
            int pred = (globalVotes[i] * 2 >= numTrees) ? 1 : 0;
            if (pred == testData[i].expireFlag) correct++;
        }
        return (double)correct / n;
    }

    double calculateDeathRate(const vector<Patient>& testData) {
        int n = (int)testData.size();

        vector<int> localVotes(n, 0);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) {
            int v = 0;
            for (const auto* t : localTrees) if (t) v += t->predict(testData[i]);
            localVotes[i] = v;
        }

        vector<int> globalVotes(n, 0);
        MPI_Allreduce(localVotes.data(), globalVotes.data(), n,
                      MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        double total = 0.0;
        for (int i = 0; i < n; i++) total += (double)globalVotes[i] / numTrees;
        return total / n;
    }
};

// ─── Data loading ─────────────────────────────────────────────────────────────

int safeStoi(const string& str) {
    if (str.empty()) return 0;
    try {
        string cleaned = str;
        cleaned.erase(remove(cleaned.begin(), cleaned.end(), ' '), cleaned.end());
        if (cleaned.empty()) return 0;
        return stoi(cleaned);
    } catch (...) {
        return 0;
    }
}

vector<Patient> loadData(const string& filename) {
    vector<Patient> patients;
    ifstream file(filename);

    if (!file.is_open()) {
        cerr << "Error: Could not open file " << filename << endl;
        return patients;
    }

    string line;
    getline(file, line);

    int lineNum = 1;
    while (getline(file, line)) {
        lineNum++;
        if (line.empty()) continue;

        try {
            stringstream ss(line);
            Patient p;
            string temp;

            auto removeQuotes = [](string& s) {
                s.erase(remove(s.begin(), s.end(), '\"'), s.end());
            };

            if (!getline(ss, temp, ',')) continue;
            removeQuotes(temp);
            p.subjectId = safeStoi(temp);

            if (!getline(ss, temp, ',')) continue;

            if (!getline(ss, temp, ',')) continue;
            removeQuotes(temp);
            p.expireFlag = safeStoi(temp);

            if (!getline(ss, temp, ',')) continue;

            if (!getline(ss, p.ethnicity, ',')) continue;
            removeQuotes(p.ethnicity);
            p.ethnicity.erase(0, p.ethnicity.find_first_not_of(" \t\r\n"));
            p.ethnicity.erase(p.ethnicity.find_last_not_of(" \t\r\n") + 1);

            if (!getline(ss, p.gender, ',')) continue;
            removeQuotes(p.gender);
            p.gender.erase(0, p.gender.find_first_not_of(" \t\r\n"));
            p.gender.erase(p.gender.find_last_not_of(" \t\r\n") + 1);

            if (!getline(ss, temp, ',')) continue;
            if (!getline(ss, temp, ',')) continue;

            if (!getline(ss, temp, ',')) {
                p.icd9Code1 = 0;
            } else {
                removeQuotes(temp);
                p.icd9Code1 = safeStoi(temp);
            }

            patients.push_back(p);
        } catch (const exception& e) {
            cerr << "Warning: Error parsing line " << lineNum << ": " << e.what() << endl;
            continue;
        }
    }

    if (patients.empty()) {
        cerr << "Error: No valid data loaded from file!" << endl;
    }

    return patients;
}

// ─── Main ─────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, mpiSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

    if (argc < 2) {
        if (rank == 0) {
            cerr << "Usage: mpirun -np <processes> " << argv[0] << " <data_file> [num_pthreads]" << endl;
        }
        MPI_Finalize();
        return 1;
    }

    int numPThreads = (argc > 2) ? atoi(argv[2]) : 2;

    auto startTotal = chrono::high_resolution_clock::now();

    auto startLoad = chrono::high_resolution_clock::now();
    vector<Patient> data = loadData(argv[1]);
    auto endLoad = chrono::high_resolution_clock::now();
    double loadTime = chrono::duration<double>(endLoad - startLoad).count();

    if (rank == 0) {
        cout << "Triple Hybrid (MPI+OpenMP+Pthread) Implementation - Random Forest" << endl;
        cout << "===================================================================" << endl;
        cout << "MPI Processes: " << mpiSize << endl;
        cout << "OpenMP Threads per process: " << omp_get_max_threads() << endl;
        cout << "Pthreads per OpenMP thread: " << numPThreads << endl;
        cout << "Total parallelism: " << (mpiSize * omp_get_max_threads() * numPThreads) << " units" << endl;
        cout << "Data loaded: " << data.size() << " patients" << endl;
        cout << "Load time: " << loadTime << " seconds" << endl;
    }

    int trainSize = (int)(data.size() * 0.8);
    vector<Patient> trainData(data.begin(), data.begin() + trainSize);
    vector<Patient> testData(data.begin() + trainSize, data.end());

    if (rank == 0) {
        cout << "Training samples: " << trainData.size()
             << "  Test samples: " << testData.size() << endl;
        cout << "Model: Random Forest (50 trees, MPI + OpenMP + Pthread triple-hybrid)" << endl;
    }

    auto startTrain = chrono::high_resolution_clock::now();
    RandomForestTripleHybrid model(50, 10, 5, rank, mpiSize, numPThreads);
    model.train(trainData, 42);
    auto endTrain = chrono::high_resolution_clock::now();
    double trainTime = chrono::duration<double>(endTrain - startTrain).count();

    if (rank == 0) {
        cout << "Training time: " << trainTime << " seconds" << endl;
    }

    auto startEval = chrono::high_resolution_clock::now();
    double accuracy  = model.calculateAccuracy(testData);
    double deathRate = model.calculateDeathRate(testData);
    auto endEval = chrono::high_resolution_clock::now();
    double evalTime = chrono::duration<double>(endEval - startEval).count();

    auto endTotal = chrono::high_resolution_clock::now();
    double totalTime = chrono::duration<double>(endTotal - startTotal).count();

    if (rank == 0) {
        cout << "Evaluation time: " << evalTime << " seconds" << endl;
        cout << "Total execution time: " << totalTime << " seconds" << endl;
        cout << "\nResults:" << endl;
        cout << "Accuracy: " << (accuracy * 100) << "%" << endl;
        cout << "Predicted Death Rate: " << (deathRate * 100) << "%" << endl;

        ofstream timingFile("timing_hybrid_triple_rf.txt");
        timingFile << "load,"      << loadTime  << endl;
        timingFile << "train,"     << trainTime << endl;
        timingFile << "eval,"      << evalTime  << endl;
        timingFile << "total,"     << totalTime << endl;
        timingFile << "accuracy,"  << accuracy  << endl;
        timingFile << "deathrate," << deathRate << endl;
        timingFile << "processes," << mpiSize   << endl;
        timingFile << "threads,"   << (mpiSize * omp_get_max_threads() * numPThreads) << endl;
        timingFile.close();
    }

    MPI_Finalize();
    return 0;
}
