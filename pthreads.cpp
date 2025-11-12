// icd9_pthread.cpp
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <chrono>
#include <pthread.h>

struct Stats {
    long long total_patients = 0;
    long long total_deaths = 0;
};

struct Row {
    std::string icd;
    int flag;
};

struct ThreadArg {
    const std::vector<Row>* rows;
    int start;
    int end;
    std::unordered_map<std::string, Stats>* local_stats;
};

std::vector<std::string> split_csv_line(const std::string &line) {
    std::vector<std::string> cols;
    std::stringstream ss(line);
    std::string item;
    while (std::getline(ss, item, ',')) {
        cols.push_back(item);
    }
    return cols;
}

void* thread_func(void* arg) {
    ThreadArg* ta = static_cast<ThreadArg*>(arg);
    const std::vector<Row>& rows = *(ta->rows);
    auto* local = ta->local_stats;

    for (int i = ta->start; i < ta->end; ++i) {
        const Row& r = rows[i];
        Stats &s = (*local)[r.icd];
        s.total_patients++;
        if (r.flag == 1) {
            s.total_deaths++;
        }
    }
    return nullptr;
}

int main(int argc, char** argv) {
    std::string filename = "fake_test.csv";
    if (argc > 1) {
        filename = argv[1];
    }

    int num_threads = 4;
    if (argc > 2) {
        num_threads = std::stoi(argv[2]);
    }

    auto t0 = std::chrono::steady_clock::now();

    std::ifstream fin(filename);
    if (!fin) {
        std::cerr << "Error: could not open " << filename << "\n";
        return 1;
    }

    std::string header;
    if (!std::getline(fin, header)) {
        std::cerr << "Error: empty CSV\n";
        return 1;
    }

    std::vector<std::string> header_cols = split_csv_line(header);
    int idx_icd = -1;
    int idx_flag = -1;
    for (size_t i = 0; i < header_cols.size(); ++i) {
        if (header_cols[i] == "ICD9_CODE_1") idx_icd = (int)i;
        if (header_cols[i] == "HOSPITAL_EXPIRE_FLAG") idx_flag = (int)i;
    }
    if (idx_icd == -1 || idx_flag == -1) {
        std::cerr << "Error: required columns not found\n";
        return 1;
    }

    std::vector<Row> rows;
    rows.reserve(100000);

    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        auto cols = split_csv_line(line);
        if ((int)cols.size() <= std::max(idx_icd, idx_flag)) continue;

        Row r;
        r.icd = cols[idx_icd];
        try {
            r.flag = std::stoi(cols[idx_flag]);
        } catch (...) {
            r.flag = 0;
        }
        rows.push_back(std::move(r));
    }

    if (num_threads > (int)rows.size()) {
        num_threads = rows.size();
    }
    if (num_threads <= 0) num_threads = 1;

    std::vector<pthread_t> threads(num_threads);
    std::vector<ThreadArg> targs(num_threads);
    std::vector<std::unordered_map<std::string, Stats>> locals(num_threads);

    int chunk = rows.size() / num_threads;
    int remainder = rows.size() % num_threads;
    int start = 0;

    for (int i = 0; i < num_threads; ++i) {
        int len = chunk + (i < remainder ? 1 : 0);
        targs[i].rows = &rows;
        targs[i].start = start;
        targs[i].end = start + len;
        targs[i].local_stats = &locals[i];
        start += len;

        pthread_create(&threads[i], nullptr, thread_func, &targs[i]);
    }

    for (int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], nullptr);
    }

    std::unordered_map<std::string, Stats> global_stats;

    for (int i = 0; i < num_threads; ++i) {
        for (const auto &kv : locals[i]) {
            const std::string &code = kv.first;
            const Stats &ls = kv.second;
            Stats &gs = global_stats[code];
            gs.total_patients += ls.total_patients;
            gs.total_deaths += ls.total_deaths;
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "ICD9_CODE_1,total_patients,total_deaths,death_rate\n";
    for (const auto &kv : global_stats) {
        const std::string &code = kv.first;
        const Stats &s = kv.second;
        double rate = (s.total_patients > 0)
                          ? (double)s.total_deaths / (double)s.total_patients
                          : 0.0;
        std::cout << code << ","
                  << s.total_patients << ","
                  << s.total_deaths << ","
                  << rate << "\n";
    }

    std::cerr << "[pthread] threads=" << num_threads
              << " elapsed_ms=" << elapsed_ms << "\n";
    return 0;
}
