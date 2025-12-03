// icd9_serial.cpp
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <chrono>

struct Stats {
    long long total_patients = 0;
    long long total_deaths = 0;
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

int main(int argc, char** argv) {
    std::string filename = "fake_test.csv";
    if (argc > 1) {
        filename = argv[1];
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

    std::unordered_map<std::string, Stats> stats;

    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        auto cols = split_csv_line(line);
        if ((int)cols.size() <= std::max(idx_icd, idx_flag)) continue;

        std::string code = cols[idx_icd];
        int flag = 0;
        try {
            flag = std::stoi(cols[idx_flag]);
        } catch (...) {
            flag = 0;
        }

        Stats &s = stats[code];
        s.total_patients++;
        if (flag == 1) {
            s.total_deaths++;
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Print results
    std::cout << "ICD9_CODE_1,total_patients,total_deaths,death_rate\n";
    for (const auto &kv : stats) {
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

    std::cerr << "[serial] elapsed_ms=" << elapsed_ms << "\n";
    return 0;
}
