// icd9_mpi.cpp
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <chrono>

#include <mpi.h>

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
    MPI_Init(&argc, &argv);

    int world_size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::string filename = "fake_test.csv";
    if (argc > 1) {
        filename = argv[1];
    }

    auto t0 = std::chrono::steady_clock::now();

    std::ifstream fin(filename);
    if (!fin) {
        if (rank == 0) {
            std::cerr << "Error: could not open " << filename << "\n";
        }
        MPI_Finalize();
        return 1;
    }

    std::string header;
    if (!std::getline(fin, header)) {
        if (rank == 0) {
            std::cerr << "Error: empty CSV\n";
        }
        MPI_Finalize();
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
        if (rank == 0) {
            std::cerr << "Error: required columns not found\n";
        }
        MPI_Finalize();
        return 1;
    }

    std::unordered_map<std::string, Stats> local_stats;

    std::string line;
    long long line_idx = 0;
    while (std::getline(fin, line)) {
        if (line.empty()) {
            line_idx++;
            continue;
        }
        if (line_idx % world_size != rank) {
            line_idx++;
            continue;
        }

        auto cols = split_csv_line(line);
        if ((int)cols.size() <= std::max(idx_icd, idx_flag)) {
            line_idx++;
            continue;
        }

        std::string code = cols[idx_icd];
        int flag = 0;
        try {
            flag = std::stoi(cols[idx_flag]);
        } catch (...) {
            flag = 0;
        }

        Stats &s = local_stats[code];
        s.total_patients++;
        if (flag == 1) {
            s.total_deaths++;
        }

        line_idx++;
    }

    // Serialize local map to send to rank 0
    std::vector<char> send_buf;
    {
        std::stringstream ss;
        for (const auto &kv : local_stats) {
            const std::string &code = kv.first;
            const Stats &s = kv.second;
            ss << code << "," << s.total_patients << "," << s.total_deaths << "\n";
        }
        std::string tmp = ss.str();
        send_buf.assign(tmp.begin(), tmp.end());
    }

    int send_size = (int)send_buf.size();
    std::vector<int> recv_sizes;
    if (rank == 0) {
        recv_sizes.resize(world_size);
    }

    MPI_Gather(&send_size, 1, MPI_INT,
               recv_sizes.data(), 1, MPI_INT,
               0, MPI_COMM_WORLD);

    std::vector<int> displs;
    std::vector<char> recv_buf;
    if (rank == 0) {
        displs.resize(world_size);
        int total_size = 0;
        for (int i = 0; i < world_size; ++i) {
            displs[i] = total_size;
            total_size += recv_sizes[i];
        }
        recv_buf.resize(total_size);
        MPI_Gatherv(send_buf.data(), send_size, MPI_CHAR,
                    recv_buf.data(), recv_sizes.data(), displs.data(), MPI_CHAR,
                    0, MPI_COMM_WORLD);
    } else {
        MPI_Gatherv(send_buf.data(), send_size, MPI_CHAR,
                    nullptr, nullptr, nullptr, MPI_CHAR,
                    0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        std::unordered_map<std::string, Stats> global_stats;

        // Add rank 0's own local map first
        for (const auto &kv : local_stats) {
            const std::string &code = kv.first;
            const Stats &s = kv.second;
            Stats &g = global_stats[code];
            g.total_patients += s.total_patients;
            g.total_deaths += s.total_deaths;
        }

        // Parse data from other ranks
        int offset = recv_sizes[0]; // rank0's part is already in local_stats
        for (int r = 1; r < world_size; ++r) {
            int size = recv_sizes[r];
            std::string chunk(recv_buf.begin() + displs[r],
                              recv_buf.begin() + displs[r] + size);
            std::stringstream ss(chunk);
            std::string line2;
            while (std::getline(ss, line2)) {
                if (line2.empty()) continue;
                auto cols = split_csv_line(line2);
                if (cols.size() < 3) continue;
                std::string code = cols[0];
                long long tp = std::stoll(cols[1]);
                long long td = std::stoll(cols[2]);
                Stats &g = global_stats[code];
                g.total_patients += tp;
                g.total_deaths += td;
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

        std::cerr << "[mpi] ranks=" << world_size
                  << " elapsed_ms=" << elapsed_ms << "\n";
    }

    MPI_Finalize();
    return 0;
}
