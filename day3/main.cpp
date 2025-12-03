#include <chrono>
#include <fstream>
#include <immintrin.h>
#include <iostream>
#include <string>
#include <vector>

using BatteryBank = std::vector<char>;

std::vector<BatteryBank> LoadInputs(const std::string &filename) {
    std::vector<BatteryBank> allBanks;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        exit(1);
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;

        BatteryBank bank(line.begin(), line.end());

        // add 32 bytes of zero padding, load safety
        size_t padding = 32;
        bank.resize(bank.size() + padding, 0);
        allBanks.push_back(std::move(bank));
    }
    return allBanks;
}

int findFirstIndex(const char *data, int size, char target) {

    while (target > '0') {
        __m256i v_target = _mm256_set1_epi8(target);
        for (int i = 0; i <= size; i += 32) {
            // load the next 32 bytes into the chunk
            __m256i chunk = _mm256_loadu_si256((const __m256i *)(data + i));
            // check if any of the values is my target
            __m256i comparison = _mm256_cmpeq_epi8(chunk, v_target);
            int mask = _mm256_movemask_epi8(comparison);
            if (mask != 0) {
                int index = i + __builtin_ctz(mask);
                // if the max is at last index, then we can't use it for first
                if (index == size - 1) {
                    continue;
                }
                // target found, get the index by counting the trailing zeros of the comparison mask
                return index;
            }
        }
        target--;
    }
    return -1;
}

int runningMax(char *start, int size) {
    if (size <= 0) {
        return 0;
    }
    // init all 32 to 0
    __m256i v_max = _mm256_setzero_si256();

    // get a stack of max values
    for (int i = 0; i < size; i += 32) {
        __m256i chunk = _mm256_loadu_si256((const __m256i *)(start + i));
        v_max = _mm256_max_epu8(v_max, chunk);
    }

    // Dump to array to find the single max byte
    alignas(32) char results[32];
    _mm256_store_si256((__m256i *)results, v_max);

    char max = 0;
    for (int k = 0; k < 32; ++k) {
        if (results[k] > max) max = results[k];
    }

    return max - '0';
}

int solveBank(BatteryBank bank) {

    // size without padding
    int realSize = bank.size() - 32;
    int firstIndex = findFirstIndex(bank.data(), realSize, '9');
    int maxTens = bank[firstIndex] - '0';
    int remainingLen = realSize - firstIndex - 1;
    // start runningMax at the number after the first
    int maxOnes = runningMax(&bank[firstIndex + 1], remainingLen);
    return maxTens * 10 + maxOnes;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cout << "usage: " << argv[0] << " [INPUT FILE PATH]" << std::endl;
        return 1;
    }
    std::string filename = argv[1];
    std::vector<BatteryBank> allBanks = LoadInputs(filename);
    std::cout << "Processing " << allBanks.size() << " banks..." << std::endl;
    long long sum = 0;

    // start timer
    auto start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < allBanks.size(); i++) {
        int res = solveBank(allBanks[i]);
        sum += res;
    }

    // stop Timer
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    std::cout << "Total Joltage: " << sum << std::endl;
    std::cout << "Calculation Time: " << duration.count() << " microseconds" << std::endl;
    std::cout << "Average per bank: " << (double)duration.count() / allBanks.size() << " us"
              << std::endl;
    return 0;
}
