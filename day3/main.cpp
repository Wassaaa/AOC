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

        // add 32 bytes of zero padding, load safety for SIMD
        size_t padding = 32;
        bank.resize(bank.size() + padding, 0);
        allBanks.push_back(std::move(bank));
    }
    return allBanks;
}

int findBestDigitInWindow(const char *data, int startIdx, int endIdx) {
    char target = '9';

    // Try to find 9, then 8, then 7... within the window
    while (target >= '0') {
        __m256i v_target = _mm256_set1_epi8(target);

        // Loop through the data in 32-byte chunks
        for (int i = startIdx; i <= endIdx; i += 32) {
            __m256i chunk = _mm256_loadu_si256((const __m256i *)(data + i));
            __m256i comparison = _mm256_cmpeq_epi8(chunk, v_target);
            int mask = _mm256_movemask_epi8(comparison);

            if (mask != 0) {
                // Determine the exact index of the match
                int localIndex = __builtin_ctz(mask);
                int globalIndex = i + localIndex;

                // ensure the found index is actually within our window.
                if (globalIndex <= endIdx) {
                    return globalIndex;
                }
                // If globalIndex > endIdx, we ignore it.
            }
        }
        target--;
    }
    return -1;
}

long long solveBank(const BatteryBank &bank, int targetDigits) {
    // Size without padding
    int realSize = bank.size() - 32;

    int currentSearchStart = 0;
    long long currentJoltage = 0;

    for (int i = 0; i < targetDigits; ++i) {
        int remainingNeeded = targetDigits - 1 - i;

        // cannot search past this index
        int searchEndLimit = realSize - 1 - remainingNeeded;

        int foundIndex = findBestDigitInWindow(bank.data(), currentSearchStart, searchEndLimit);

        int digit = bank[foundIndex] - '0';
        currentJoltage = currentJoltage * 10 + digit;

        // advance start for the next digit
        currentSearchStart = foundIndex + 1;
    }

    return currentJoltage;
}

int main(int argc, char **argv) {
    // Default values
    int digitsToFind = 12;
    std::string filename = "../day3/input";

    // Override if arguments are provided
    if (argc == 3) {
        digitsToFind = std::stoi(argv[1]);
        filename = argv[2];
    } else if (argc != 1) {
        std::cout << "usage: " << argv[0] << " [JOLTAGE SIZE] [INPUT FILE PATH]" << std::endl;
        std::cout << "Defaults: 12 digits, input.txt" << std::endl;
        return 1;
    }

    std::vector<BatteryBank> allBanks = LoadInputs(filename);
    std::cout << "Processing " << allBanks.size() << " banks..." << std::endl;

    long long sum = 0;

    // Start timer
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < allBanks.size(); i++) {
        long long res = solveBank(allBanks[i], digitsToFind);
        sum += res;
    }

    // Stop Timer
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    std::cout << "Total Joltage: " << sum << std::endl;
    std::cout << "Calculation Time: " << duration.count() << " microseconds" << std::endl;
    std::cout << "Average per bank: " << (double)duration.count() / allBanks.size() << " us"
              << std::endl;
    return 0;
}
