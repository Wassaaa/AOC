#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <execution>
#include <fstream>
#include <iostream>
#include <numeric>
#include <ranges>
#include <vector>

struct Event {
    int64_t coords;
    int8_t type;
};

void LoadInputs(const std::string &filename, std::vector<std::pair<int64_t, int64_t>> &ranges,
                std::vector<int64_t> &queries) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cout << "couldn't open input file." << std::endl;
        exit(1);
    }

    std::string line;
    bool parsingRanges = true;
    while (std::getline(file, line)) {
        if (line.empty()) {
            parsingRanges = false;
            continue;
        }
        if (parsingRanges) {
            size_t dash = line.find('-');
            if (dash == std::string::npos) {
                std::cout << "no dash, weird a f" << std::endl;
            }
            int64_t start = 0, end = 0;
            std::from_chars(line.data(), line.data() + dash, start);
            std::from_chars(line.data() + dash + 1, line.data() + line.size(), end);
            ranges.emplace_back(start, end);
            continue;
        }
        int64_t query = 0;
        std::from_chars(line.data(), line.data() + line.size(), query);
        queries.emplace_back(query);
    }
}

bool compareEvents(const Event &a, const Event &b) {
    if (a.coords == b.coords) {
        return a.type > b.type;
    }
    return a.coords < b.coords;
}

int main() {
    std::vector<int64_t> queries;
    std::vector<Event> events;
    {
        std::vector<std::pair<int64_t, int64_t>> ranges;
        LoadInputs("/home/wsl/AOC/day5/input", ranges, queries);
        events.reserve(ranges.size() * 2);
        // parse int events, [coords, +1 or -1]
        for (auto range : ranges) {
            events.emplace_back(range.first, 1);
            // + 1 the end of the range to make it exclusive instead of inclusinve value
            events.emplace_back(range.second + 1, -1);
        }
    }
    auto core_start_time = std::chrono::high_resolution_clock::now();
    // sort the events
    std::sort(std::execution::unseq, events.begin(), events.end(), compareEvents);

    std::vector<int8_t> values(events.size());
    std::vector<int64_t> coords(events.size());
    std::vector<int32_t> depths(events.size());

    auto event_indices = std::views::iota((size_t)0, events.size());
    std::for_each(std::execution::unseq, event_indices.begin(), event_indices.end(), [&](size_t i) {
        // scatter the Event struct
        values[i] = events[i].type;
        coords[i] = events[i].coords;
    });
    // cumulative sum = inclusive_scan [1, -1, -1, 1, 1] -> [1, 0, -1, 0, 1]
    std::inclusive_scan(std::execution::unseq, values.begin(), values.end(), depths.begin());
    auto core_end_time = std::chrono::high_resolution_clock::now();
    auto core_dur = core_end_time - core_start_time;
    // binary search and count fresh

    auto task1_start_time = std::chrono::high_resolution_clock::now();
    int fresh_count = 0;
    for (int64_t query : queries) {
        auto it = std::upper_bound(coords.begin(), coords.end(), query);
        int index = std::distance(coords.begin(), it) - 1;
        // depth bigger than 0 means its in one of the ranges.
        if (index >= 0 && depths[index] > 0) {
            fresh_count++;
        }
    }
    auto task1_end_time = std::chrono::high_resolution_clock::now();
    auto task1_dur = std::chrono::duration_cast<std::chrono::microseconds>(
        core_dur + (task1_end_time - task1_start_time));

    std::cout << "Fresh Ingredients: " << fresh_count << "\n";
    std::cout << "Task 1 Time: " << task1_dur.count() << " us\n";

    auto task2_start_time = std::chrono::high_resolution_clock::now();
    int64_t total_fresh_length = 0;

    if (!coords.empty()) {
        auto indices = std::views::iota((size_t)0, coords.size() - 1);

        total_fresh_length =
            std::transform_reduce(std::execution::unseq, indices.begin(), indices.end(), (int64_t)0,
                                  std::plus<>(), [&](size_t i) {
                                      // Branchless Math
                                      // (depths[i] > 0) becomes integer 1 or 0.
                                      return (coords[i + 1] - coords[i]) * (depths[i] > 0);
                                  });
    }
    auto task2_end_time = std::chrono::high_resolution_clock::now();
    auto task2_dur = std::chrono::duration_cast<std::chrono::microseconds>(
        core_dur + (task2_end_time - task2_start_time));
    std::cout << "Task 2 Time: " << task2_dur.count() << " us\n";

    std::cout << "Ingredients considered fresh: " << total_fresh_length << "\n";
}
