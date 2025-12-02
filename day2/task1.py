import os
import math

def main() -> None:
    total_task_one = 0
    total_task_two = 0

    # file input
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "input.txt")
    with open(file_path, encoding="utf-8") as f:
        file_content = f.read()
    ranges = file_content.strip().split(',')

    for range in ranges:
        start = range.split('-')[0]
        end = range.split('-')[1]
        total_task_one += solve_range_one(start, end)
        total_task_two += solve_range_two(start, end)
    print(total_task_one)
    print(total_task_two)

def solve_range_two(start: str, end: str) -> int:
    return 1


def solve_range_one(start: str, end: str) -> int:
    sum_val = 0
    # Calculate half-length (3 digits -> 2, 4 digits -> 2)
    seed_len = (len(start) + 1) // 2
    # Creates 11, 101, 1001...
    mult = (10 ** seed_len) + 1

    # Enforce minimum digits
    min_seed = 10 ** (seed_len - 1)

    # "Teleport" to the start of the range
    calc_seed = int(start) // mult
    seed = max(min_seed, calc_seed)

    # Correction for int division
    if seed * mult < int(start):
        seed += 1

    while True:
        # Detect if seed grew in digits
        seed_digits = int(math.log10(seed)) + 1
        if seed_digits > seed_len:
            seed_len += 1
            # Fast update: 11 -> 101
            mult = (mult - 1) * 10 + 1

        val = seed * mult

        if val > int(end):
            break

        sum_val += val
        seed += 1

    return sum_val

if __name__ == "__main__":
    main()
