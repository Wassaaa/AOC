import os

def main() -> None:
    ids_task_one = set()
    ids_task_two = set()

    # file input
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "input.txt")
    with open(file_path, encoding="utf-8") as f:
        file_content = f.read()
    ranges = file_content.strip().split(',')

    for range in ranges:
        start = range.split('-')[0]
        end = range.split('-')[1]
        ids_task_one.update(solve_task_1(start, end))
        ids_task_two.update(solve_task_2(start, end))
    print(sum(ids_task_one))
    print(sum(ids_task_two))


def find_pattern_ids(start_int: int, end_int: int, seed_len: int, repeats: int) -> set:
    found = set()

    # Calculate the Generic Multiplier
    mult = 0
    for i in range(repeats):
        mult += 10**(i * seed_len)

    # Determine Seed Boundaries
    min_seed = 10**(seed_len - 1)
    max_seed = 10**seed_len - 1

    # Early exits
    min_id = min_seed * mult
    if min_id > end_int:
        return found
    max_id = max_seed * mult
    if max_id < start_int:
        return found

    # Teleport to Start
    calc_seed = (start_int + mult - 1) // mult
    current_seed = max(min_seed, calc_seed)

    # The Generation Loop
    while current_seed <= max_seed:
        val = current_seed * mult

        if val > end_int:
            break

        if val >= start_int:
            found.add(val)

        current_seed += 1

    return found

def solve_task_2(start: str, end: str) -> set:
    start_int = int(start)
    end_int = int(end)
    found_ids = set()

    for total_len in range(len(start), len(end) + 1):
        for seed_len in range(1, (total_len // 2) + 1):
            if total_len % seed_len == 0:
                repeats = total_len // seed_len

                # Call the engine
                new_ids = find_pattern_ids(start_int, end_int, seed_len, repeats)
                found_ids.update(new_ids)

    return found_ids

def solve_task_1(start: str, end: str) -> set:
    start_int = int(start)
    end_int = int(end)
    found_ids = set()

    seed_len = 1
    while True:
        # 2 repeats
        new_ids = find_pattern_ids(start_int, end_int, seed_len, repeats=2)

        min_possible = (10**(seed_len-1)) * (10**seed_len + 1)
        if not new_ids and min_possible > end_int:
            break

        found_ids.update(new_ids)
        seed_len += 1

    return found_ids

if __name__ == "__main__":
    main()
