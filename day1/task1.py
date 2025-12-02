import numpy as np

file_path = "./input.txt"
with open(file_path, "r", encoding="utf-8") as f:
    file_content = f.read()

real_input = file_content.strip()


def solve_safe_one(input_data):
    if not input_data.strip():
        return 0

    moves = [
        int(line[1:]) * (-1 if line[0] == "L" else 1)
        for line in input_data.strip().split("\n")
        if line
    ]

    arr_moves = np.array(moves, dtype=np.int32)
    cumulative_pos = np.cumsum(arr_moves)
    matches = np.count_nonzero((cumulative_pos + 50) % 100 == 0)

    return matches


def solve_safe_two(input_data):
    if not input_data.strip():
        return 0

    moves = [
        int(line[1:]) * (-1 if line[0] == "L" else 1)
        for line in input_data.strip().split("\n")
        if line
    ]

    arr_moves = np.array(moves, dtype=np.int32)
    # array with positions + 1 slot for starting position
    all_positions = np.zeros(
        len(arr_moves) + 1, dtype=np.int32
    )
    all_positions[0] = 50  # Starting position
    all_positions[1:] = (
        np.cumsum(arr_moves) + 50
    )
    # Cumulative positions adjusted by +50
    # example rotations: L30, R100, R130, L90
    # turns into cumulative positions = [50, 20, 120, 250, 160]
    starts = all_positions[:-1]  # 50, 20, 120, 250
    ends = all_positions[1:]  # 20, 120, 250, 160

    crossings = np.zeros(len(arr_moves), dtype=np.int32)
    right_mask = arr_moves >= 0
    if np.any(right_mask):
        crossings[right_mask] = (ends[right_mask] // 100) - (starts[right_mask] // 100)

    # -1 offset to correctly handle landing EXACTLY on 0
    left_mask = arr_moves < 0
    if np.any(left_mask):
        crossings[left_mask] = ((starts[left_mask] - 1) // 100) - (
            (ends[left_mask] - 1) // 100
        )

    return np.sum(crossings)  # 0 + 1 + 1 + 1


test_input = """
L68
L30
R48
L5
R60
L55
L1
L99
R14
L82
"""

print(f"Test Result Task 1 (Should be 3): {solve_safe_one(test_input)}")

print(f"Secret Password Task 1: {solve_safe_one(real_input)}")

print(f"Test Result Task 2 (Should be 6): {solve_safe_two(test_input)}")

print(f"Secret Password Task 2: {solve_safe_two(real_input)}")
