import numpy as np
import io
import os
import time


def do_op(a: int, b: int, op: str) -> int:
    if op == "+":
        return a + b
    return a * b


def task_two_pythonic(input_data: str) -> int:
    lines = input_data.splitlines()

    digit_rows = lines[:-1]
    ops = lines[-1].split()
    cols = zip(*digit_rows)

    grand_total = 0
    block_total = 0
    current_op = None

    for col in cols:
        s = "".join(col).strip()
        if not s:
            grand_total += block_total
            block_total = 0
            ops.pop(0)
            current_op = None
            continue

        num = int(s)
        if not current_op:
            current_op = ops[0]
            block_total = num
            continue
        if ops[0] == "+":
            block_total += num
        if ops[0] == "*":
            block_total *= num

    return grand_total + block_total


def solve_task_two(input_data: str) -> int:
    lines = input_data.split("\n")
    width = max(len(line) for line in lines)

    grid = np.genfromtxt(
        io.StringIO(input_data),
        delimiter=[1] * width,
        dtype="U1",
        comments=None,
        autostrip=False,
    )

    # Transpose
    grid_T = grid.T

    grand_total = 0
    current_op = ""
    current_total = None

    for row in grid_T:
        if not current_op:
            current_op = row[-1][0]

        digits = row[:-1]
        col_str = "".join(digits).strip()
        if col_str:
            val = int(col_str)
            if not current_total:
                current_total = val
            else:
                current_total = do_op(current_total, val, current_op)
        else:
            current_op = ""
            grand_total += current_total
            current_total = None

    return grand_total + current_total


def task_two(input_data: str) -> int:
    lines = input_data.split("\n")
    width = len(lines[0])
    # width = max(len(line) for line in lines)

    grid = np.genfromtxt(
        io.StringIO(input_data),
        delimiter=[1] * width,
        dtype="U1",
        comments=None,
        autostrip=False,
    )

    digits_grid = grid[:-1]
    op_row = grid[-1]

    col_totals = np.zeros(width, dtype=int)

    for row in digits_grid:
        is_digit = row != " "
        if np.any(is_digit):
            col_totals[is_digit] = col_totals[is_digit] * 10 + row[is_digit].astype(int)

    is_empty_col = np.all(digits_grid == " ", axis=0)

    block_ids = np.cumsum(is_empty_col)

    grand_total = 0

    unique_blocks = np.unique(block_ids)

    for bid in unique_blocks:
        block_mask = (block_ids == bid) & (~is_empty_col)
        if not np.any(block_mask):
            continue

        numbers = col_totals[block_mask]
        ops_in_block = op_row[block_mask]

        valid_ops = ops_in_block[ops_in_block != " "]
        if len(valid_ops) == 0:
            continue
        op = valid_ops[0]

        # Calculate
        if op == "+":
            grand_total += np.sum(numbers)
        elif op == "*":
            grand_total += np.prod(numbers)

    return grand_total


def solve_task_one(input_data: str) -> int:
    lines = input_data.strip().split("\n")
    operators = np.array(lines[-1].split())
    numbers = np.array([line.split() for line in lines[:-1]], dtype=int)

    numbers_to_add = numbers[:, (operators == "+")]
    numbers_to_mult = numbers[:, (operators == "*")]

    adds = np.sum(numbers_to_add)
    mults = np.prod(numbers_to_mult, axis=0)

    return np.sum(adds) + np.sum(mults)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "input")
    with open(file_path, encoding="utf-8") as f:
        file_content = f.read()
    # Task 1
    start_time = time.perf_counter_ns()
    result = solve_task_one(file_content)
    end_time = time.perf_counter_ns()
    duration_ms = (end_time - start_time) / 1_000_000
    print(f"--- Task 1 ---")
    print(f"Result: {result}")
    print(f"Time: {duration_ms:.4f} ms")

    # Task 2
    start_time = time.perf_counter_ns()
    # result = oliip(file_content)
    result = task_two_pythonic(file_content)
    end_time = time.perf_counter_ns()
    duration_ms = (end_time - start_time) / 1_000_000
    print(f"--- Task 2 ---")
    print(f"Result: {result}")
    print(f"Time: {duration_ms:.4f} ms")


if __name__ == "__main__":
    main()
