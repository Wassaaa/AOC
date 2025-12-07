import os
import time
import sys
from collections import defaultdict
from functools import cache


def get_input(filename: str) -> bytes:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, filename)
    with open(file_path, "rb") as f:
        return f.read()


def task_one(lines: bytes) -> int:
    # setup
    trans_split = bytes.maketrans(b"^.S", b"101")
    split_map = [int(line.translate(trans_split), 2) for line in lines]
    beams = int(lines[0].translate(trans_split), 2)
    total = 0

    # sim
    for y in range(1, len(lines)):
        hits = beams & split_map[y]
        continued = beams & ~hits
        # bit shift if hit
        if hits:
            total += hits.bit_count()
            beams = hits << 1 | hits >> 1
            beams = beams | continued
        # no hits, continue down
    return total


def task_two(lines: bytes) -> int:
    # setup, list of sets for x of splitters
    split_map = [{i for i, c in enumerate(line) if c in b"^S"} for line in lines]
    # beams is a dictionary of {x: count} to track multple beams at the same spot
    beams = {x: 1 for x in split_map[0]}
    total_universes = sum(beams.values()) if beams else 1

    # sim
    for y in range(1, len(lines)):
        next_beams = defaultdict(int)
        splitters = split_map[y]

        # handle hits
        for x, count in beams.items():
            if x in splitters:
                total_universes += count
                next_beams[x - 1] += count
                next_beams[x + 1] += count
            else:
                next_beams[x] += count
        beams = next_beams
    return total_universes


data = open("input").read().strip().split("\n")
y_max = len(data)
x_max = len(data[0])
x_start, y_start = data[0].find("S"), 0


@cache
def req(x, y):
    if not (0 <= x < x_max) or not (0 <= y < y_max):
        return 1

    if data[y][x] == ".":
        return req(x, y + 1)
    else:
        return req(x - 1, y + 2) + req(x + 1, y + 2)


def task_two_cache() -> int:
    return req(x_start, y_start)


def main() -> None:
    lines = get_input("input").splitlines()

    # --- Run Task 1 ---
    t0 = time.perf_counter()
    splits = task_one(lines)
    t1 = time.perf_counter()

    print(f"Task 1 Result: {splits}")
    print(f"Task 1 Time  : {(t1 - t0) * 1000:.4f} ms")

    # --- Run Task 2 ---
    t0 = time.perf_counter()
    # universes = task_two(lines)
    universes = task_two_cache()
    t1 = time.perf_counter()

    print(f"Task 2 Result: {universes}")
    print(f"Task 2 Time  : {(t1 - t0) * 1000:.4f} ms")


if __name__ == "__main__":
    main()
