import numpy as np
import numpy.typing as npt
from numba import cuda
import sys
import time

# Types
HostArray = npt.NDArray[np.int8]


def render_frame(d_data, width):
    host_grid = d_data.copy_to_host()
    height = host_grid.size // width
    grid_2d = host_grid.reshape((height, width))

    lines = []
    for row in grid_2d:
        line = "".join(["█" if cell == 1 else "·" for cell in row])
        lines.append(line)
    # \033[H moves cursor to top-left
    # \033[J clears everything below
    output = "\033[H\033[J" + "\n".join(lines)
    sys.stdout.write(output + "\n")
    sys.stdout.flush()


@cuda.jit(device=True)
def is_accessible(data: HostArray, flat_idx: int, width: int) -> bool:
    result = 0

    current_x = flat_idx % width
    current_y = flat_idx // width
    height = data.size // width

    for dx in range(-1, 2):
        for dy in range(-1, 2):
            if dx == 0 and dy == 0:
                continue
            n_x = current_x + dx
            n_y = current_y + dy

            if n_x >= 0 and n_x < width and n_y >= 0 and n_y < height:
                n_idx = n_y * width + n_x
                if n_idx < data.size:
                    result += data[n_idx]

    return result < 4


@cuda.jit
def solve_grid(
    data_array: HostArray, result_array: HostArray, width: int, counter: npt.NDArray
) -> None:
    thread_id = cuda.grid(1)

    # im a leftover, I leave
    if thread_id >= data_array.size:
        return
    # im not a roll, I leave
    if data_array[thread_id] == 0:
        return
    # check if accessible n(rolls) < 4
    if thread_id < data_array.size:
        if is_accessible(data_array, thread_id, width):
            result_array[thread_id] = 1
            cuda.atomic.add(counter, 0, 1)


@cuda.jit
def update_input(data_array: HostArray, result_array: HostArray) -> None:
    thread_id = cuda.grid(1)
    if thread_id >= data_array.size:
        return
    # remove the 'paper roll' if it was accessible, and reset the result_array to 0s
    if result_array[thread_id] == 1:
        data_array[thread_id] = 0
    result_array[thread_id] = 0


def main() -> None:

    t0_setup = time.perf_counter()
    file_path = "/home/wsl/AOC/day4/input.txt"
    with open(file_path, "r", encoding="utf-8") as f:
        file_content = f.read()
    input = file_content.strip()

    width = input.find("\n")

    flat_list = [1 if c == "@" else 0 for c in input if c not in ("\n")]
    h_data = np.array(flat_list, dtype=np.int8)

    # send input array
    d_data = cuda.to_device(h_data)
    d_result = cuda.device_array(shape=(d_data.size), dtype=np.int8)
    d_counter = cuda.to_device(np.zeros(1, dtype=np.int32))

    # prep for liftoff
    block_size = 128
    blocks_per_grid = (d_data.size + block_size - 1) // block_size

    last_sum = np.zeros(1, dtype=np.int32)

    t1_setup = time.perf_counter()
    print(f"Setup & Upload Time: {(t1_setup - t0_setup) * 1000:.4f} ms")

    # warmup for correct timing, compile the kernel code
    solve_grid[1, 1](d_data, d_result, width, d_counter)
    update_input[1, 1](d_data, d_result)
    cuda.synchronize()

    print(f"Launching Kernel: {blocks_per_grid} blocks x {block_size} threads")

    t0_compute = time.perf_counter()

    iterations = 0
    # keep all the data on the GPU and run until there is no more moves
    while True:
        iterations += 1
        solve_grid[blocks_per_grid, block_size](d_data, d_result, width, d_counter)
        current_sum = d_counter.copy_to_host()
        if current_sum == last_sum:
            break

        # render_frame(d_data, width)
        # time.sleep(0.05)
        # print(f"current sum:  {current_sum}")
        update_input[blocks_per_grid, block_size](d_data, d_result)
        last_sum = current_sum

    t1_compute = time.perf_counter()

    total_time = (t1_compute - t0_compute) * 1000
    print(f"Compute Time:      {total_time:.4f} ms")
    print(f"Total Iterations:  {iterations}")
    print(f"Avg Time per Iter: {total_time / iterations:.4f} ms")

    print(f"Accessible rolls:  {current_sum}")


if __name__ == "__main__":
    main()
