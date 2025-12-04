import numpy as np
import numpy.typing as npt
from numba import cuda

# Types
HostArray = npt.NDArray[np.int8]

@cuda.jit(device=True)
def is_accessible(data: HostArray, flat_idx:int, width:int) -> bool:
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

      if (n_x >= 0 and n_x < width and n_y >= 0 and n_y < height):
        result += data[n_y * width + n_x]

  return result < 4



@cuda.jit
def solve_grid(data_array: HostArray, result_array: HostArray, width: int) -> None:
  thread_id = cuda.grid(1)

  # im not a roll, I leave
  if data_array[thread_id] == 0:
    return
  # check if accessible n(rolls) < 4
  if thread_id < data_array.size:
    result_array[thread_id] = is_accessible(data_array, thread_id, width)

@cuda.jit
def update_input(data_array: HostArray, result_array: HostArray) -> None:
  thread_id = cuda.grid(1)
  if result_array[thread_id] == 1:
    data_array[thread_id] = 0
  result_array[thread_id] = 0

def main() -> None:

  file_path = "/home/wsl/AOC/day4/input.txt"
  with open(file_path, "r", encoding="utf-8") as f:
    file_content = f.read()
  input = file_content.strip()

  width = input.find('\n')

  flat_list = [
    1 if c == '@' else 0
    for c in input
    if c not in ('\n')
  ]
  h_data = np.array(flat_list, dtype=np.int8)

  # send the 1D array to GPU
  d_data = cuda.to_device(h_data)
  d_result = cuda.device_array(shape=(d_data.size), dtype=np.int8)

  # prep for liftoff
  block_size = 16
  blocks_per_grid = (d_data.size + block_size - 1) // block_size
  print(f"Launching Kernel: {blocks_per_grid} blocks x {block_size} threads")

  final_sum = 0

  while True:
    solve_grid[blocks_per_grid, block_size](d_data, d_result, width)
    result_data = d_result.copy_to_host()
    current_sum = np.sum(result_data)
    print(f"current sum:  {final_sum}")
    if current_sum == 0:
      break
    final_sum += current_sum
    update_input[blocks_per_grid, block_size](d_data, d_result)

  print(f"Accessible rolls:  {final_sum}")

if __name__ == "__main__":
    main()
