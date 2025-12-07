from numba import cuda
import numpy.typing as npt
import numpy as np
import os

HostArray = npt.NDArray[np.int8]


def get_input(filename: str) -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, filename)
    with open(file_path, "rb") as f:
        return f.read()


def main() -> None:
    lines = get_input("input").splitlines()

    trans_split = bytes.maketrans(b"^.S", b"101")
    split_map = [int(line.translate(trans_split), 2) for line in lines]

    beams = int(lines[0].translate(trans_split), 2)

    total = 0

    for y in range(1, len(lines)):
        hits = beams & split_map[y]
        continued = beams & ~hits
        if hits:
            total += hits.bit_count()
            beams = hits << 1 | hits >> 1
            beams = beams | continued
    return total


if __name__ == "__main__":
    main()
