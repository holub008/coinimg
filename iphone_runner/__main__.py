"""
A runner for series of images taken with an iphone:

1. taking images on an iphone (images labeled like IMG_2848.HEIC)
2. images are taken in portrait (i.e. need a 270 deg rotation)
3. images are taken in sequence obverse, reverse for each coin
"""
import sys
import os
from src import concatenate


def main(in_dir: str, out_dir: str) -> None:
    all_files = [fn for fn in os.listdir(in_dir) if not fn.startswith('.')]
    if len(all_files) % 2 != 0:
        raise ValueError('Input dir must contain even number of files (i.e. pairs)')

    ordered_files = sorted(all_files)
    for file_ix in range(0, len(ordered_files), 2):
        obverse_path = os.path.join(in_dir, ordered_files[file_ix])
        reverse_path = os.path.join(in_dir, ordered_files[file_ix + 1])

        out_file = concatenate(obverse_path, reverse_path, rotate=270)
        out_path = os.path.join(out_dir, f'{ordered_files[file_ix]}_combined.jpg')
        out_file.save(out_path)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError('Give two args for in dir, out dir')
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.isdir(input_dir):
        raise ValueError('Input dir is not valid')

    if not os.path.isdir(output_dir):
        print('Making output dir...')
        os.mkdir(output_dir)

    main(input_dir, output_dir)