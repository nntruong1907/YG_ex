"""
Default command line parameter:
    python split_data.py --source Yoga-82 --destination yoga_pose --test_split 0.2
    
"""

import os
import random
import shutil

import argparse


def split_into_train_test(images_origin, images_dest, test_split):
    """Splits a directory of sorted images into training and test sets.

    Args:
      images_origin: Path to the directory with your images. This directory
        must include subdirectories for each of your labeled classes. For example:
        yoga_poses/
        |__ Chair_Pose/
            |______ 00000128.jpg
            |______ 00000181.jpg
            |______ ...
        |__ Cobra_Pose/
            |______ 00000243.jpg
            |______ 00000306.jpg
            |______ ...
        ...
      images_dest: Path to a directory where you want the split dataset to be
        saved. The results looks like this:
        split_yoga_poses/
        |__ train/
            |__ Chair_Pose/
                |______ 00000128.jpg
                |______ ...
        |__ test/
            |__ Cobra_Pose/
                |______ 00000181.jpg
                |______ ...
      test_split: Fraction of data to reserve for test (float between 0 and 1).
    """
    _, dirs, _ = next(os.walk(images_origin))

    TRAIN_DIR = os.path.join(images_dest, "train")
    TEST_DIR = os.path.join(images_dest, "test")
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)
    total_files = 0
    if os.path.exists('slit_data.txt'):
        os.remove('slit_data.txt')
        
    for dir in dirs:
        # Get all filenames for this dir, filtered by filetype
        filenames = os.listdir(os.path.join(images_origin, dir))
        filenames = [
            os.path.join(images_origin, dir, f)
            for f in filenames
            if (
                f.endswith(".png")
                or f.endswith(".jpg")
                or f.endswith(".jpeg")
                or f.endswith(".bmp")
            )
        ]
        # Shuffle the files, deterministically
        filenames.sort()
        random.seed(42)
        random.shuffle(filenames)

        # Divide them into train/test dirs
        os.makedirs(os.path.join(TEST_DIR, dir), exist_ok=True)
        os.makedirs(os.path.join(TRAIN_DIR, dir), exist_ok=True)
        test_count = int(len(filenames) * test_split)
        train_count = int(len(filenames) - test_count)
        for i, file in enumerate(filenames):
            if i < test_count:
                destination = os.path.join(TEST_DIR, dir, os.path.split(file)[1])
            else:
                destination = os.path.join(TRAIN_DIR, dir, os.path.split(file)[1])
            shutil.copyfile(file, destination)
        data = f'Moved {test_count} of {len(filenames)} from class "{dir}" into test\nMoved {train_count} of {len(filenames)} from class "{dir}" into train \n'
        print(data)

        with open(f"slit_data.txt", "a") as f:
            f.writelines(data)

        total_files += len(filenames)
    print(f'Your split dataset is in "{images_dest}" with test split {test_split}')
    print(f"Total file: {total_files}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        "-src",
        type=str,
        default="Yoga-82",
        help="folder containing images yoga poses",
    )
    parser.add_argument(
        "--destination",
        "-dest",
        type=str,
        default="yoga_pose",
        help="folder for training and test sets",
    )
    parser.add_argument("--test_split", type=float, default=0.26, help="test set ratio")
    args = parser.parse_args()
    # Lấy danh sách tất cả các tham số đã đăng ký
    registered_args = parser._actions
    # Hiển thị tên và mô tả của các tham số
    print("Description:")
    for arg in registered_args:
        if arg.help is not argparse.SUPPRESS:
            default = arg.default if arg.default is not argparse.SUPPRESS else None
            print(f"\t --{arg.dest}\t {default}\t {arg.help}")
    print("Setup Configuration:")
    for name, value in parser.parse_args()._get_kwargs():
        print(f"\t{name} {value}")
    print("=" * 100)

    ####################
    # SPLIT DATA INTO TRAIN TEST #
    ####################
    split_into_train_test(args.source, args.destination, args.test_split)
