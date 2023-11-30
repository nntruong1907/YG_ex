"""
Default command line parameter:
    python make_csv.py --source yoga_pose --save data
    
"""

import os
import argparse

from def_lib import MoveNetPreprocessor

def make_file_skeleton(images_folder_path, folder_csv_path):
    labels_folder = os.path.basename(images_folder_path)
    csv_out_path = folder_csv_path + "/" + labels_folder + "_data.csv"
    preprocessor = MoveNetPreprocessor(
        images_in_folder=images_folder_path,
        csvs_out_path=csv_out_path,
    )
    preprocessor.process()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        "-src",
        type=str,
        default="yoga_pose",
        help="folder containing data for yoga poses",
    )
    parser.add_argument(
        "--save",
        "-s",
        type=str,
        default="data",
        help="folder to save extracted skeleton data",
    )
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

    # Tạo các thư mục cần thiết
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    ####################
    # CREACT SKELECTON FILE #
    ####################
    for folder in os.listdir(args.source):
        folder_path = os.path.join(args.source, folder)
        make_file_skeleton(folder_path, args.save)
