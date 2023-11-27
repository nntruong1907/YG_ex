"""
Default command line parameter:
    python bar_chart.py --source data --type_data csv --save_image no
    
Other:
    python bar_chart.py --source yoga_pose --type_data img --save_image no

"""

import os
from datetime import datetime

import argparse
import pandas as pd
import matplotlib.pyplot as plt


def plot_bar_chart_from_csv(folder_source_path, csv_file_path, save_image_option="yes"):
    # Đọc dữ liệu từ tệp CSV
    data = pd.read_csv(folder_source_path + "/" + csv_file_path)
    labels_chart = csv_file_path
    labels = data.iloc[:, -1]
    # Các thuộc tính tùy chỉnh của biểu đồ
    plt.figure(figsize=(12, 8))
    plt.title(f"Số lượng mẫu cho mỗi giá trị nhãn trong {labels_chart}")
    # Vẽ biểu đồ cột
    ax = labels.value_counts().sort_index().plot(kind="bar", color="blue")
    # Thêm số liệu lên trên từng cột
    for p in ax.patches:
        ax.annotate(
            str(p.get_height()),
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=(0, 5),
            textcoords="offset points",
        )
    plt.xlabel("Tư thế yoga")
    plt.ylabel("Số lượng")
    plt.xticks(rotation=45, ha="right")
    plt.legend(["Số lượng mẫu"])
    plt.tight_layout()

    # Lấy ngày tháng năm hiện tại
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Tạo đường dẫn tới tập tin lưu
    file_name = os.path.splitext(os.path.basename(csv_file_path))[0]
    chart_file_path = os.path.join(
        "bar_charts", f"{file_name}_bar_chart_{current_time}.png"
    )

    data = {
        "Gia_tri_nhan": labels.value_counts().sort_index().index.tolist(),
        "So_luong": labels.value_counts().sort_index().tolist(),
    }

    # Lưu tùy chọn
    if save_image_option == "yes":
        plt.savefig(chart_file_path)
        print(f"Biểu đồ đã được lưu tại: {chart_file_path}")
        # Tạo DataFrame từ thông tin
        df = pd.DataFrame(data)
        # Lưu DataFrame vào tệp CSV
        df.to_csv(
            os.path.join("bar_charts", f"{file_name}_bar_chart_{current_time}.csv"),
            index=False,
        )

    else:
        print("Không lưu biểu đồ.")
    # Hiển thị biểu đồ
    plt.show()
    plt.close()


def plot_bar_chart_from_img(foler_path, save_image_option="yes"):
    pose_counts = {}
    # Cắt lấy tên thư mục cuối cùng từ đường dẫn
    labels_chart = os.path.basename(foler_path)
    # Đếm số lượng ảnh cho từng tư thế
    pose_folders = [
        folder
        for folder in os.listdir(foler_path)
        if os.path.isdir(os.path.join(foler_path, folder))
    ]
    for pose_folder in pose_folders:
        pose_path = os.path.join(foler_path, pose_folder)
        pose_images_count = len(os.listdir(pose_path))
        pose_counts[pose_folder] = pose_images_count
    # Vẽ biểu đồ cột chung
    plt.figure(figsize=(12, 8))
    plt.title(f"Số lượng ảnh của từng tư thế yoga trong {labels_chart}")
    plt.bar(pose_counts.keys(), pose_counts.values(), color="blue")
    # Thêm số liệu lên trên từng cột
    for pose_name, count in pose_counts.items():
        plt.text(pose_name, count + 0.1, str(count), ha="center")
    plt.xlabel("Tư thế yoga")
    plt.ylabel("Số lượng ảnh")
    plt.xticks(rotation=45, ha="right")
    plt.legend(["Số lượng ảnh"])
    plt.tight_layout()
    # Lấy ngày tháng năm hiện tại
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Tạo đường dẫn tới tập tin lưu
    chart_file_path = os.path.join(
        "bar_charts", f"{labels_chart}_img_bar_chart_{current_time}.png"
    )
    data = {
        "Gia_tri_nhan": pose_counts.keys(),
        "So_luong": pose_counts.values(),
    }
    # Lưu tùy chọn
    if save_image_option == "yes":
        plt.savefig(chart_file_path)
        print(f"Biểu đồ đã được lưu tại: {chart_file_path}")
        # Tạo DataFrame từ thông tin
        df = pd.DataFrame(data)
        # Lưu DataFrame vào tệp CSV
        df.to_csv(
            os.path.join("bar_charts", f"{labels_chart}_img_bar_chart_{current_time}.csv"),
            index=False,
        )
    else:
        print("Không lưu biểu đồ.")
    # Hiển thị biểu đồ
    plt.show()
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        "-src",
        type=str,
        default="data",
        help="folder containing file yoga poses: data / yoga_pose",
    )
    parser.add_argument(
        "--save_image",
        "-si",
        type=str,
        default="yes",
        help="chart save options: yes/no",
    )
    parser.add_argument(
        "--type_data", "-td", type=str, default="csv", help="data type options: csv/img"
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
    list_dir = ["./bar_charts"]
    for d in list_dir:
        if not os.path.exists(d):
            os.makedirs(d)

    ####################
    # CREACT BAR CHART #
    ####################

    if args.type_data in ["csv"]:
        for file in os.listdir(args.source):
            plot_bar_chart_from_csv(args.source, file, args.save_image)
    elif args.type_data in ["img"]:
        for folder in os.listdir(args.source):
            folder_path = os.path.join(args.source, folder)
            plot_bar_chart_from_img(folder_path, args.save_image)
    else:
        None
