import os
import requests
import openpyxl

def download_images_from_file(file_path, output_folder, failed_images_wb):
    # Tạo thư mục đầu ra nếu chưa tồn tại
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with open(file_path, 'r') as file:
        # Lấy tên của file txt (bỏ phần đuôi .txt)
        file_name = os.path.splitext(os.path.basename(file_path))[0]

        # Tạo thư mục con trong thư mục đầu ra với tên của file txt
        output_subfolder = os.path.join(output_folder, file_name)
        if not os.path.exists(output_subfolder):
            os.makedirs(output_subfolder)

        lines = file.readlines()
        for line in lines:
            # Tách tên tệp và đường dẫn ảnh từ mỗi dòng trong file txt
            image_name, image_url = line.strip().split('\t')

            # Tạo đường dẫn lưu trữ hình ảnh đã tải xuống
            save_path = os.path.join(output_subfolder, image_name.split('/')[1])

            # Tải xuống hình ảnh và lưu vào thư mục con tương ứng
            try:
                response = requests.get(image_url)
                response.raise_for_status()
                with open(save_path, 'wb') as img_file:
                    img_file.write(response.content)
                print(f"Tải xuống {image_name} thành công.")
            except requests.exceptions.RequestException as e:
                print(f"Lỗi trong quá trình tải xuống {image_name}: {e}")
                # Lưu thông tin file không tải được vào file Excel
                failed_images_wb.append([file_name, image_name, image_url, str(e)])

# Đường dẫn đến thư mục chứa các file txt
folder_path = 'yoga_dataset_links'

# Lấy danh sách các file txt trong thư mục
file_list = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

# Thư mục đầu ra để lưu trữ hình ảnh
output_folder = 'output_images'

# Khởi tạo một workbook để lưu thông tin về các file không tải được
failed_images_wb = openpyxl.Workbook()
failed_images_sheet = failed_images_wb.active
failed_images_sheet.append(['File txt', 'Tên ảnh', 'Đường dẫn', 'Lỗi'])

# Duyệt qua từng file txt và thực hiện tải xuống
for file in file_list:
    file_path = os.path.join(folder_path, file)
    download_images_from_file(file_path, output_folder, failed_images_sheet)

# Lưu workbook chứa thông tin về các file không tải được vào file Excel
failed_images_wb.save('failed_images.xlsx')
