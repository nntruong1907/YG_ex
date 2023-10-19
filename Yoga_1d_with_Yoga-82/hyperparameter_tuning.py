import os
import json
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from def_lib import load_csv_svm, preprocess_data

list_dir = [
    "./hp_tuning",
]
for d in list_dir:
    if not os.path.exists(d):
        os.makedirs(d)

now = datetime.datetime.now()  # current date and time
timestring = now.strftime("%Y-%m-%d_%H-%M-%S")  # https://strftime.org/
name_saved = "hp_" + str(timestring)

# Load dữ liệu và chuẩn bị X_train, y_train
seed = 42
test_size = 0.15
np.random.seed(seed)
tf.random.set_seed(seed)
path_data = "data"
train_path = f"{path_data}/train_data.csv"
test_path = f"{path_data}/test_data.csv"
# train_path = f"{path_data}/train_data_yg20.csv"
# test_path = f"{path_data}/test_data_yg20.csv"
# train_path = f"{path_data}/train_data_yg6.csv"
# test_path = f"{path_data}/test_data_yg6.csv"
# Load the train data
X_train, y_train, class_names = load_csv_svm(train_path)
X_test, y_test, _ = load_csv_svm(test_path)
num_classes = len(class_names)
# Pre-process data
print("Pre-process data...")
X_train = preprocess_data(X_train)
X_test = preprocess_data(X_test)

X_train = np.array(X_train)
X_test = np.array(X_test)

print("=" * 100)
print("X_train shape ", X_train.shape)
print("X_test shape ", X_test.shape)
print("y_train shape ", y_train.shape)
print("y_test shape ", y_test.shape)
print("X_train type ", type(X_train))
print("X_test type ", type(X_test))
print("y_train type ", type(y_train))
print("y_test type ", type(y_test))
print("=" * 100)

# Tạo mô hình SVM đa lớp
svm_model = SVC()

# Định nghĩa lưới tham số bạn muốn tinh chỉnh
param_grid = {
    "C": [0.1, 1, 10, 100],
    "kernel": ["linear", "poly", "rbf"],
    "degree": [1, 2, 3],
    "gamma": ["scale", "auto"] + [0.001, 0.01, 0.1, 1],
    "decision_function_shape": ["ovo", "ovr"],
    
}

# param_grid = [
#     {
#         "C": [0.1, 1, 10, 100],
#         "kernel": ["poly"],
#         "degree": [1, 2, 3],
#         "gamma": ["scale", "auto"] + [0.001, 0.01, 0.1, 1],
#     },
#     {
#         "C": [0.1, 1, 10, 100],
#         "kernel": ["rbf"],
#         "gamma": ["scale", "auto"] + [0.001, 0.01, 0.1, 1],
#     },
# ]

# Tạo đối tượng GridSearchCV
grid_search = GridSearchCV(
    svm_model, param_grid, cv=5, verbose=1, n_jobs=-1, refit=True
)

# Tiến hành tìm kiếm siêu tham số tốt nhất
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_  # Mô hình tốt nhất

# Lấy thông tin về thời gian huấn luyện cho từng tổ hợp
mean_fit_times = grid_search.cv_results_["mean_fit_time"]
std_fit_times = grid_search.cv_results_["std_fit_time"]

# In thông tin về thời gian huấn luyện cho từng tổ hợp
for mean_time, std_time, params in zip(
    mean_fit_times, std_fit_times, grid_search.cv_results_["params"]
):
    print(
        f"Tổ hợp {params} - Thời gian trung bình: {mean_time:.2f} giây, Độ lệch chuẩn: {std_time:.2f} giây"
    )

# Hiển thị kết quả tốt nhất và siêu tham số tương ứng
print(f"Độ chính xác tốt nhất: {grid_search.best_score_:.2f}")
print("Các siêu tham số tốt nhất:")
print(grid_search.best_params_)
print("Đánh giá mô hình trên tập kiểm tra:")
# Đánh giá mô hình trên tập kiểm tra
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
precision, recall, fscore, support = precision_recall_fscore_support(
    y_test, y_pred, average=None
)
accuracy = accuracy_score(y_test, y_pred)
print(f"Độ chính xác trên tập kiểm tra: {accuracy * 100:.2f}%")

# Chuyển `grid_search.cv_results_` thành một DataFrame
cv_results_df = pd.DataFrame(grid_search.cv_results_)
csv_file = "hp_tuning" + "/" + name_saved + ".csv"
cv_results_df.to_csv(csv_file, index=False)

best_params = json.dumps(grid_search.best_params_)  # Siêu tham số tốt nhất
txt_file = "hp_tuning" + "/" + "best_" + name_saved + ".txt"
with open(txt_file, "w") as file:
    file.write(best_params)
