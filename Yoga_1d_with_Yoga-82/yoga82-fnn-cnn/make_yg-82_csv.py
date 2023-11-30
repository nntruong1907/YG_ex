from def_yg82 import list_class_82, list_class_20, list_class_6, make_class_mapping, make_yg_csv, sort_yg82

boundaries_yg6 = [0, 20, 38, 46, 53, 72, len(list_class_82)]
boundaries_yg20 = [0, 3, 8, 15, 20, 25, 29, 31, 35, 38, 43, 46, 51, 53, 61, 66, 68, 72, 79, 80, len(list_class_82)]

class_name_mapping_yg_6 = make_class_mapping(list_class_82, list_class_6, boundaries_yg6)
class_name_mapping_yg_20 = make_class_mapping(list_class_82, list_class_20, boundaries_yg20)

path_data = 'data'
train_csv_path = f"{path_data}/train_data.csv"
test_csv_path = f"{path_data}/test_data.csv"

make_yg_csv(train_csv_path, class_name_mapping_yg_6, list_class_6)
make_yg_csv(test_csv_path, class_name_mapping_yg_6, list_class_6)
make_yg_csv(train_csv_path, class_name_mapping_yg_20, list_class_20)
make_yg_csv(test_csv_path, class_name_mapping_yg_20, list_class_20)

# sort_yg82(train_csv_path, list_class_82)
# sort_yg82(test_csv_path, list_class_82)