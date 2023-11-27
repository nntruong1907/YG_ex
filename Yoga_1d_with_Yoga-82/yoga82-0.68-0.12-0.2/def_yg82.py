list_class_6 = [
    "Standing",
    "Sitting",
    "Balancing",
    "Inverted",
    "Reclining",
    "Wheel"
]

list_class_20 = [
    "Standing_Straight", "Standing_Forward_bend", "Standing_Side_bend", "Standing_Others",
    "Sitting_Normal1", "Sitting_Normal2", "Sitting_Split", "Sitting_Forward_bend", "Sitting_Twist",
    "Balancing_Front", "Balancing_Side",
    "Inverted_Legs_straight_up", "Inverted_Legs_bend",
    "Reclining_Up-facing", "Reclining_Down-facing", "Reclining_Side_facing", "Reclining_Plank_balance",
    "Wheel_Up-facing", "Wheel_Down-facing", "Wheel_Others"
]
list_class_82 = [
    '1_Eagle', '2_Tree', '3_Chair', '4_Standing_Forward_Bend', '5_Wide-Legged_Forward_Bend', 
    '6_Dolphin', '7_Downward-Facing_Dog', '8_Intense_Side_Stretch', '9_Half_Moon', 
    '10_Extended_Revolved_Triangle', '11_Extended_Revolved_Side_Angle', '12_Gate', '13_Warrior_I', 
    '14_Reverse_Warrior', '15_Low_Lunge', '16_Warrior_II', '17_Warrior_III', '18_Lord_of_the_Dance', 
    '19_Standing_big_toe_hold', '20_Standing_Split', '21_Sitting', '22_Bound_Angle', '23_Garland', 
    '24_Staff', '25_Noose', '26_Cow_Face', '27_Hero_and_thunderbolt', '28_Bharadvajas_Twist', 
    '29_Half_Lord_of_the_Fishes', '30_Split', '31_Wide-Angle_Seated_Forward_Bend', '32_Head-to-Knee_Forward_Bend', 
    '33_Revolved_Head-to-Knee', '34_Seated_Forward_Bend', '35_Tortoise', '36_Shooting_bow', '37_Heron', 
    '38_King_Pigeon', '39_Crane_Crow', '40_Shoulder-Pressing', '41_Cockerel', '42_Scale', '43_Firefly', 
    '44_Side_Crane_Crow', '45_Eight-Angle', '46_Sage_Koundinya', '47_Handstand', '48_Headstand', '49_Shoulderstand', 
    '50_Feathered_Peacock', '51_Legs-Up-the-Wall', '52_Plow', '53_Scorpion', '54_Corpse', '55_Fish', '56_Happy_Baby', 
    '57_Reclining_Hand-to-Big-Toe', '58_Wind_Relieving', '59_Reclining_cobbler', '60_Reclining_hero', '61_Yogic_sleep', 
    '62_Cobra', '63_Frog', '64_Locust', '65_Child', '66_Extended_Puppy', '67_Side-Reclining_Leg_Lift', '68_Side_Plank', 
    '69_Dolphin_Plank', '70_Low_Plank_Four-Limbed_Staff', '71_Plank', '72_Peacock', '73_Upward_Bow', 
    '74_Upward_Facing_Two-Foot_Staff', '75_Upward_Plank', '76_Pigeon', '77_Bridge', '78_Wild_Thing', '79_Camel_Pose', 
    '80_Cat_Cow', '81_Boat', '82_Bow'
]

import pandas as pd

def make_class_mapping(list_class_82, list_class, boundaries):
    class_name_mapping = {}
    for i in range(1, len(boundaries)):
        start = boundaries[i-1]
        end = boundaries[i]
        folder_name = list_class[i-1]

        for class_name in list_class_82[start:end]:
            class_name_mapping[class_name] = folder_name
    return class_name_mapping
def make_yg_csv(csv_path, class_name_mapping, list_class):
    df = pd.read_csv(csv_path)
    df['class_name'] = df['class_name'].replace(class_name_mapping)
    for i in range(len(list_class)):
        df.loc[df['class_name'] == list_class[i], 'class_no'] = i
    df = df.sort_values(by='class_no')
    df.to_csv(f"{csv_path.split('.')[0]}_yg{len(list_class)}.csv", index = False)
    print(f"Lưu thành công {csv_path.split('.')[0]}_yg{len(list_class)}.csv")

    return df

def sort_yg82(csv_path, list_class_82):
    df = pd.read_csv(csv_path)
    for i in range(len(list_class_82)):
        df.loc[df['class_name'] == list_class_82[i], 'class_no'] = i
    df = df.sort_values(by='class_no')

    df.to_csv(f"{csv_path.split('.')[0]}_yg{len(list_class_82)}.csv", index = False)

    print(f"Lưu thành công {csv_path.split('.')[0]}_yg{len(list_class_82)}.csv")

    return df

