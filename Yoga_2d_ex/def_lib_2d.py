import os
import datetime
import psutil
import shutil
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
from contextlib import redirect_stdout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
import truongmodel_2d

"""
================================================= DEF PRECROCESSING DATA ================================================
"""


def make_df(images_folder_path, folder_csv_path):
    labels_folder = os.path.basename(images_folder_path)
    csv_out_path = folder_csv_path + "/" + labels_folder + "_data.csv"
    image_paths = []
    labels = []
    classes = os.listdir(images_folder_path)
    for c in classes:
        class_folder = images_folder_path + "/" + c
        for image in os.listdir(class_folder):
            image_path = class_folder + "/" + image
            image_paths.append(image_path)
            labels.append(c)

    print(f"Number sample of {labels_folder}:", len(image_paths))
    df = pd.DataFrame({"image_path": image_paths, "class_name": labels})
    df.to_csv(csv_out_path, index=False)


def load_csv(csv_path):
    df = pd.read_csv(csv_path)

    return df


def balance(df, n, working_dir, img_size):
    df = df.copy()
    print("Initial length of dataframe is ", len(df))
    aug_dir = os.path.join(working_dir, "aug")  # directory to store augmented images
    if os.path.isdir(aug_dir):  # start with an empty directory
        shutil.rmtree(aug_dir)
    os.mkdir(aug_dir)
    for label in df["class_name"].unique():
        dir_path = os.path.join(aug_dir, label)
        os.mkdir(dir_path)  # make class directories within aug directory
    # create and store the augmented images
    total = 0
    gen = ImageDataGenerator(
        # rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    groups = df.groupby("class_name")  # group by class
    for label in df["class_name"].unique():  # for every class
        group = groups.get_group(
            label
        )  # a dataframe holding only rows with the specified label
        sample_count = len(group)  # determine how many samples there are in this class
        if sample_count < n:  # if the class has less than target number of images
            aug_img_count = 0
            delta = n - sample_count  # number of augmented images to create
            target_dir = os.path.join(
                aug_dir, label
            )  # define where to write the images
            msg = "{0:40s} for class {1:^30s} creating {2:^5s} augmented images".format(
                " ", label, str(delta)
            )
            print(msg, "\r", end="")  # prints over on the same line
            aug_gen = gen.flow_from_dataframe(
                group,
                x_col="image_path",
                y_col=None,
                target_size=(img_size, img_size),
                class_mode=None,
                batch_size=1,
                shuffle=False,
                save_to_dir=target_dir,
                save_prefix="aug-",
                color_mode="rgb",
                save_format="jpg",
            )
            while aug_img_count < delta:
                images = next(aug_gen)
                aug_img_count += len(images)
            total += aug_img_count
    print("Total Augmented images created= ", total)
    # create aug_df and merge with train_df to create composite training set ndf
    aug_fpaths = []
    aug_labels = []
    classlist = os.listdir(aug_dir)
    for klass in classlist:
        classpath = os.path.join(aug_dir, klass)
        flist = os.listdir(classpath)
        for f in flist:
            fpath = os.path.join(classpath, f)
            aug_fpaths.append(fpath)
            aug_labels.append(klass)
    Fseries = pd.Series(aug_fpaths, name="image_path")
    Lseries = pd.Series(aug_labels, name="class_name")
    aug_df = pd.concat([Fseries, Lseries], axis=1)
    df = pd.concat([df, aug_df], axis=0).reset_index(drop=True)
    print("Length of augmented dataframe is now ", len(df))
    return df, total


def make_gens(batch_size, train_df, test_df, valid_df, img_size):
    trgen = ImageDataGenerator(rescale=1.0 / 255)
    t_and_v_gen = ImageDataGenerator(rescale=1.0 / 255)
    msg = "{0:70s} for train generator".format(" ")
    print(msg, "\r", end="")  # prints over on the same line
    train_gen = trgen.flow_from_dataframe(
        train_df,
        x_col="image_path",
        y_col="class_name",
        target_size=(img_size, img_size),
        class_mode="categorical",
        color_mode="rgb",
        shuffle=True,
        batch_size=batch_size,
    )
    msg = "{0:70s} for valid generator".format(" ")
    print(msg, "\r", end="")  # prints over on the same line
    valid_gen = t_and_v_gen.flow_from_dataframe(
        valid_df,
        x_col="image_path",
        y_col="class_name",
        target_size=(img_size, img_size),
        class_mode="categorical",
        color_mode="rgb",
        shuffle=False,
        batch_size=batch_size,
    )
    # for the test_gen we want to calculate the batch size and test steps such that batch_size X test_steps= number of samples in test set
    # this insures that we go through all the sample in the test set exactly once.
    length = len(test_df)
    test_batch_size = sorted(
        [
            int(length / n)
            for n in range(1, length + 1)
            if length % n == 0 and length / n <= 80
        ],
        reverse=True,
    )[0]
    test_steps = int(length / test_batch_size)
    msg = "{0:70s} for test generator".format(" ")
    print(msg, "\r", end="")  # prints over on the same line
    test_gen = t_and_v_gen.flow_from_dataframe(
        test_df,
        x_col="image_path",
        y_col="class_name",
        target_size=(img_size, img_size),
        class_mode="categorical",
        color_mode="rgb",
        shuffle=False,
        batch_size=test_batch_size,
    )
    # from the generator we can get information we will need later
    classes = list(train_gen.class_indices.keys())
    class_indices = list(train_gen.class_indices.values())
    class_count = len(classes)
    labels = test_gen.labels
    print(
        "test batch size: ",
        test_batch_size,
        "  test steps: ",
        test_steps,
        " number of classes : ",
        class_count,
    )
    return train_gen, test_gen, valid_gen, test_batch_size, test_steps, classes


def make_test_gen(test_df, img_size):
    length = len(test_df)
    test_batch_size = sorted(
        [
            int(length / n)
            for n in range(1, length + 1)
            if length % n == 0 and length / n <= 80
        ],
        reverse=True,
    )[0]
    test_steps = int(length / test_batch_size)
    t_gen = ImageDataGenerator(rescale=1.0 / 255)
    test_gen = t_gen.flow_from_dataframe(
        test_df,
        x_col="image_path",
        y_col="class_name",
        target_size=(img_size, img_size),
        class_mode="categorical",
        color_mode="rgb",
        shuffle=False,
        batch_size=test_batch_size,
    )

    return test_gen


def show_image_samples(gen):
    t_dict = gen.class_indices
    classes = list(t_dict.keys())
    images, labels = next(gen)  # get a sample batch from the generator
    plt.figure(figsize=(25, 25))
    length = len(labels)
    if length < 25:  # show maximum of 25 images
        r = length
    else:
        r = 25
    for i in range(r):
        plt.subplot(5, 5, i + 1)
        image = images[i]
        plt.imshow(image)
        index = np.argmax(labels[i])
        class_name = classes[index]
        plt.title(class_name, color="blue", fontsize=18)
        plt.axis("off")
    plt.show()


"""
==================================================== DEF SAVE FILE ======================================================
"""


def save_model_summary(folder_saved, name_saved, model):
    model_name = name_saved.split("_")[0]
    model_summary_path = f"{folder_saved}/summary_{name_saved}.txt"
    if model_name == "svm":
        data = f"C: {model.C}, Kernel: {model.kernel}, Gamma: {model.gamma}, Degree: {model.degree}, Decision function: {model.decision_function_shape}"
        with open(model_summary_path, "w") as f:
            f.write(data)
    else:
        with open(model_summary_path, "w") as f:
            with redirect_stdout(f):
                model.summary()


def save_log_txt(folder_saved, name_saved, history_callback):
    ep_arr = range(1, len(history_callback.history["accuracy"]) + 1, 1)
    train_acc = history_callback.history["accuracy"]
    val_acc = history_callback.history["val_accuracy"]
    train_loss = history_callback.history["loss"]
    val_loss = history_callback.history["val_loss"]
    title_cols = np.array(
        ["epoch", "train_acc", "valid_acc", "train_loss", "valid_loss"]
    )
    res = (ep_arr, train_acc, val_acc, train_loss, val_loss)
    res = np.transpose(res)
    combined_res = np.array(np.vstack((title_cols, res)))
    np.savetxt(
        f"{folder_saved}/{name_saved}.txt", combined_res, fmt="%s", delimiter="\t"
    )


def save_log_csv(folder_saved, name_saved, history_callback):
    hist_df = pd.DataFrame(history_callback.history)
    hist_df.index += 1
    hist_df.to_csv(f"{folder_saved}/{name_saved}.csv", index_label="epoch")


def save_measure(
    folder_saved,
    name_saved,
    samples_training,
    samples_validation,
    samples_test,
    elapsed,
    consumes_memory,
    history_callback,
):
    num_epoch = len(history_callback.history["accuracy"])
    best_train_accuracy = max(history_callback.history["accuracy"])
    best_train_epoch = (
        history_callback.history["accuracy"].index(best_train_accuracy) + 1
    )
    best_val_accuracy = max(history_callback.history["val_accuracy"])
    best_val_epoch = (
        history_callback.history["val_accuracy"].index(best_val_accuracy) + 1
    )
    model_training = "Samples training set: " + str(
        samples_training
    ) + "\n" + "Samples validation set: " + str(
        samples_validation
    ) + "\n" + "Samples test set: " + str(
        samples_test
    ) + "\n" "Running time: " + str(
        elapsed
    ) + "\n" + "Memory consumed during training (%): " + str(
        consumes_memory
    ) + "\n" + "Total epochs: " + str(
        num_epoch
    ) + "\n" + "Best train accuracy: " + str(
        best_train_accuracy
    ) + " / epoch: " + str(
        best_train_epoch
    ) + "\n" + "Best val accuracy: " + str(
        best_val_accuracy
    ) + " / epoch: " + str(
        best_val_epoch
    )
    print(model_training)
    with open(f"{folder_saved}/measure_{name_saved}.txt", "w") as f:
        f.writelines(model_training)
    f.close()


def save_evaluate(folder_saved, name_saved, model, df_train, df_test):
    loss_train, accuracy_train = model.evaluate(df_train)
    loss_test, accuracy_test = model.evaluate(df_test)
    data_eval = (
        "LOSS TRAIN: "
        + str(loss_train)
        + " / ACCURACY TRAIN: "
        + str(accuracy_train)
        + "\n"
        + "LOSS TEST: "
        + str(loss_test)
        + " / ACCURACY TEST: "
        + str(accuracy_test)
    )
    print(data_eval)

    with open(f"{folder_saved}/evaluation_{name_saved}.txt", "w") as file:
        file.writelines(data_eval)
    file.close()


def plot_acc(folder_saved, name_saved, history_callback):
    plt.plot(history_callback.history["accuracy"])
    plt.plot(history_callback.history["val_accuracy"])
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["train", "val"], loc="best")
    plt.savefig(f"{folder_saved}/accuracy_{name_saved}.png")
    plt.close()


def plot_loss(folder_saved, name_saved, history_callback):
    plt.plot(history_callback.history["loss"])
    plt.plot(history_callback.history["val_loss"])
    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["train", "val"], loc="best")
    plt.savefig(f"{folder_saved}/loss_{name_saved}.png")
    plt.close()


def plot_confusion_matrix(
    plot_confusion_matrix_path,
    cm,
    classes,
    normalize=False,
    title="Confusion matrix",
    cmap=plt.cm.Blues,
):
    """Plots the confusion matrix."""
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    fig = plt.gcf()
    fig.set_size_inches(12.5, 10.5)  # Đặt kích thước hình ảnh
    plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.25)
    fig.savefig(plot_confusion_matrix_path)  # Lưu hình ảnh
    plt.close()


"""
========================================================= DEF EX ========================================================
"""


def load_datafame(path_data, test_size):
    seed = 42
    np.random.seed(seed)
    tf.random.set_seed(seed)

    train_path = f"{path_data}/train_data.csv"
    test_path = f"{path_data}/test_data.csv"

    df = load_csv(train_path)
    train_df, valid_df = train_test_split(
        df, stratify=df["class_name"], test_size=test_size, random_state=seed
    )
    test_df = load_csv(test_path)

    return train_df, valid_df, test_df


def run_exp(
    name_saved,
    model_name,
    train_df,
    valid_df,
    test_df,
    img_size,
    epochs,
    batch_size,
    e_patience,
    num_conv_layers_per_maxpool=2,
    num_maxpools=1,
    num_dense_layers=5,
):
    seed = 42
    np.random.seed(seed)
    tf.random.set_seed(seed)

    batch_size = 32
    (
        train_gen,
        test_gen,
        valid_gen,
        test_batch_size,
        test_steps,
        class_names,
    ) = make_gens(batch_size, train_df, test_df, valid_df, img_size)

    num_classes = len(class_names)

    samples_training = len(train_gen.filepaths)
    samples_validation = len(valid_gen.filepaths)
    samples_test = len(test_gen.filepaths)

    ###################
    # Construct model #
    ###################
    if model_name in ["fcnn2d"]:
        name_saved = name_saved.replace("fcnn2d", "fcnn2d" + f"-{num_dense_layers}dense")
        model = truongmodel_2d.fcnn2d_model(
            input_shape=(img_size, img_size, 3),
            num_classes=num_classes,
            dropout_fc=0.2,
            num_dense_layers=num_dense_layers,  # 5
        )

    elif model_name in ["conv2d"]:
        name_saved = name_saved.replace(
            "conv2d",
            "conv2d"
            + f"-{num_maxpools}maxpool-{num_conv_layers_per_maxpool}convpermaxpool",
        )
        model = truongmodel_2d.conv2d_model(
            input_shape=(img_size, img_size, 3),
            num_classes=num_classes,
            num_maxpools=num_maxpools,  # 1
            num_conv_layers_per_maxpool=num_conv_layers_per_maxpool,  # 2
            dropout_cnn=0.2,
            dropout_f=0.5,
        )

    # Lưu kiến trúc mô hình
    save_model_summary("model_training", name_saved, model)

    ####################
    # Network training #
    ####################
    checkpoint_path = f"save_models/weights.best_{name_saved}.hdf5"
    checkpoint = keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor="val_accuracy",
        verbose=1,
        save_best_only=True,
        mode="max",
    )
    earlystopping = keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=e_patience,
    )
    # Lấy thông tin bộ nhớ trước khi huấn luyện bắt đầu
    memory_info_before = psutil.virtual_memory()
    memory_usage_before = memory_info_before.percent
    start = datetime.datetime.now()
    history_callback = model.fit(
        train_gen,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        validation_data=valid_gen,
        callbacks=[checkpoint, earlystopping],
    )
    # Lấy thông tin bộ nhớ sau khi huấn luyện kết thúc
    memory_info_after = psutil.virtual_memory()
    memory_usage_after = memory_info_after.percent
    elapsed = datetime.datetime.now() - start
    consumes_memory = memory_usage_after - memory_usage_before
    # memory_usage = round(consumes_memory, 5)
    print(f"Memory used before training: {memory_usage_before}%")
    print(f"Memory used after training: {memory_usage_after}%")
    print(f"Memory used: {consumes_memory} %")
    print("=" * 100)

    ####################
    # Save log and trained model #
    ####################
    save_log_txt("log", name_saved, history_callback)
    save_log_csv("log", name_saved, history_callback)
    # Save model
    model.save(f"save_models/{name_saved}.h5")
    model_json = model.to_json()
    with open(f"save_models/{name_saved}.json", "w") as json_file:
        json_file.write(model_json)
    json_file.close()
    ####################
    #  model training  #
    ####################
    save_measure(
        "model_training",
        name_saved,
        samples_training,
        samples_validation,
        samples_test,
        elapsed,
        consumes_memory,
        history_callback,
    )

    ####################
    #  draw acc and loss  #
    ####################
    plot_acc("figures", name_saved, history_callback)
    plot_loss("figures", name_saved, history_callback)

    ####################
    #    EVALUATION    #
    ####################
    save_evaluate(
        "statistics",
        name_saved,
        model,
        train_gen,
        test_gen,
    )
    ####################
    # confusion matrix #
    ####################
    model = tf.keras.models.load_model(f"save_models/{name_saved}.h5")

    y_true = test_gen.labels
    class_names = list(test_gen.class_indices.keys())
    preds = model.predict(test_gen)

    ytrue = np.array(y_true)
    ypred = np.argmax(preds, axis=1)

    f1score = f1_score(ytrue, ypred, average="weighted") * 100
    plot_confusion_matrix_path = f"figures/confusion_matrix_{name_saved}.png"
    plot_confusion_matrix_nor_path = f"figures/confusion_matrix_nor_{name_saved}.png"
    # Plot the confusion matrix
    cm = confusion_matrix(ytrue, ypred)
    plot_confusion_matrix(
        plot_confusion_matrix_path,
        cm,
        class_names,
        title="Confusion Matrix of Yoga Pose Model",
    )

    plot_confusion_matrix(
        plot_confusion_matrix_nor_path,
        cm,
        class_names,
        normalize=True,
        title="Normalized Confusion Matrix of Yoga Pose Model",
    )

    # Print the classification report
    print(
        "\nClassification Report:\n",
        classification_report(ytrue, ypred, target_names=class_names, zero_division=0),
    )

    with open(f"statistics/classification_report_{name_saved}.txt", "w") as f:
        f.writelines(
            classification_report(
                ytrue, ypred, target_names=class_names, zero_division=0
            )
        )
    f.close()


"""
======================================================== DEF TEST =======================================================
"""


def find_newest_model_with_prefix(folder_path, prefix):
    file_list = os.listdir(folder_path)
    matching_files = [file for file in file_list if file.startswith(prefix)]

    if not matching_files:
        return None

    latest_file = max(
        matching_files, key=lambda x: os.path.getctime(os.path.join(folder_path, x))
    )
    # print(name_without_extension)
    name_without_extension = latest_file.rsplit(".", 1)[0]
    # print(latest_file)

    return name_without_extension
