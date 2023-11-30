"""
Default command line parameter:
    python train_ex.py --model fnn1d --source data --test_size 0.15 --epochs 80 --batch_size 32 --e_patience 10

Other:
    python train_ex.py --model conv1d --source data --test_size 0.15 --epochs 80 --batch_size 32 --e_patience 10
    python train_ex.py --model svm --source data --C 100 --kernel rbf --gamma scale --df ovo --k_fold 5
    python train_ex.py --model svm --source data --C 100 --kernel rbf --gamma scale --d 1 --df ovo --k_fold 5

"""

import os
import datetime
import json

import argparse

from def_lib import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="svm",
        help="model name: svm / fnn1d / conv1d",
    )
    parser.add_argument(
        "--source",
        "-src",
        type=str,
        default="data",
        help="folder containing file skeleton .csv",
    )
    parser.add_argument(
        "--test_size", type=float, default=0.15, help="validation set ratio"
    )
    parser.add_argument(
        "--epochs", "-ep", type=int, default=80, help="number of epochs for training"
    )
    parser.add_argument("--batch_size", "-bs", type=int, default=32, help="batch size")
    parser.add_argument(
        "--e_patience",
        "-epp",
        type=int,
        default=10,
        help="set the number of epoch patient, after this number of epoch if the results do not improve then stop learning",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=100,
        help="Regularization Parameter: 0.1 /1 /10 /100",
    )
    parser.add_argument("--kernel", default="rbf", help="kernel: linear /poly /rbf")
    parser.add_argument(
        "--gamma",
        default="scale",
        help="gamma: scale /auto /0.001 /0.01 /0.1 /1",
    )
    parser.add_argument(
        "--d",
        default=1,
        help="degree for 'poly' kernel",
    )
    parser.add_argument(
        "--df",
        default="ovo",
        help="decision function shape: ovo /ovr",
    )
    parser.add_argument(
        "--k_fold",
        default=5,
        help="cross-validation with k folds",
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

    list_dir = [
        "./log",
        "./model_training",
        "./figures",
        "./save_models",
        "./statistics",
    ]
    for d in list_dir:
        if not os.path.exists(d):
            os.makedirs(d)

    ###################
    # Model training  #
    ###################

    if args.model not in ["svm"]:
        (
            processed_X_train,
            y_train,
            processed_X_val,
            y_val,
            processed_X_test,
            y_test,
            class_names,
        ) = load_data(args.source, args.test_size)
        if args.model in ["fnn1d"]:
            max_dense_layers = 10
            for num_dense in range(1, max_dense_layers + 1):
                print("=" * 150)
                print(f"{num_dense} dense layer")
                now = datetime.datetime.now()
                timestring = now.strftime("%Y-%m-%d_%H-%M-%S")  # https://strftime.org/
                name_saved = (
                    str(args.model)
                    + "_tsize-"
                    + str(args.test_size)
                    + "_ep-"
                    + str(args.epochs)
                    + "_bs-"
                    + str(args.batch_size)
                    + "_epp-"
                    + str(args.e_patience)
                    + "_"
                    + str(timestring)
                )
                run_exp(
                    name_saved=name_saved,
                    model_name=args.model,
                    processed_X_train=processed_X_train,
                    y_train=y_train,
                    processed_X_val=processed_X_val,
                    y_val=y_val,
                    processed_X_test=processed_X_test,
                    y_test=y_test,
                    class_names=class_names,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    e_patience=args.e_patience,
                    num_dense_layers=num_dense,
                )

        elif args.model in ["conv1d"]:
            max_num_maxpools = 5
            max_num_conv_layers_per_maxpool = 5
            # Iterate through the number of max-pooling layers
            for num_maxpools in range(1, max_num_maxpools + 1):
                # Iterate through the number of Conv1D layers per max-pooling layer
                for num_conv_layers_per_maxpool in range(
                    1, max_num_conv_layers_per_maxpool + 1
                ):
                    print("=" * 150)
                    print(
                        f"{num_maxpools} maxpool, {num_conv_layers_per_maxpool} conv per maxpool"
                    )
                    now = datetime.datetime.now()
                    timestring = now.strftime(
                        "%Y-%m-%d_%H-%M-%S"
                    )  # https://strftime.org/
                    name_saved = (
                        str(args.model)
                        + "_tsize-"
                        + str(args.test_size)
                        + "_ep-"
                        + str(args.epochs)
                        + "_bs-"
                        + str(args.batch_size)
                        + "_epp-"
                        + str(args.e_patience)
                        + "_"
                        + str(timestring)
                    )

                    run_exp(
                        name_saved=name_saved,
                        model_name=args.model,
                        processed_X_train=processed_X_train,
                        y_train=y_train,
                        processed_X_val=processed_X_val,
                        y_val=y_val,
                        processed_X_test=processed_X_test,
                        y_test=y_test,
                        class_names=class_names,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        e_patience=args.e_patience,
                        num_conv_layers_per_maxpool=num_conv_layers_per_maxpool,
                        num_maxpools=num_maxpools,
                    )

    else:
        (
            processed_X_train,
            y_train,
            processed_X_test,
            y_test,
            class_names,
        ) = load_svm_data(args.source)
        now = datetime.datetime.now()
        timestring = now.strftime("%Y-%m-%d_%H-%M-%S")
        name_saved = (
            str(args.model)
            + "_C-"
            + str(args.C)
            + "_kernel-"
            + str(args.kernel)
            + "_gamma-"
            + str(args.gamma)
            + "_d-"
            + str(args.d)
            + "_df-"
            + str(args.df)
            + "_fold-"
            + str(args.k_fold)
            + "_"
            + str(timestring)
        )

        run_svm_exp(
            name_saved=name_saved,
            processed_X_train=processed_X_train,
            y_train=y_train,
            processed_X_test=processed_X_test,
            y_test=y_test,
            class_names=class_names,
            C=args.C,
            kernel=args.kernel,
            gamma=args.gamma,
            d=args.d,
            df=args.df,
            k_fold=args.k_fold,
        )

    # Luu lai cac thong so nhap tu ban phim de chay thuc nghiem
    with open(f"log/{name_saved}_config.json", "w") as fp:
        json.dump(str(args), fp)
