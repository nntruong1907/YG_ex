from tensorflow import keras
from keras.models import Model
from keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Dense,
    Dropout,
    Flatten,
)


def fcnn2d_model(
    input_shape,
    num_classes=10,
    num_dense_layers=5,
    dropout_fc=0.2,
    optimizers_func="adam",
    learning_rate=0.0005,
    loss_func="categorical_crossentropy",
):
    """
    Tạo mô hình Fully Connected Neural Network (FCNN)2D tùy chỉnh.
    """

    num_units_list = [64, 128, 128, 256, 265, 128, 128, 64]
    if num_dense_layers < len(num_units_list):
        dense_units = num_units_list[:num_dense_layers]
    else:
        # Nếu num_dense_layers lớn hơn số lớp đơn vị được định nghĩa sẵn, thì sẽ sử dụng lặp lại lớp đơn vị cuối cùng.
        num_repeats = num_dense_layers - len(num_units_list)
        dense_units = num_units_list + [num_units_list[-1]] * num_repeats

    input_layer = Input(shape=input_shape)
    x = input_layer
    x = Flatten()(x)

    for units in dense_units:
        x = Dense(units, activation="relu")(x)
        if dropout_fc > 0:
            x = Dropout(dropout_fc)(x)

    output_layer = Dense(num_classes, activation="softmax")(x)

    # Tạo mô hình từ các lớp đã định nghĩa
    model = Model(
        inputs=input_layer, outputs=output_layer, name=f"fcnn2d_{num_dense_layers}_dense"
    )

    if optimizers_func == "adam":
        optimizers_func = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizers_func == "sgd":
        optimizers_func = keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizers_func == "rmsprop":
        optimizers_func = keras.optimizers.RMSprop(learning_rate=learning_rate)

    model.compile(loss=loss_func, optimizer=optimizers_func, metrics=["accuracy"])

    return model


def conv2d_model(
    input_shape,
    num_classes=10,
    num_maxpools=1,
    num_conv_layers_per_maxpool=2,
    filter_size=3,
    pool_size=2,
    dropout_cnn=0.2,
    dropout_f=0.5,
    optimizers_func="adam",
    learning_rate=0.0005,
    loss_function="categorical_crossentropy",
):
    """
    Tạo mô hình Convolution 2D tùy chỉnh.

    """
    num_filters_list = [32, 64, 128, 256, 128, 64]
    if num_conv_layers_per_maxpool < len(num_filters_list):
        num_filters = num_filters_list[:num_conv_layers_per_maxpool]
    else:
        num_repeats = num_conv_layers_per_maxpool - len(num_filters_list)
        num_filters = num_filters_list + [num_filters_list[-1]] * num_repeats

    inputs = Input(shape=input_shape)
    x = inputs

    for _ in range(num_maxpools):
        for filters in num_filters:
            x = Conv2D(
                filters, (filter_size, filter_size), activation="relu", padding="same"
            )(x)
            if dropout_cnn > 0:
                x = Dropout(dropout_cnn)(x)
        x = MaxPooling2D((pool_size, pool_size))(x)

    x = Flatten()(x)
    if dropout_f > 0:
        x = Dropout(dropout_f)(x)

    x = Dense(64, activation="relu")(x)
    if dropout_cnn > 0:
        x = Dropout(dropout_cnn)(x)

    outputs = Dense(num_classes, activation="softmax")(x)

    model = keras.Model(
        inputs,
        outputs,
        name=f"conv2d-{num_maxpools}maxpool_{num_conv_layers_per_maxpool*num_maxpools}conv",
    )

    if optimizers_func == "adam":
        optimizers_func = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizers_func == "sgd":
        optimizers_func = keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizers_func == "rmsprop":
        optimizers_func = keras.optimizers.RMSprop(learning_rate=learning_rate)

    model.compile(loss=loss_function, optimizer=optimizers_func, metrics=["accuracy"])

    return model
