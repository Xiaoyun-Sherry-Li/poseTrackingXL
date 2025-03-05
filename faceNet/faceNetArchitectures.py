import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import numpy as np

''' Sliced-channel Model architecture definitions '''
def s5(inputs_shape=(128,128,1), data_augmentation=None, base_filters=25):
    inputs = keras.Input(inputs_shape)
    # normalize inputs
    x = preprocessing.Rescaling(scale=1. / 127.5, offset=-1.0)(inputs)
    # augment images
    if data_augmentation is not None:
        x = data_augmentation(x)
    # big, strided conv and maxpool, 128->32
    x = layers.Conv2D(filters=base_filters, kernel_size=7, padding='same', strides=2,
                      activation='selu', kernel_initializer='lecun_normal')(x)
    x = layers.Conv2D(filters=base_filters*2, kernel_size=3, padding='same',
                      activation='selu', kernel_initializer='lecun_normal')(x)
    x = layers.MaxPool2D(2)(x)
    # Repeated fine-scale conv and maxpool
    # 32->16
    x = layers.Conv2D(filters=base_filters * 2, kernel_size=3, padding='same',
                      activation='selu', kernel_initializer='lecun_normal')(x)
    x = layers.Conv2D(filters=base_filters * 4, kernel_size=3, padding='same',
                      activation='selu', kernel_initializer='lecun_normal')(x)
    x = layers.MaxPool2D(2)(x)
    # 16->1
    x = layers.Conv2D(filters=base_filters * 4, kernel_size=3, padding='same',
                      activation='selu', kernel_initializer='lecun_normal')(x)
    x = layers.GlobalAveragePooling2D(name='view_features')(x)
    # dropout, prediction, and return model
    x = layers.AlphaDropout(0.2)(x)
    view_pred = layers.Dense(1, activation='sigmoid', name='view_pred')(x)
    return keras.Model(inputs, view_pred)


''' Multi-view weighted prediction model architectures '''
def j4(inputs_shape=(128,128,6), viewMdl=s5()):
    nViews = inputs_shape[-1]
    inputs = layers.Input(inputs_shape)
    allSlices = layers.Lambda(lambda z: tf.split(z, nViews, axis=-1))(inputs)

    # find feature and prediction layers in the single-view model
    feature_layer_idx = [i for i, x in enumerate(viewMdl.layers) if x.name=='view_features'][0]
    pred_layer_idx = [i for i, x in enumerate(viewMdl.layers) if x.name == 'view_pred'][0]

    # set 'trainable' to false for all layers before image features (optional?)
    # for l in viewMdl.layers[:feature_layer_num-1]:
    #     l.trainable = False

    # gather features and predictions to define new model w/ weighting output
    # optionally, use a hidden layer here between feature and weighting layer
    feature_layer = viewMdl.layers[feature_layer_idx]
    x = layers.AlphaDropout(0.2, name='feature_dropout')(feature_layer.output)
    x = layers.Dense(10, name='feature_hidden', activation='selu', kernel_initializer='lecun_normal')(x)
    view_weight = layers.Dense(1, name='view_weight')(x)
    pred_layer = viewMdl.layers[pred_layer_idx]
    featMdl = keras.Model(viewMdl.input, [pred_layer.output, view_weight])

    # Make predictions and average
    allPreds = [featMdl(view) for view in allSlices]
    avg_pred = layers.Average(name='avg_pred')([x[0] for x in allPreds])

    # Take 'weights'/scores and average, using preset and non-tunable weights
    view_weights = layers.Concatenate(name='view_weights')([x[1] for x in allPreds])
    joint_pred = layers.Dense(1, activation='sigmoid', name='joint_pred')
    joint_pred_out = joint_pred(view_weights)
    joint_pred.set_weights([np.ones((nViews, 1)), np.zeros((1))])
    joint_pred.trainable = False

    return keras.Model(inputs, [joint_pred_out, avg_pred])


''' Multi-Channel Model architecture definitions '''
def v1(inputs_shape, data_augmentation):
    inputs = keras.Input(inputs_shape)
    # normalize inputs
    x = preprocessing.Rescaling(scale=1. / 127.5, offset=-1.0)(inputs)
    if data_augmentation is not None:
        # augment images
        x = data_augmentation(x)

    # Process each channel independently: Depthwise Convolution, MaxPool2D
    # if using filter size 7, stride 2, 32x32 output
    x = layers.DepthwiseConv2D(7, strides=2, padding='same', depth_multiplier=10,
                               activation='selu', depthwise_initializer='lecun_normal')(x)
    x = layers.MaxPool2D(2)(x)
    # repeat depthwise conv using small filters and stride 1
    x = layers.DepthwiseConv2D(3, strides=1, padding='same', depth_multiplier=2,
                               activation='selu', depthwise_initializer='lecun_normal')(x)
    x = layers.MaxPool2D(2)(x)
    # Compress channels using 1x1 convolution
    x = layers.Conv2D(filters=50, kernel_size=1, padding='same', activation='selu', kernel_initializer='lecun_normal')(x)

    # Combine across channels: Conv2D, MaxPool2D
    # use filter size 3, stride 1, and repeat twice times to go from 16x16 to 4x4
    x = layers.Conv2D(filters=25, kernel_size=3, padding='same', activation='selu', kernel_initializer='lecun_normal')(x)
    x = layers.MaxPool2D(2)(x)
    x = layers.Conv2D(filters=25, kernel_size=3, padding='same', activation='selu', kernel_initializer='lecun_normal')(x)
    x = layers.MaxPool2D(2)(x)

    # Fully connected layer to combine, then classification head
    x = layers.Flatten()(x)
    x = layers.Dense(25, activation='selu', kernel_initializer='lecun_normal')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    # define resulting model
    return keras.Model(inputs, outputs)

def v2(inputs_shape, data_augmentation):
    inputs = keras.Input(inputs_shape)
    # normalize inputs
    x = preprocessing.Rescaling(scale=1. / 127.5, offset=-1.0)(inputs)
    if data_augmentation is not None:
        # augment images
        x = data_augmentation(x)

    # Process each channel independently and downsample with strided Depthwise Convolutions
    # 128->64
    x = layers.DepthwiseConv2D(5, strides=2, padding='same', depth_multiplier=10)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    # 64->32
    x = layers.DepthwiseConv2D(5, strides=2, padding='same', depth_multiplier=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    # 32->16
    x = layers.DepthwiseConv2D(5, strides=2, padding='same', depth_multiplier=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Compress channels using 1x1 convolution
    x = layers.Conv2D(filters=50, kernel_size=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Combine across channels and downsample with Conv2D
    #16->8
    x = layers.Conv2D(filters=25, kernel_size=3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    # 8->4
    x = layers.Conv2D(filters=25, kernel_size=3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Global Average Pooling to reduce spatial dimension
    x = layers.GlobalAveragePooling2D()(x)
    # Classification head directly from global average
    outputs = layers.Dense(1, activation='sigmoid')(x)
    # define resulting model
    return keras.Model(inputs, outputs)

def v3(inputs_shape, data_augmentation):
    inputs = keras.Input(inputs_shape)
    # normalize inputs
    x = preprocessing.Rescaling(scale=1. / 127.5, offset=-1.0)(inputs)
    if data_augmentation is not None:
        # augment images
        x = data_augmentation(x)

    # Process each channel independently and downsample
    # 128->32
    x = layers.DepthwiseConv2D(7, strides=2, padding='same', depth_multiplier=10)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(2)(x)

    # combine across channels
    x = layers.Conv2D(filters=100, kernel_size=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Combine across channels and downsample with Conv2D
    #32->16
    x = layers.Conv2D(filters=25, kernel_size=3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    # 16->8
    x = layers.Conv2D(filters=25, kernel_size=3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    # 8->4
    x = layers.Conv2D(filters=25, kernel_size=3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Global Average Pooling to reduce spatial dimension
    x = layers.GlobalAveragePooling2D()(x)
    # Classification head directly from global average
    outputs = layers.Dense(1, activation='sigmoid')(x)
    # define resulting model
    return keras.Model(inputs, outputs)

def v4(inputs_shape, data_augmentation):
    inputs = keras.Input(inputs_shape)
    # normalize inputs
    x = preprocessing.Rescaling(scale=1. / 127.5, offset=-1.0)(inputs)
    if data_augmentation is not None:
        # augment images
        x = data_augmentation(x)

    # Process each channel independently and downsample with strided Depthwise Convolutions
    # 128->64
    x = layers.DepthwiseConv2D(5, strides=2, padding='same', depth_multiplier=10)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    # 64->32
    x = layers.DepthwiseConv2D(5, strides=2, padding='same', depth_multiplier=5)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    # 32->16
    x = layers.DepthwiseConv2D(5, strides=2, padding='same', depth_multiplier=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Compress channels using 1x1 convolution
    x = layers.Conv2D(filters=100, kernel_size=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Combine across channels and downsample with Conv2D
    #16->8
    x = layers.Conv2D(filters=40, kernel_size=3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    # 8->4
    x = layers.Conv2D(filters=40, kernel_size=3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Global Average Pooling to reduce spatial dimension
    x = layers.GlobalAveragePooling2D()(x)
    # Classification head directly from global average
    outputs = layers.Dense(1, activation='sigmoid')(x)
    # define resulting model
    return keras.Model(inputs, outputs)

def v5(inputs_shape, data_augmentation):
    inputs = keras.Input(inputs_shape)
    # normalize inputs
    x = preprocessing.Rescaling(scale=1. / 127.5, offset=-1.0)(inputs)
    if data_augmentation is not None:
        # augment images
        x = data_augmentation(x)

    # Process each channel independently and downsample with strided Depthwise Convolutions
    # 128->64
    x = layers.DepthwiseConv2D(5, strides=2, padding='same', depth_multiplier=20)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    # 64->32
    x = layers.DepthwiseConv2D(5, strides=2, padding='same', depth_multiplier=10)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    # 32->16
    x = layers.DepthwiseConv2D(5, strides=2, padding='same', depth_multiplier=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Compress channels using 1x1 convolution
    x = layers.Conv2D(filters=120, kernel_size=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Combine across channels and downsample with Conv2D
    #16->8
    x = layers.Conv2D(filters=30, kernel_size=3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    # 8->4
    x = layers.Conv2D(filters=30, kernel_size=3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Global Average Pooling to reduce spatial dimension
    x = layers.GlobalAveragePooling2D()(x)
    # Classification head directly from global average
    outputs = layers.Dense(1, activation='sigmoid')(x)
    # define resulting model
    return keras.Model(inputs, outputs)

def v6(inputs_shape, data_augmentation):
    inputs = keras.Input(inputs_shape)
    # normalize inputs
    x = preprocessing.Rescaling(scale=1. / 127.5, offset=-1.0)(inputs)
    if data_augmentation is not None:
        # augment images
        x = data_augmentation(x)

    # Process each channel independently and downsample with strided Depthwise Convolutions
    # 128->64
    x = layers.DepthwiseConv2D(5, strides=2, padding='same', depth_multiplier=10)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    # 64->32
    x = layers.DepthwiseConv2D(5, strides=2, padding='same', depth_multiplier=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    # 32->16
    x = layers.DepthwiseConv2D(5, strides=2, padding='same', depth_multiplier=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Compress channels using 1x1 convolution
    x = layers.Conv2D(filters=120, kernel_size=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Combine across channels and downsample with Conv2D
    #16->8
    x = layers.Conv2D(filters=60, kernel_size=3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    # 8->4
    x = layers.Conv2D(filters=60, kernel_size=3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Global Average Pooling to reduce spatial dimension
    x = layers.GlobalAveragePooling2D()(x)
    # Classification head directly from global average
    outputs = layers.Dense(1, activation='sigmoid')(x)
    # define resulting model
    return keras.Model(inputs, outputs)

def v7(inputs_shape, data_augmentation):
    inputs = keras.Input(inputs_shape)
    # normalize inputs
    x = preprocessing.Rescaling(scale=1. / 127.5, offset=-1.0)(inputs)
    if data_augmentation is not None:
        # augment images
        x = data_augmentation(x)

    # Process each channel independently and downsample with strided Depthwise Convolutions
    # 128->64
    x = layers.DepthwiseConv2D(5, strides=2, padding='same', depth_multiplier=10)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    # 64->32
    x = layers.DepthwiseConv2D(5, strides=2, padding='same', depth_multiplier=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    # 32->16
    x = layers.DepthwiseConv2D(5, strides=2, padding='same', depth_multiplier=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.2)(x)

    # Compress channels using 1x1 convolution
    x = layers.Conv2D(filters=100, kernel_size=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.2)(x)

    # Combine across channels and downsample with Conv2D
    #16->8
    x = layers.Conv2D(filters=50, kernel_size=3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.2)(x)
    # 8->4
    x = layers.Conv2D(filters=50, kernel_size=3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Global Average Pooling to reduce spatial dimension
    x = layers.GlobalAveragePooling2D()(x)
    # Classification head directly from global average
    outputs = layers.Dense(1, activation='sigmoid')(x)
    # define resulting model
    return keras.Model(inputs, outputs)

def v8(inputs_shape, data_augmentation):
    inputs = keras.Input(inputs_shape)
    # normalize inputs
    x = preprocessing.Rescaling(scale=1. / 127.5, offset=-1.0)(inputs)
    if data_augmentation is not None:
        # augment images
        x = data_augmentation(x)

    # Process each channel independently: Depthwise Convolution, MaxPool2D
    # if using filter size 7, stride 2, 128->32 output
    x = layers.DepthwiseConv2D(7, strides=2, padding='same', depth_multiplier=10,
                               activation='selu', depthwise_initializer='lecun_normal')(x)
    x = layers.MaxPool2D(2)(x)
    # Combine across channels: Conv2D, MaxPool2D
    # use filter size 3, stride 1 (32->16)
    x = layers.Conv2D(filters=100, kernel_size=3, padding='same', activation='selu', kernel_initializer='lecun_normal')(x)
    x = layers.MaxPool2D(2)(x)
    x = layers.Dropout(0.2)(x)
    # 16->8
    x = layers.Conv2D(filters=100, kernel_size=3, padding='same', activation='selu', kernel_initializer='lecun_normal')(x)
    x = layers.MaxPool2D(2)(x)
    x = layers.Dropout(0.2)(x)
    # 8->4
    x = layers.Conv2D(filters=100, kernel_size=3, padding='same', activation='selu', kernel_initializer='lecun_normal')(
        x)
    x = layers.MaxPool2D(2)(x)
    x = layers.Dropout(0.2)(x)

    # Fully connected layer to combine, then classification head
    x = layers.Flatten()(x)
    x = layers.Dense(25, activation='selu', kernel_initializer='lecun_normal')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    # define resulting model
    return keras.Model(inputs, outputs)

def v9(inputs_shape, data_augmentation):
    inputs = keras.Input(inputs_shape)
    # normalize inputs
    x = preprocessing.Rescaling(scale=1. / 127.5, offset=-1.0)(inputs)
    if data_augmentation is not None:
        # augment images
        x = data_augmentation(x)

    # Process each channel independently: Depthwise Convolution, MaxPool2D
    # if using filter size 7, stride 2, 128->32 output
    x = layers.DepthwiseConv2D(7, strides=2, padding='same', depth_multiplier=10,
                               activation='selu', depthwise_initializer='lecun_normal')(x)
    x = layers.Conv2D(filters=60, kernel_size=1, padding='same', activation='selu', kernel_initializer='lecun_normal')(x)
    y = layers.GlobalAveragePooling2D()(x)

    x = layers.MaxPool2D(2)(x)
    # use filter size 3, stride 1 (32->16)
    x = layers.DepthwiseConv2D(3, padding='same', depth_multiplier=2,
                               activation='selu', depthwise_initializer='lecun_normal')(x)
    x = layers.Conv2D(filters=100, kernel_size=1, padding='same', activation='selu', kernel_initializer='lecun_normal')(x)
    x = layers.MaxPool2D(2)(x)
    x = layers.Dropout(0.2)(x)
    # 16->8
    x = layers.DepthwiseConv2D(3, padding='same', depth_multiplier=2,
                               activation='selu', depthwise_initializer='lecun_normal')(x)
    x = layers.Conv2D(filters=100, kernel_size=1, padding='same', activation='selu', kernel_initializer='lecun_normal')(x)
    x = layers.MaxPool2D(2)(x)
    x = layers.Dropout(0.2)(x)
    # 8->4
    x = layers.DepthwiseConv2D(3, padding='same', depth_multiplier=2,
                               activation='selu', depthwise_initializer='lecun_normal')(x)
    x = layers.Conv2D(filters=100, kernel_size=1, padding='same', activation='selu', kernel_initializer='lecun_normal')(x)
    x = layers.MaxPool2D(2)(x)
    x = layers.Dropout(0.2)(x)

    # Fully connected layer to combine
    x = layers.Flatten()(x)
    x = layers.Dense(25, activation='selu', kernel_initializer='lecun_normal')(x)
    # Same for intermediate supervision output
    y = layers.Dense(25, activation='selu', kernel_initializer='lecun_normal')(y)
    # Combine both for output
    outputs = [layers.Dense(1, activation='sigmoid', name='model_output')(x),
               layers.Dense(1, activation='sigmoid', name='int_output')(y)]

    # define resulting model
    return keras.Model(inputs, outputs)

