import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import numpy as np

#%% Sliced-channel Model architecture definitions

def s1(inputs_shape, data_augmentation, base_filters=25):
    nViews = inputs_shape[2]
    inputs = keras.Input(inputs_shape)
    # normalize inputs jointly
    x = preprocessing.Rescaling(scale=1. / 127.5, offset=-1.0)(inputs)
    # slice image by view/channel
    allSlices = layers.Lambda(lambda z: tf.split(z, nViews, axis=-1))(inputs)

    # define view-encoding network
    view_in = layers.Input(allSlices[0].shape[1:])
    # augment images for each channel independently
    if data_augmentation is not None:
        x = data_augmentation(view_in)
    else:
        x = view_in
    # big, strided conv and maxpool, 128->32
    x = layers.Conv2D(filters=base_filters, kernel_size=7, padding='same', strides=2,
                      activation='selu', kernel_initializer='lecun_normal')(x)
    x = layers.MaxPool2D(2)(x)
    # Repeated fine-scale conv and maxpool
    # 32->16
    x = layers.Conv2D(filters=base_filters * 2, kernel_size=3, padding='same',
                      activation='selu', kernel_initializer='lecun_normal')(x)
    x = layers.MaxPool2D(2)(x)
    # 16->8
    x = layers.Conv2D(filters=base_filters * 4, kernel_size=3, padding='same',
                      activation='selu', kernel_initializer='lecun_normal')(x)
    x = layers.MaxPool2D(2)(x)
    # 8->1
    x = layers.Conv2D(filters=base_filters * 8, kernel_size=3, padding='same',
                      activation='selu', kernel_initializer='lecun_normal')(x)
    x = layers.GlobalAveragePooling2D()(x)
    # Apply dropout on both sides of a hidden layer
    x = layers.AlphaDropout(0.25)(x)
    view_features = layers.Dense(base_filters*2, name = 'view_features', activation='selu', kernel_initializer='lecun_normal')(x)
    view_features = layers.AlphaDropout(0.25)(view_features)
    # Define predictions and view-fusion weights from features/hidden layer
    view_pred = layers.Dense(1, activation='sigmoid', name='view_pred')(view_features)
    view_weight = layers.Dense(1, name='view_weight')(view_features)
    enc = keras.Model(view_in,[view_pred, view_weight])

    # Make predictions on each image and gather results
    allPreds = [enc(x) for x in allSlices]
    avg_pred = layers.Average(name='avg_pred')([x[0] for x in allPreds])
    # Take weighted combination of view preds to make joint prediction
    view_preds = layers.Concatenate(name='view_preds')([x[0] for x in allPreds])
    view_weights = layers.Concatenate(name='view_weights')([x[1] for x in allPreds])
    view_weights = layers.Softmax()(view_weights)
    joint_pred = layers.Dot(axes=1, name='joint_pred')([view_preds, view_weights])

    # Define full model
    return keras.Model(inputs, [joint_pred, avg_pred])

def s2(inputs_shape, data_augmentation, base_filters=16):
    nViews = inputs_shape[2]
    inputs = keras.Input(inputs_shape)
    # normalize inputs jointly
    x = preprocessing.Rescaling(scale=1. / 127.5, offset=-1.0)(inputs)
    # slice image by view/channel
    # allSlices = [layers.Lambda(lambda z: z[...,nView:nView+1])(x) for nView in range(nViews)]
    allSlices = []
    for nView in range(nViews):
        allSlices.append(layers.Lambda(lambda z: z[...,nView:nView+1])(x))

    # define view-encoding network
    view_in = layers.Input(allSlices[0].shape[1:])
    # augment images for each channel independently
    if data_augmentation is not None:
        x = data_augmentation(view_in)
    else:
        x = view_in
    # big, strided conv and maxpool, 128->32
    x = layers.Conv2D(filters=base_filters, kernel_size=7, padding='same', strides=2,
                      activation='selu', kernel_initializer='lecun_normal')(x)
    x = layers.MaxPool2D(2)(x)
    # Repeated fine-scale conv and maxpool
    # 32->16
    x = layers.Conv2D(filters=base_filters * 2, kernel_size=3, padding='same',
                      activation='selu', kernel_initializer='lecun_normal')(x)
    x = layers.MaxPool2D(2)(x)
    # 16->8
    x = layers.Conv2D(filters=base_filters * 4, kernel_size=3, padding='same',
                      activation='selu', kernel_initializer='lecun_normal')(x)
    x = layers.MaxPool2D(2)(x)
    # 8->1
    x = layers.Conv2D(filters=base_filters * 8, kernel_size=3, padding='same',
                      activation='selu', kernel_initializer='lecun_normal')(x)
    view_features = layers.GlobalAveragePooling2D()(x)
    view_pred = layers.Dense(1, name='view_pred')(view_features)
    # define encoding model
    enc = keras.Model(view_in,view_pred)

    # Combine predictions from each view
    allPreds = layers.Concatenate()([enc(x) for x in allSlices])
    joint_pred = layers.Dense(1, activation='sigmoid', name='joint_pred')(allPreds)

    # Define full model
    return keras.Model(inputs, joint_pred)

def s3(inputs_shape, data_augmentation, base_filters=25):
    inputs = keras.Input(inputs_shape)
    # normalize inputs
    x = preprocessing.Rescaling(scale=1. / 127.5, offset=-1.0)(inputs)
    # augment images
    if data_augmentation is not None:
        x = data_augmentation(x)
    # big, strided conv and maxpool, 128->32
    x = layers.Conv2D(filters=base_filters, kernel_size=7, padding='same', strides=2,
                      activation='selu', kernel_initializer='lecun_normal')(x)
    x = layers.MaxPool2D(2)(x)
    # Repeated fine-scale conv and maxpool
    # 32->16
    x = layers.Conv2D(filters=base_filters * 2, kernel_size=3, padding='same',
                      activation='selu', kernel_initializer='lecun_normal')(x)
    x = layers.MaxPool2D(2)(x)
    # 16->8
    x = layers.Conv2D(filters=base_filters * 4, kernel_size=3, padding='same',
                      activation='selu', kernel_initializer='lecun_normal')(x)
    x = layers.MaxPool2D(2)(x)
    # 8->4
    x = layers.Conv2D(filters=base_filters * 8, kernel_size=3, padding='same',
                      activation='selu', kernel_initializer='lecun_normal')(x)
    x = layers.MaxPool2D(2)(x)
    # flatten, featurize and dropout, then predict
    x = layers.Flatten()(x)
    x = layers.AlphaDropout(0.25)(x)
    view_features = layers.Dense(base_filters*2, name = 'view_features', activation='selu', kernel_initializer='lecun_normal')(x)
    view_features = layers.AlphaDropout(0.25)(view_features)
    view_pred = layers.Dense(1, activation='sigmoid', name='view_pred')(view_features)
    # return model
    return keras.Model(inputs, view_pred)

def s4(inputs_shape=(128,128,1), data_augmentation=None, base_filters=25):
    inputs = keras.Input(inputs_shape)
    # normalize inputs
    x = preprocessing.Rescaling(scale=1. / 127.5, offset=-1.0)(inputs)
    # augment images
    if data_augmentation is not None:
        x = data_augmentation(x)
    # big, strided conv and maxpool, 128->32
    x = layers.Conv2D(filters=base_filters, kernel_size=7, padding='same', strides=2,
                      activation='selu', kernel_initializer='lecun_normal')(x)
    x = layers.MaxPool2D(2)(x)
    # Repeated fine-scale conv and maxpool
    # 32->16
    x = layers.Conv2D(filters=base_filters * 2, kernel_size=3, padding='same',
                      activation='selu', kernel_initializer='lecun_normal')(x)
    x = layers.MaxPool2D(2)(x)
    # 16->8
    x = layers.Conv2D(filters=base_filters * 4, kernel_size=3, padding='same',
                      activation='selu', kernel_initializer='lecun_normal')(x)
    x = layers.MaxPool2D(2)(x)
    # 8->1
    x = layers.Conv2D(filters=base_filters * 8, kernel_size=3, padding='same',
                      activation='selu', kernel_initializer='lecun_normal')(x)
    x = layers.GlobalAveragePooling2D(name='view_features')(x)
    # dropout, prediction, and return model
    x = layers.AlphaDropout(0.2)(x)
    view_pred = layers.Dense(1, activation='sigmoid', name='view_pred')(x)
    return keras.Model(inputs, view_pred)

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

#%% Multi-view weighted prediction model architectures

def j1(inputs_shape, viewMdl):
    nViews = inputs_shape[-1]
    inputs = layers.Input(inputs_shape)
    allSlices = layers.Lambda(lambda z: tf.split(z, nViews, axis=-1))(inputs)

    # find feature and prediction layers in the single-view model
    feature_layer_num = [i for i,x in enumerate(viewMdl.layers) if x.name=='view_features'][0]
    pred_layer_num = [i for i, x in enumerate(viewMdl.layers) if x.name == 'view_pred'][0]

    # set 'trainable' to false for all layers (optional, train final layer?)
    for l in viewMdl.layers[:feature_layer_num]:
        l.trainable = False

    # gather features and predictions to define new model w/ weighting output
    # optionally, use a hidden layer here between feature and weighting layer
    feature_layer = viewMdl.layers[feature_layer_num]
    x = layers.AlphaDropout(0.2, name='feature_dropout')(feature_layer.output)
    x = layers.Dense(10, name='feature_hidden', activation='selu', kernel_initializer='lecun_normal')(x)
    view_weight = layers.Dense(1, name='view_weight')(x)
    pred_layer = viewMdl.layers[pred_layer_num]
    featMdl = keras.Model(viewMdl.input, [pred_layer.output, view_weight])

    # Make predictions and average
    allPreds = [featMdl(view) for view in allSlices]
    avg_pred = layers.Average(name='avg_pred')([x[0] for x in allPreds])

    # Take weighted combination of view preds to make joint prediction
    view_preds = layers.Concatenate(name='view_preds')([x[0] for x in allPreds])
    view_weights = layers.Concatenate(name='view_weights')([x[1] for x in allPreds])
    view_weights = layers.Softmax()(view_weights)
    joint_pred = layers.Dot(axes=1, name='joint_pred')([view_preds, view_weights])

    return keras.Model(inputs, [joint_pred, avg_pred])

def j2(inputs_shape, viewMdl):
    nViews = inputs_shape[-1]
    inputs = layers.Input(inputs_shape)
    allSlices = layers.Lambda(lambda z: tf.split(z, nViews, axis=-1))(inputs)

    # find feature and prediction layers in the single-view model
    feature_layer_num = [i for i,x in enumerate(viewMdl.layers) if x.name=='view_features'][0]
    pred_layer_num = [i for i, x in enumerate(viewMdl.layers) if x.name == 'view_pred'][0]

    # lock weights for layers before feature layer
    for l in viewMdl.layers[:feature_layer_num]:
        l.trainable = False

    # gather features and pred from pretrained model, add hidden layer and new weighting outputs
    feature_layer = viewMdl.layers[feature_layer_num]
    x = layers.AlphaDropout(0.2, name='feature_dropout')(feature_layer.output)
    x = layers.Dense(10, name='feature_hidden', activation='selu', kernel_initializer='lecun_normal')(x)
    view_weight = layers.Dense(1, name='view_weight')(x)
    pred_layer = viewMdl.layers[pred_layer_num]
    featMdl = keras.Model(viewMdl.input, [pred_layer.output, view_weight])

    # Make predictions
    allPreds = [featMdl(view) for view in allSlices]
    # Take weighted combination of view preds to make joint prediction
    view_preds = layers.Concatenate(name='view_preds')([x[0] for x in allPreds])
    view_weights = layers.Concatenate(name='view_weights')([x[1] for x in allPreds])
    view_weights = layers.Softmax()(view_weights)
    joint_pred = layers.Dot(axes=1, name='joint_pred')([view_preds, view_weights])

    return keras.Model(inputs, joint_pred)

# like j1 but weighted average instead of softmax
def j3(inputs_shape, viewMdl):
    nViews = inputs_shape[-1]
    inputs = layers.Input(inputs_shape)
    allSlices = layers.Lambda(lambda z: tf.split(z, nViews, axis=-1))(inputs)

    # find feature and prediction layers in the single-view model
    feature_layer_num = [i for i,x in enumerate(viewMdl.layers) if x.name=='view_features'][0]
    pred_layer_num = [i for i, x in enumerate(viewMdl.layers) if x.name == 'view_pred'][0]

    # set 'trainable' to false for all layers (optional, train final layer?)
    for l in viewMdl.layers[:feature_layer_num]:
        l.trainable = False

    # gather features and predictions to define new model w/ weighting output
    # optionally, use a hidden layer here between feature and weighting layer
    feature_layer = viewMdl.layers[feature_layer_num]
    x = layers.AlphaDropout(0.2, name='feature_dropout')(feature_layer.output)
    x = layers.Dense(10, name='feature_hidden', activation='selu', kernel_initializer='lecun_normal')(x)
    view_weight = layers.Dense(1, activation='sigmoid', name='view_weight')(x)
    pred_layer = viewMdl.layers[pred_layer_num]
    featMdl = keras.Model(viewMdl.input, [pred_layer.output, view_weight])

    # Make predictions and average
    allPreds = [featMdl(view) for view in allSlices]
    avg_pred = layers.Average(name='avg_pred')([x[0] for x in allPreds])

    # Take weighted combination of view preds to make joint prediction
    view_preds = layers.Concatenate(name='view_preds')([x[0] for x in allPreds])
    view_weights = layers.Concatenate(name='view_weights')([x[1] for x in allPreds])
    view_weights = layers.Lambda(lambda z: tf.math.divide(z, tf.math.reduce_sum(z, axis=-1, keepdims=True)),
                                 name='norm_view_weights')(view_weights)
    joint_pred = layers.Dot(axes=1, name='joint_pred')([view_preds, view_weights])

    return keras.Model(inputs, [joint_pred, avg_pred])

# like j1 but jointPred directly from weights
def j4(inputs_shape=(128,128,6), viewMdl=s5()):
    nViews = inputs_shape[-1]
    inputs = layers.Input(inputs_shape)
    allSlices = layers.Lambda(lambda z: tf.split(z, nViews, axis=-1))(inputs)

    # find feature and prediction layers in the single-view model
    feature_layer_num = [i for i,x in enumerate(viewMdl.layers) if x.name=='view_features'][0]
    pred_layer_num = [i for i, x in enumerate(viewMdl.layers) if x.name == 'view_pred'][0]

    # set 'trainable' to false for all layers before image features (optional?)
    # for l in viewMdl.layers[:feature_layer_num-1]:
    #     l.trainable = False

    # gather features and predictions to define new model w/ weighting output
    # optionally, use a hidden layer here between feature and weighting layer
    feature_layer = viewMdl.layers[feature_layer_num]
    x = layers.AlphaDropout(0.2, name='feature_dropout')(feature_layer.output)
    x = layers.Dense(10, name='feature_hidden', activation='selu', kernel_initializer='lecun_normal')(x)
    view_weight = layers.Dense(1, name='view_weight')(x)
    pred_layer = viewMdl.layers[pred_layer_num]
    featMdl = keras.Model(viewMdl.input, [pred_layer.output, view_weight])

    # Make predictions and average
    allPreds = [featMdl(view) for view in allSlices]
    avg_pred = layers.Average(name='avg_pred')([x[0] for x in allPreds])

    # Take 'weights'/scores and average, using preset and non-tunable weights
    view_weights = layers.Concatenate(name='view_weights')([x[1] for x in allPreds])
    joint_pred = layers.Dense(1, activation='sigmoid', name='joint_pred')
    joint_pred_out = joint_pred(view_weights)
    joint_pred.set_weights([np.ones((6,1)), np.zeros((1))])
    joint_pred.trainable = False

    return keras.Model(inputs, [joint_pred_out, avg_pred])

#%% Multi-Channel Model architecture defitions

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

