#%% Imports

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from deepposekit.io import TrainingGenerator, DataGenerator
from deepposekit.augment import FlipAxis
import imgaug.augmenters as iaa
import imgaug as ia
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from deepposekit.callbacks import Logger, ModelCheckpoint

skeleton = './comSkeleton.csv'
dataGenPath = './comNet_Dataset.h5'
modelPath = './comNet_Model_05.h5'
logPath = './comNet_Log.h5'
batch_size = 8

gpu = 0
gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[gpu],'GPU')
tf.config.experimental.set_memory_growth(gpus[gpu], True)
#%% Define Augmentations

data_generator = DataGenerator(dataGenPath, mode='full')
augmenter = []
augmenter.append(FlipAxis(data_generator, axis=1)) # flip image left-right with 50% prob
#augmenter.append(iaa.flip.Fliplr(p=0.5))
augmenter.append(iaa.Sometimes(0.8, iaa.Affine(scale=(0.6, 1.4),
                            order=ia.ALL,
                            cval=ia.ALL,
                            mode='symmetric')))
augmenter.append(iaa.Sometimes(0.8, iaa.Affine(translate_percent={'x': (-0.2, 0.2), 'y': (-0.2, 0.2)},
                            order=ia.ALL,
                            cval=ia.ALL,
                            mode='symmetric')))
augmenter.append(iaa.Sometimes(0.25, iaa.Rot90(k=2)))
augmenter.append(iaa.Sometimes(0.5, iaa.LinearContrast((0.75, 1.25))))
augmenter.append(iaa.Sometimes(0.5, iaa.Add((-20, 20))))
augmenter.append(iaa.Sometimes(0.1, iaa.GaussianBlur(sigma = (1.0, 3.0))))
augmenter = iaa.Sequential(augmenter)

#%% Plot augmented image examples

image, keypoints = data_generator[0]
image, keypoints = augmenter(images=image, keypoints=keypoints)
keypoints[keypoints<0] = np.NaN
plt.figure(figsize=(5,5))
image = image[0] if image.shape[-1] is 3 else image[0, ..., 0]
cmap = None if image.shape[-1] is 3 else 'gray'
plt.imshow(image, cmap=cmap, interpolation='none')
for idx, jdx in enumerate(data_generator.graph):
    if jdx > -1:
        plt.plot(
            [keypoints[0, idx, 0], keypoints[0, jdx, 0]],
            [keypoints[0, idx, 1], keypoints[0, jdx, 1]],
            'r-'
        )
plt.scatter(keypoints[0, :, 0], keypoints[0, :, 1], c=np.arange(data_generator.keypoints_shape[0]), s=15, cmap=plt.cm.hsv, zorder=3)

plt.show()

#%% Set up training data pipeline and options

train_generator = TrainingGenerator(generator=data_generator,
                                    downsample_factor=2,
                                    augmenter=augmenter,
                                    validation_split=0.1,
                                    sigma=5,
                                    use_graph=True)
#train_generator.get_config()

#%% Set up model

from deepposekit.models import StackedDenseNet

# model = StackedDenseNet(train_generator, pretrained=False, n_stacks=1, growth_rate=48, n_transitions=-1)
model = StackedDenseNet(train_generator, n_stacks=2, growth_rate=40, n_transitions=-1)
#model.get_config()

#%% Set up callbacks
#should probably monitor val_output_1_loss instead of val_loss for choosing best model!
logger = Logger(validation_batch_size=batch_size, filepath=logPath)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, verbose=1, patience=20)
model_checkpoint = ModelCheckpoint(
    modelPath,
    monitor="val_loss",
    verbose=1,
    save_best_only=True,
)
early_stop = EarlyStopping(
    monitor="val_loss",
    min_delta=0.001,
    patience=42,
    verbose=1
)

callbacks = [logger, model_checkpoint, reduce_lr, early_stop]

#%% Fit the Model
loss_weights = [0.1,0.3,0.6]
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.MeanSquaredError(), loss_weights=loss_weights)
model.fit(
    batch_size=batch_size,
    validation_batch_size=batch_size,
    callbacks=callbacks,
    epochs=999,
    n_workers=8,
    steps_per_epoch=None,
    use_multiprocessing=False
)

#%%
from deepposekit.io import ImageGenerator

image_generator = ImageGenerator(data_generator)
predictions = model.predict(image_generator, verbose=1)

allData = data_generator.get_keypoints([data_generator.index])

#%% Plot Training Results
import h5py
f = h5py.File(logPath,'r')
plt.plot(f['logs']['loss'][25:]),
plt.plot(f['logs']['val_loss'][25:]),
plt.show()

euc_err = np.percentile(f['logs']['euclidean'],[5,25,50,75,95],axis=1)
plt.plot(euc_err[:,25:,0].T),plt.title('Head'),plt.show(),
plt.plot(euc_err[:,25:,1].T),plt.title('Body'),plt.show(),
plt.plot(euc_err[:,25:,2].T),plt.title('Tail'),plt.show(),
#%%
thisInd = train_generator.val_index
thisPred = predictions[thisInd]
thisData = allData[thisInd]
errors =  thisPred[:,:,:2]-thisData
nonvisConf = thisPred[np.isnan(thisData[:,:,0]),2]
visConf = thisPred[~np.isnan(thisData[:,:,0]),2]
plt.hist([visConf, nonvisConf], bins=25, density=True),\
plt.legend(['Visible', 'Non-Visible']),plt.xlabel('Confidence'),plt.show()

errDist = np.sqrt(np.sum(errors**2, axis=2))
plt.plot(np.nanmean(errDist, axis=0)),plt.plot(np.nanmedian(errDist, axis=0)),\
plt.ylabel('err dist (px)'),plt.xlabel('keypoint'),\
plt.legend(['Mean','Median']),plt.show()
#%% Display Images
idx = np.random.randint(len(train_generator.val_index))
idx = train_generator.val_index[idx]

image = image_generator[idx][0, ..., 0]
keypoints = predictions[idx]

plt.figure(figsize=(5,5))
plt.imshow(image, cmap='gray', interpolation='none')
plt.scatter(keypoints[:, 0], keypoints[:, 1], c=np.arange(data_generator.keypoints_shape[0]),
            s=50*keypoints[:,2]**2, cmap=plt.cm.Set3, zorder=3)
plt.xlim(keypoints[1,0] + [-50,50])
plt.ylim(keypoints[1,1] + [50,-50])
plt.show()

