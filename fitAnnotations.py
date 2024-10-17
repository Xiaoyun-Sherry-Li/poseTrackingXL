#%% Imports

import numpy as np
import matplotlib.pyplot as plt
from deepposekit.io import TrainingGenerator, DataGenerator
from deepposekit.augment import FlipAxis
import imgaug.augmenters as iaa
import imgaug as ia
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from deepposekit.callbacks import Logger, ModelCheckpoint

modelPath = 'C:\\Users\\Selmaan\\PycharmProjects\supervisely_dpk_training\\model_testing.h5'
dataGenPath = 'C:\\Users\\Selmaan\\PycharmProjects\supervisely_dpk_training\\testing.h5'
skeleton = 'C:\\Users\\Selmaan\\PycharmProjects\supervisely_dpk_training\\skeleton.csv'
logPath = 'C:\\Users\\Selmaan\\PycharmProjects\supervisely_dpk_training\\log_testing.h5'
#%% Define Augmentations

data_generator = DataGenerator(dataGenPath, mode='full')
augmenter = []
augmenter.append(FlipAxis(data_generator, axis=1)) # flip image left-right with 50% prob
augmenter.append(iaa.Sometimes(0.8, iaa.Affine(scale={"x": (0.95, 1.05), "y": (0.95, 1.05)},
                            translate_percent={'x': (-0.05, 0.05), 'y': (-0.05, 0.05)},
                            order=ia.ALL,
                            cval=ia.ALL,
                            mode='symmetric')))
augmenter.append(iaa.Sometimes(0.8, iaa.Affine(scale=(0.8, 1.2),
                                               mode = 'symmetric',
                                               order = ia.ALL,
                                               cval=ia.ALL)))
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
plt.scatter(keypoints[0, :, 0], keypoints[0, :, 1], c=np.arange(data_generator.keypoints_shape[0]), s=50, cmap=plt.cm.hsv, zorder=3)

plt.show()

#%% Set up training data pipeline and options

train_generator = TrainingGenerator(generator=data_generator,
                                    downsample_factor=2,
                                    augmenter=augmenter,
                                    validation_split=0.15,
                                    sigma=10,
                                    use_graph=False)
#train_generator.get_config()

#%% Set up model

from deepposekit.models import StackedDenseNet

model = StackedDenseNet(train_generator, pretrained=False, n_stacks=1, growth_rate=24, n_transitions=5)
#model.get_config()

#%% Set up callbacks
#should probably monitor val_output_1_loss instead of val_loss for choosing best model!
logger = Logger(validation_batch_size=1, filepath=logPath)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, verbose=1, patience=10)
model_checkpoint = ModelCheckpoint(
    modelPath,
    monitor="val_loss",
    verbose=1,
    save_best_only=True,
)
early_stop = EarlyStopping(
    monitor="val_loss",
    min_delta=0.001,
    patience=20,
    verbose=1
)

callbacks = [early_stop, reduce_lr, model_checkpoint, logger]

#%% Fit the Model
model.fit(
    batch_size=1,
    validation_batch_size=1,
    callbacks=callbacks,
    epochs=500,
    n_workers=8,
    steps_per_epoch=None,
    use_multiprocessing=False
)

#%%
from deepposekit.io import ImageGenerator

image_generator = ImageGenerator(data_generator)
predictions = model.predict(image_generator, verbose=1)

allData = data_generator.get_keypoints([data_generator.index])

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
import pandas as pd
ptNames = pd.read_csv(skeleton)['name']

idx = np.random.randint(len(image_generator))

image = image_generator[idx][0, ..., 0]
keypoints = predictions[idx]

plt.figure(figsize=(5,5))
plt.imshow(image, cmap='gray', interpolation='none')
plt.scatter(keypoints[:, 0], keypoints[:, 1], c=np.arange(data_generator.keypoints_shape[0]),
            s=50*keypoints[:,2]**2, cmap=plt.cm.Set1, zorder=3)
kMin = keypoints.min(axis=0)[:2]
kMax = keypoints.max(axis=0)[:2]
plt.xlim([kMin[0]-50, kMax[0]+50]),
plt.ylim([kMax[1]+50, kMin[1]-50])
plt.show()

# for i in range(keypoints.shape[0]):
#     plt.figure(figsize=(5,5))
#     plt.imshow(image, cmap='gray', interpolation='none')
#     plt.plot(keypoints[i, 0], keypoints[i, 1],'rx')
#     plt.xlim([kMin[0]-50, kMax[0]+50]),
#     plt.ylim([kMax[1]+50, kMin[1]-50]),
#     plt.title('%s_conf: %f' %(ptNames[i],keypoints[i,2]))
#     plt.show()
