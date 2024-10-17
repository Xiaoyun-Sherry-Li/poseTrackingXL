#%% Imports

import numpy as np
import matplotlib.pyplot as plt
from deepposekit.io import TrainingGenerator, DataGenerator
from deepposekit.augment import FlipAxis
import imgaug.augmenters as iaa
import imgaug as ia
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from deepposekit.callbacks import Logger, ModelCheckpoint
skeleton = "./fullSkeleton.csv"
dataGenPath = "./fullNet_Dataset.h5"
modelPath = "./fullNet_Model.h5"
logPath = "./fullNet_Log.h5"
batch_size = 4
#%% Define Augmentations

data_generator = DataGenerator(dataGenPath, mode='full')
augmenter = []
augmenter.append(FlipAxis(data_generator, axis=1)) # flip image left-right with 50% prob
# augmenter.append(iaa.flip.Fliplr(p=0.5))
augmenter.append(iaa.Sometimes(0.8, iaa.Affine(scale={"x": (0.95, 1.05), "y": (0.95, 1.05)},
                            translate_percent={'x': (-0.15, 0.15), 'y': (-0.15, 0.15)},
                            order=ia.ALL,
                            cval=ia.ALL,
                            mode='symmetric')))
augmenter.append(iaa.Sometimes(0.8, iaa.Affine(scale=(0.75, 1.25),
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
plt.scatter(keypoints[0, :, 0], keypoints[0, :, 1], c=np.arange(data_generator.keypoints_shape[0]), s=15, cmap=plt.cm.hsv, zorder=3)

plt.show()

#%% Set up training data pipeline and options

train_generator = TrainingGenerator(generator=data_generator,
                                    downsample_factor=2,
                                    augmenter=augmenter,
                                    validation_split=0.1,
                                    sigma=5,
                                    use_graph=False)
#train_generator.get_config()

#%% Set up model

from deepposekit.models import StackedDenseNet

# model = StackedDenseNet(train_generator, pretrained=False, n_stacks=1, growth_rate=48, n_transitions=-1)
model = StackedDenseNet(train_generator, n_stacks=2, growth_rate=48, n_transitions=-1)
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

callbacks = [early_stop, reduce_lr, model_checkpoint, logger]

#%% Fit the Model
model.fit(
    batch_size=batch_size,
    validation_batch_size=batch_size,
    callbacks=callbacks,
    epochs=999,
    n_workers=32,
    steps_per_epoch=None,
    use_multiprocessing=True
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

import seaborn as sns
import pandas
stdPts = ['topBeak', 'botBeak','topHead','backHead','centerChest','centerBack','baseTail','tipTail',
            'leftEye','leftNeck','leftWing','leftAnkle','leftFoot',
           'rightEye','rightNeck','rightWing','rightAnkle','rightFoot']
pt_label = []
for n in range(errDist.shape[0]):
    pt_label.extend(stdPts)
data = pandas.DataFrame(data = {'ann_error': errDist.ravel(),
                                'keypoint': pt_label})
sns.catplot(x='ann_error',y='keypoint',data=data, kind='boxen', order=stdPts),
plt.xlim([-1, 20])
plt.show()
#%% Display Images
idx = np.random.randint(len(train_generator.val_index))
idx = train_generator.val_index[idx]

image = image_generator[idx][0, ..., 0]
keypoints = predictions[idx]

plt.figure(figsize=(5,5))
plt.imshow(image, cmap='gray', interpolation='none')
plt.scatter(keypoints[:, 0], keypoints[:, 1], c=np.arange(data_generator.keypoints_shape[0]),
            s=50*keypoints[:,2]**2, cmap=plt.cm.Set1, zorder=3)
for idx, jdx in enumerate(data_generator.graph):
    if jdx > -1:
        plt.plot(
            [keypoints[idx, 0], keypoints[jdx, 0]],
            [keypoints[idx, 1], keypoints[jdx, 1]],
            'r-'
        )
plt.xlim(keypoints[5,0] + [-50,50])
plt.ylim(keypoints[5,1] + [50,-50])
plt.show()

