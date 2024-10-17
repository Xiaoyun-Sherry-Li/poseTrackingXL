from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers.experimental import preprocessing
import numpy as np
import glob
from scipy.io import loadmat
import tensorflow as tf
import matplotlib.pyplot as plt
#%% config
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

#%% Load Data
labelDir = "Z:\Selmaan\Seed Carrying Labeling\Labeled Data"
# allFn = glob.glob(labelDir+'\seedLabel_*.mat') + glob.glob(labelDir+'\\autoSeedRelabel_*.mat') + glob.glob(labelDir+'\\manualSeedRelabel_*.mat')
allFn = glob.glob(labelDir+'\*.mat')
allIms = []
allLabels = []
for fn in allFn:
    tmp = loadmat(fn)
    allIms.append(tmp['allIms'])
    allLabels.append(tmp['seedLabels'])

allIms = np.transpose(np.reshape(np.concatenate(allIms, axis=3), (128,128,-1)), [2,0,1])
allIms = np.expand_dims(allIms, axis=3)
allLabels = np.tile(np.concatenate(allLabels, axis=0).flatten(), 6)
# select only valid image/labels
validLabels = np.isfinite(allLabels)
allIms = allIms[validLabels]
allLabels = allLabels[validLabels]
nLabels = validLabels.sum()

# recast data
allIms = allIms.astype('float32')
allLabels = allLabels.astype('float32')
#%% Network architecture
import faceNet.faceNetArchitectures as fn

# define input layer
inputs_shape = allIms.shape[1:]
data_augmentation = Sequential(
    [
        preprocessing.RandomFlip("horizontal_and_vertical"),
        preprocessing.RandomTranslation(height_factor = 0.2, width_factor=0.2, fill_mode="reflect", interpolation="bilinear"),
        preprocessing.RandomRotation(factor=.1, fill_mode="reflect", interpolation="bilinear")
    ]
)

mdl = fn.s4(inputs_shape, data_augmentation)

#%% Train/Validation Split
validFrac = 0.1
nValid = round(validFrac * len(allLabels))
allInd = np.random.permutation(len(allLabels))
allTrainInd = allInd[nValid:]
allValidInd = allInd[:nValid]
classBalanceDict = {0:len(allTrainInd)/(allLabels[allTrainInd] == 0).sum()/2,
                   1:len(allTrainInd)/(allLabels[allTrainInd] == 1).sum()/2}
print(classBalanceDict)

# validFrac = 0.1 # pick this fraction of positive labeled frames
# balanceFactor = 4 # pick this many negative labeled frames for each positive
#
# # Create balanced validation set
# trueLabels = np.where(allLabels==1)[0]
# falseLabels = np.where(allLabels==0)[0]
# numTrueValid = round(validFrac * len(trueLabels))
# numFalseValid = balanceFactor * numTrueValid
# # Add correct ratios to validation set, and all else to training set
# tmpInd = np.random.permutation(len(trueLabels))
# trueTrainInd = trueLabels[tmpInd[numTrueValid:]]
# trueValidInd = trueLabels[tmpInd[:numTrueValid]]
# tmpInd = np.random.permutation(len(falseLabels))
# falseTrainInd = falseLabels[tmpInd[numFalseValid:]]
# falseValidInd = falseLabels[tmpInd[:numFalseValid]]
# # Combine into train and valid sets
# allTrainInd = np.concatenate((trueTrainInd,falseTrainInd),axis=0)
# allValidInd = np.concatenate((trueValidInd,falseValidInd),axis=0)
# # set class weights within training set
# classBalanceDict = {0:(balanceFactor+1)/(balanceFactor*2), 1:(balanceFactor+1)/2}

#%% Model Training
# define callbacks to reduce LR (and to save model checkpoints?)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.25, patience=20)
stop_tr = keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=45, restore_best_weights=False)
# reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", mode="min", factor=0.25, patience=20)
# stop_tr = keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=45, restore_best_weights=False)

# fit model, include accuracy metric, define validation data, and weight sample classes by inverse proportionality
opt = keras.optimizers.Adam(learning_rate=1e-3)
mdl.compile(optimizer=opt, loss=keras.losses.BinaryCrossentropy(), metrics=keras.metrics.AUC()),
hist = mdl.fit(x=allIms[allTrainInd], y=allLabels[allTrainInd], validation_data=(allIms[allValidInd], allLabels[allValidInd]),
        epochs=1000, batch_size=1500, class_weight=classBalanceDict,
        callbacks=[reduce_lr, stop_tr])

# plot training history
h = hist.history
plt.plot(h['loss']),plt.plot(h['val_loss']),plt.show()
plt.plot(h['auc']),plt.plot(h['val_auc']),plt.show()

#%% Prediction
# select indices for display
testIndices = allValidInd

valIms = allIms[testIndices]

# predict on all validation data and display 10 random images with prediction
val = mdl.predict(allIms[testIndices], batch_size=1500).flatten()
for i in np.random.randint(len(testIndices), size=(10,1)):
    plt.imshow(valIms[i[0]]),
    plt.title('Im#' + str(i) + ': ' + str(val[i])),
    plt.show(),

# show histogram of predicted values for true and false targets separately
plt.hist([val[allLabels[testIndices]==0],val[allLabels[testIndices]==1]],density=True),
plt.show(),
