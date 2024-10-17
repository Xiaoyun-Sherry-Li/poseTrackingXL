from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers.experimental import preprocessing
import numpy as np
import glob
from scipy.io import loadmat
import tensorflow as tf
import matplotlib.pyplot as plt
#%% config
gpu = 0
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[gpu],'GPU')
tf.config.experimental.set_memory_growth(gpus[gpu], True)

#%% Load Data
labelDir = "Z:\Selmaan\Seed Carrying Labeling\Labeled Data"
# allFn = glob.glob(labelDir+'\*.mat')
# labelDir = "/media/selmaan/Locker/Selmaan/Seed Carrying Labeling/Labeled Data"
allFn = glob.glob(labelDir+'/seedLabel_*.mat') + glob.glob(labelDir+'/manualSeedRelabel*.mat')
allIms = []
allLabels = []
for fn in allFn:
    tmp = loadmat(fn)
    allIms.append(tmp['allIms'])
    allLabels.append(tmp['seedLabels'])

allIms = np.transpose(np.concatenate(allIms, axis=3), [3,0,1,2])
allLabels = np.concatenate(allLabels, axis=0).flatten()
# select only valid image/labels
validLabels = np.isfinite(allLabels)
allIms = allIms[validLabels]
allLabels = allLabels[validLabels]
nLabels = validLabels.sum()

# recast data
allIms = allIms.astype('float32')
allLabels = allLabels.astype('float32')
#%% Network architecture
import faceNetArchitectures as fn

# define input layer
inputs_shape = allIms.shape[1:]
# viewMdl = keras.models.load_model('s4-singleView')

#train from a naive network (be sure to make all layers trainable in joint model!)
data_augmentation = Sequential(
    [
        preprocessing.RandomFlip("horizontal"),
        preprocessing.RandomTranslation(height_factor = 0.2, width_factor=0.2, fill_mode="reflect", interpolation="bilinear"),
        preprocessing.RandomRotation(factor=.1, fill_mode="reflect", interpolation="bilinear")
    ]
)
viewMdl = fn.s5((128,128,1), data_augmentation)

# make joint prediction model
jointMdl = fn.j4(inputs_shape, viewMdl)

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
# define callbacks to reduce LR
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_joint_pred_auc", mode="max", factor=0.25, patience=10)
stop_tr = keras.callbacks.EarlyStopping(monitor="val_joint_pred_auc", mode="max", patience=31, restore_best_weights=False)

# Train joint model
## maybe use learning rate 1e-3 when using j4 model?
opt = keras.optimizers.Adam(learning_rate=5e-4)
jointMdl.compile(optimizer=opt, loss=keras.losses.BinaryCrossentropy(),
                 metrics=keras.metrics.AUC(), loss_weights=[0.75, 0.25]),
hist = jointMdl.fit(x=allIms[allTrainInd], y=allLabels[allTrainInd], validation_data=(allIms[allValidInd], allLabels[allValidInd]),
        epochs=1000, batch_size=200, class_weight=classBalanceDict,
        callbacks=[reduce_lr, stop_tr])

# plot training history
h = hist.history
plt.plot(h['loss'][10:]),plt.plot(h['val_loss'][10:]),plt.show()
plt.plot(h['joint_pred_auc'][10:]),plt.plot(h['val_joint_pred_auc'][10:]),plt.show()

#%% Prediction
# select indices for display
testIndices = allValidInd
# testIndices = np.arange(17000,17700)

# reshape 6-channel images into 2x3 montages
valIms = allIms[testIndices]
valIms = np.concatenate((valIms[...,:3], valIms[...,3:]),axis=1)
valIms = np.concatenate((valIms[...,0], valIms[...,1], valIms[...,2]),axis=2)

# define new network with view-specific weights and make predictions
weights_layer = [l for l in jointMdl.layers if l.name=='view_weights'][0]
jp_layer = [l for l in jointMdl.layers if l.name=='joint_pred'][0]
predMdl = keras.Model(inputs=jointMdl.input, outputs=[jp_layer.output, weights_layer.output])
val = predMdl.predict(allIms[testIndices], batch_size=200)

# display 10 random images
for i in np.random.randint(len(testIndices), size=(10,1)):
    plt.imshow(valIms[i[0]], cmap='gray'),
    weightString = [str(v+1) + '-' + str(np.round(val[1][i,v], decimals=1)) for v in range(6)]
    weightString = ', '.join(weightString)
    plt.title('Score-' + str(np.round(val[0][i][0], decimals=3)) + ': ' + weightString),
    plt.show(),

# show histogram of predicted values for true and false targets separately
jp = val[0].flatten()
plt.hist([jp[allLabels[testIndices]==0],jp[allLabels[testIndices]==1]],density=True),
plt.show(),
