#%%
import numpy as np
from deepposekit.io import initialize_dataset
from deepposekit import Annotator
import csv
import pySBA
import mat73
from dpk_utils import resize_and_pad_rows

ds_fac = 4
ds_size = (2816//ds_fac, 1696//ds_fac)
skeleton = 'C:\\Users\\Selmaan\\PycharmProjects\\supervisely_dpk_training\\comNet\\comSkeleton.csv'
dataGenPath = 'C:\\Users\\Selmaan\\PycharmProjects\\supervisely_dpk_training\\comNet\\comNet_Dataset.h5'
trainingFiles = []
with open('.\comTrainingFiles.csv', newline='') as csvFile:
    reader = csv.reader(csvFile)
    for row in reader:
        trainingFiles.append(row[0])

def avgBodyParts(data3D, nParts=18):
    headInd = [8,13]
    bodyInd = [4,5,6,10,15]
    tailInd = [7]
    nFrames = data3D.shape[0]
    data3D = np.reshape(data3D, (nFrames,nParts,3))
    headPts = data3D[:,headInd,:].mean(axis=1)
    bodyPts = data3D[:,bodyInd,:].mean(axis=1)
    tailPts = data3D[:,tailInd,:].mean(axis=1)
    # data3D = np.reshape(data3D, (nFrames, 3, 18))
    # headPts = data3D[:, :, headInd].mean(axis=2)
    # bodyPts = data3D[:, :, bodyInd].mean(axis=2)
    # tailPts = data3D[:, :, tailInd].mean(axis=2)
    return (headPts, bodyPts, tailPts)

def projectData(matfile):
    camParams = pySBA.convertParams(matfile['camParams'])
    pt3d = avgBodyParts(matfile['data_3D'])
    sba = pySBA.PySBA(camParams, np.NaN, np.NaN, np.NaN, np.NaN)
    nFrames = pt3d[0].shape[0]
    nCams = camParams.shape[0]
    nParts = len(pt3d)
    allLabels = np.full((nFrames, nCams, nParts, 2), np.NaN)
    for nCam in range(nCams):
        for nPart in range(nParts):
            allLabels[:, nCam, nPart, :] = sba.project(pt3d[nPart], np.tile(camParams[nCam],(nFrames,1)))

    return allLabels

#%%
allLabels = []
allImages = []
for fn in trainingFiles:
    print(fn)
    matfile = mat73.loadmat(fn)
    theseLabels = projectData(matfile)
    theseImages = []
    for data in matfile['videos']:
        theseImages.append(data[0])
    allLabels.append(theseLabels)
    allImages.append(theseImages)

allLabels = np.concatenate(allLabels, axis=0)
nCams = allLabels.shape[1]
allCam = []
for nCam in range(nCams):
    allCam.append(np.concatenate([i[nCam] for i in allImages], axis=2))
del allImages
#%%
import matplotlib.pyplot as plt
nCam = 2
nFrame = 10
plt.imshow(theseImages[nCam][:,:,nFrame], cmap='gray')
plt.plot(theseLabels[nFrame,nCam,0,0],theseLabels[nFrame,nCam,0,1],'r*')
plt.plot(theseLabels[nFrame,nCam,1,0],theseLabels[nFrame,nCam,1,1],'b*')
plt.plot(theseLabels[nFrame,nCam,2,0],theseLabels[nFrame,nCam,2,1],'g*')
plt.show(),


#%%
imData = []
annData = []
for nCam in range(nCams):
    thisCam = np.transpose(allCam[nCam], axes=[2,0,1])
    thisLabel = allLabels[:,nCam,:,:]
    nFrames = thisLabel.shape[0]
    for nFrame in range(nFrames):
        if np.isfinite(thisLabel[nFrame].sum()):
            ds_ann = thisLabel[nFrame] / ds_fac
            ds_im = resize_and_pad_rows(thisCam[nFrame], ds_size)
            imData.append(ds_im)
            annData.append(ds_ann)

imData = np.asarray(imData)
annData = np.asarray(annData)

#%%

initialize_dataset(datapath = dataGenPath, images=imData[:,:,:,np.newaxis], skeleton=skeleton, keypoints=annData, overwrite=True)

#%%

app = Annotator(datapath=dataGenPath,
                dataset='images',
                skeleton=skeleton,
                shuffle_colors=False,
                scale=3,
                text_scale=1/4)
app.run()