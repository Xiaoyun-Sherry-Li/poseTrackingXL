#%%
import numpy as np
from deepposekit.io import initialize_dataset
from deepposekit import Annotator
import csv
import pySBA
import mat73
from dpk_utils import resize_and_pad_rows

ds_fac = 2
ds_size = (2816//ds_fac, 1696//ds_fac)
skeleton = '.\\fullSkeleton.csv'
dataGenPath = '.\\fullNet_Dataset.h5'
trainingFiles = []
with open('..\comNet\comTrainingFiles.csv', newline='') as csvFile:
    reader = csv.reader(csvFile)
    for row in reader:
        trainingFiles.append(row[0])

def formatData(data3D, nParts=18):
    nFrames = data3D.shape[0]
    data3D = np.reshape(data3D, (nFrames,nParts,3))
    return (data3D, nFrames, nParts)

def projectData(matfile):
    camParams = pySBA.convertParams(matfile['camParams'])
    (pt3d, nFrames, nParts) = formatData(matfile['data_3D'])
    sba = pySBA.PySBA(camParams, np.NaN, np.NaN, np.NaN, np.NaN) #points_2d[:, :2], camera_ind, point_2dind3d, points_2d[:, 2])
    nCams = camParams.shape[0]
    allLabels = np.full((nFrames, nCams, nParts, 2), np.NaN)
    for nCam in range(nCams):
        for nPart in range(nParts):
            allLabels[:, nCam, nPart, :] = sba.project(pt3d[:,nPart], np.tile(camParams[nCam],(nFrames,1)))

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
nCam = 1
nFrame = 10
plt.imshow(theseImages[nCam][:,:,nFrame], cmap='gray')
plt.plot(theseLabels[nFrame,nCam,0,0],theseLabels[nFrame,nCam,0,1],'r*')
plt.plot(theseLabels[nFrame,nCam,5,0],theseLabels[nFrame,nCam,5,1],'b*')
plt.plot(theseLabels[nFrame,nCam,17,0],theseLabels[nFrame,nCam,17,1],'g*')
centroid = np.mean(theseLabels[nFrame,nCam],axis=0)
plt.xlim(centroid[0]-200,centroid[0]+200),
plt.ylim([centroid[1]+200,centroid[1]-200]),
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