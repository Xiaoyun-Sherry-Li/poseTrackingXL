#%%
import numpy as np
from deepposekit.io import initialize_dataset
from deepposekit import Annotator
import csv
import pySBA
import mat73
from dpk_utils import resize_and_pad_rows, crop_from_com
from deepposekit.models import load_model

w3d = 0.125
crop_size = (320, 320)
ds_fac = 4
ds_size = (2816//ds_fac, 1696//ds_fac)
drop_predict_ind = [] #index of annotations to set to NaN
skeleton = '.\\postureSkeleton.csv'
dataGenPath = '.\\postureNet_Dataset.h5'
comModelPath = 'Z:\Selmaan\DPK-transfer\\comNet_Model_05.h5'
comModel = load_model(comModelPath, compile=False)
trainingFiles = []
with open('..\comNet\comTrainingFiles.csv', newline='') as csvFile:
    reader = csv.reader(csvFile)
    for row in reader:
        trainingFiles.append(row[0])

def formatData(data3D, nParts=18):
    nFrames = data3D.shape[0]
    data3D = np.reshape(data3D, (nFrames, nParts, 3))
    return (data3D, nFrames, nParts)

def projectData(matfile):
    camParams = pySBA.convertParams(matfile['camParams'])
    (pt3d, nFrames, nParts) = formatData(matfile['data_3D'])
    sba = pySBA.PySBA(camParams, np.NaN, np.NaN, np.NaN, np.NaN) #points_2d[:, :2], camera_ind, point_2dind3d, points_2d[:, 2])
    nCams = camParams.shape[0]
    allLabels = np.full((nFrames, nCams, nParts, 2), np.NaN)
    allCamScales = np.full((nFrames,nCams), np.NaN)
    for nCam in range(nCams):
        rVec = camParams[nCam][:3].reshape((1, 3))
        tVec = camParams[nCam][3:6]
        for nPart in range(nParts):
            allLabels[:, nCam, nPart, :] = sba.project(pt3d[:, nPart], np.tile(camParams[nCam], (nFrames,1)))
        pt3d_centroid = np.mean(pt3d,axis=1) # average over parts
        pt3d_centroid = sba.rotate(pt3d_centroid, np.tile(rVec, (nFrames, 1))) # rotate to camera coordinates
        camDist = pt3d_centroid[:,2] + tVec[2] # get z-axis distance ie along optical axis
        camScale = camParams[nCam][6] / camDist # convert to focal length divided by distance
        allCamScales[:, nCam] = camScale

    return allLabels, allCamScales

#%%
allLabels = []
allImages = []
allScales = []
for fn in trainingFiles:
    print(fn)
    matfile = mat73.loadmat(fn)
    theseLabels, theseScales = projectData(matfile)
    theseImages = []
    for data in matfile['videos']:
        theseImages.append(data[0])
    allLabels.append(theseLabels)
    allImages.append(theseImages)
    allScales.append(theseScales)

allLabels = np.concatenate(allLabels, axis=0)
allScales = np.concatenate(allScales, axis=0)
nCams = allLabels.shape[1]
allCam = []
for nCam in range(nCams):
    allCam.append(np.concatenate([i[nCam] for i in allImages], axis=2))
del allImages
#%%
import matplotlib.pyplot as plt

nFrame = 50
for nCam in range(nCams):
    plt.imshow(theseImages[nCam][:,:,nFrame], cmap='gray')
    plt.plot(theseLabels[nFrame,nCam,0,0],theseLabels[nFrame,nCam,0,1],'r*')
    plt.plot(theseLabels[nFrame,nCam,5,0],theseLabels[nFrame,nCam,5,1],'b*')
    plt.plot(theseLabels[nFrame,nCam,17,0],theseLabels[nFrame,nCam,17,1],'g*')
    centroid = np.mean(theseLabels[nFrame,nCam],axis=0)
    thisScale = w3d * theseScales[nFrame,nCam],
    print([nCam, thisScale])
    plt.xlim(centroid[0]-thisScale,centroid[0]+thisScale),
    plt.ylim([centroid[1]+thisScale,centroid[1]-thisScale]),
    plt.show(),

#%%
imData = []
annData = []
for nCam in range(nCams):
    thisCam = np.transpose(allCam[nCam], axes=[2, 0, 1])
    thisLabel = allLabels[:, nCam, :, :]
    thisScale = allScales[:, nCam]
    nFrames = thisLabel.shape[0]
    for nFrame in range(nFrames):
        if np.isfinite(thisLabel[nFrame].sum()):
            ds_im = resize_and_pad_rows(thisCam[nFrame], ds_size)
            preds = comModel.predict(ds_im.reshape((1, ds_im.shape[0], ds_im.shape[1], 1)))[0]
            bodyCom = np.round(ds_fac * preds[1,:2]) # take x-y body coordinates
            half_width = np.round(w3d * thisScale[nFrame])
            crop_img, min_ind, crop_scale = crop_from_com(thisCam[nFrame], 
                                                            bodyCom,
                                                            half_width,
                                                            crop_size)
            crop_ann = (thisLabel[nFrame] - min_ind) * crop_scale
            imData.append(crop_img)
            annData.append(crop_ann)

imData = np.asarray(imData)
annData = np.asarray(annData)
annData[:, drop_predict_ind, :] = np.NaN
#%%

initialize_dataset(datapath = dataGenPath,
                    images=imData[:, :, :, np.newaxis],
                    skeleton=skeleton,
                    keypoints=annData,
                    overwrite=True)

#%%

app = Annotator(datapath=dataGenPath,
                dataset='images',
                skeleton=skeleton,
                shuffle_colors=False,
                scale=3,
                text_scale=1/4)
app.run()