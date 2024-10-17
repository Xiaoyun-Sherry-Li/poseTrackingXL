#%%
import numpy as np
import pySBA
from supervisely_utils import get_pt_dict, getImageAndAnnotations, getCameraArray

basePath = 'D:\\Supervisely\\Chickadee-1\\'
datasets = ['IND32 200226', 'IND32 200309', 'AMB23 200315']
           # 'IND32 200315', , 'EMR36 200327']


#%%
ptDict = get_pt_dict(basePath=basePath)
imData = []
annData = []
annotator = []
ind3D = []
indCam = []
for n, dataset in enumerate(datasets):
    sideData = dataset + ' SideImages'
    topData = dataset + ' TopImages'
    iSide, aSide, mSide = getImageAndAnnotations(sideData, ptDict, basePath, center_coords=True)
    iTop, aTop, mTop = getImageAndAnnotations(topData, ptDict, basePath, center_coords=True)
    i = iSide + iTop
    a = aSide + aTop
    m = mSide + mTop
    for nIm in range(len(i)):
        for nAnn in range(a[nIm].shape[1]):
            imData.append(i[nIm])
            annData.append(a[nIm][:,nAnn,:])
            ind3D.append(m[nIm][2] + 25*len(ptDict)*n) # assumes each dataset has 25 images
            annotator.append(m[nIm][3][nAnn])
            indCam.append(m[nIm][0])
imData = np.asarray(imData)
annData = np.asarray(annData)
ind3D = np.asarray(ind3D)
annotator = np.asarray(annotator)
indCam = np.asarray(indCam)

#%%
thisAnnotator = 0
annotator_ind = annotator < 2 #annotator == thisAnnotator
points_3d = np.full( (ind3D[annotator_ind].max()+1, 3), 0.0)
points_2d = annData[annotator_ind].transpose([1, 0, 2]).reshape([-1, 2], order='F')
point_2dind3d = ind3D[annotator_ind].reshape(-1, order='C')
camera_ind = np.tile(indCam[annotator_ind], (len(ptDict),1)).reshape(-1, order='F')
validInd = np.isfinite(np.sum(points_2d, axis=1))
cameraArray = getCameraArray()

sba = pySBA.PySBA(cameraArray, points_3d, points_2d[validInd], camera_ind[validInd], point_2dind3d[validInd])
sba.annotator = np.tile(annotator[annotator_ind], (len(ptDict),1)).reshape(-1, order='F')[validInd]
sba.bundleAdjust_nocam(),
sba.bundleAdjust_sharedcam(),
r = sba.project(sba.points3D[sba.point2DIndices], sba.cameraArray[sba.cameraIndices]) - sba.points2D
r = np.sqrt(np.sum(r**2, axis=1))
point_ind = np.mod(sba.point2DIndices, len(ptDict))

#%%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas

stdAnnotators = ['selmaan', 'stephaniemhale', 'nsobers']
stdPts = ['topBeak', 'botBeak','topHead','backHead','centerChest','centerBack','baseTail','tipTail',
            'leftEye','leftNeck','leftWing','leftAnkle','leftFoot',
           'rightEye','rightNeck','rightWing','rightAnkle','rightFoot']
data = pandas.DataFrame(data = {'rp_err': r,
                                'keypoint': np.array(stdPts)[point_ind],
                                'annotator': np.array(stdAnnotators)[sba.annotator]})

sns.catplot(x='rp_err',y='keypoint', hue='annotator' ,data=data, kind='boxen', order=stdPts),
plt.title(stdAnnotators[0] + ': %f; ' %np.mean(r[sba.annotator==0]) + stdAnnotators[1] + ': %f' %np.mean(r[sba.annotator==1]))
plt.xlim([-1, 25])
plt.show()

#%%

prct_clip = 99



#%%
#from deepposekit.io import initialize_dataset
#from deepposekit import Annotator
#imSize = (1408, 2816)
#skeleton = 'C:\\Users\\Selmaan\\PycharmProjects\\supervisely_dpk_training\\skeleton.csv'
#dataGenPath = 'C:\\Users\\Selmaan\\PycharmProjects\supervisely_dpk_training\\testing.h5'

# Expand image with zero border at right and bottom edges
zeroRows = np.full((imData.shape[0], imSize[0]-imData.shape[1], imData.shape[2]), 0, dtype='uint8')
zeroColumns = np.full((imData.shape[0], imSize[0], imSize[1] - imData.shape[2]), 0, dtype='uint8')
imData = np.concatenate( (np.concatenate((imData,zeroRows), axis=1), zeroColumns), axis=2)

#%%

initialize_dataset(datapath = dataGenPath, images=imData[:,:,:,np.newaxis], skeleton=skeleton, keypoints=annData, overwrite=True)

#%%

app = Annotator(datapath=dataGenPath,
                dataset='images',
                skeleton=skeleton,
                shuffle_colors=False,
                scale=3,
                text_scale=1)
app.run()