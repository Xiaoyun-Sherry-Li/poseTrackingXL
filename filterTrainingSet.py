#%%
import numpy as np
import pySBA
from supervisely_utils import get_pt_dict, getImageAndAnnotations, getCameraArray, resize_for_crop, crop_from_predictions
from deepposekit.models import load_model

ds_size = (704, 352)
crop_size = (320, 320)
crop_border_fac = 0.85
confidence_thresh = 0.1
basePath = 'D:\\Supervisely\\Chickadee-1\\'
datasets = ['IND32 200226', 'IND32 200309', 'AMB23 200315',
            'IND32 200315', 'EMR36 200327']
skeleton = 'C:\\Users\\Selmaan\\PycharmProjects\\supervisely_dpk_training\\skeleton.csv'
cropModelPath = 'C:\\Users\\Selmaan\\PycharmProjects\supervisely_dpk_training\\cropNet\\cropNet_Model.h5'
postureModelPath = 'C:\\Users\\Selmaan\\PycharmProjects\supervisely_dpk_training\\cropNet\\postureNet_Model.h5'
crop_model = load_model(cropModelPath, compile=False)
posture_model = load_model(postureModelPath, compile=False)
ptDict = get_pt_dict(basePath=basePath)

#%%
imgData = []
kData = []
ind3D = []
indCam = []
for n, dataset in enumerate(datasets):
    sideData = dataset + ' SideImages'
    topData = dataset + ' TopImages'
    iSide, aSide, mSide = getImageAndAnnotations(sideData, ptDict, basePath, center_coords=False)
    iTop, aTop, mTop = getImageAndAnnotations(topData, ptDict, basePath, center_coords=False)
    i = iSide + iTop
    a = aSide + aTop
    m = mSide + mTop
    for nIm in range(len(i)):
        ds_img, ds_fac = resize_for_crop(i[nIm], ds_size)
        preds = crop_model.predict(ds_img.reshape((1, ds_img.shape[0], ds_img.shape[1], 1)))[0]
        preds = ds_fac * preds[preds[:, 2] > confidence_thresh, :2]
        crop_img, min_ind, max_ind = crop_from_predictions(i[nIm], preds, crop_size, crop_border_fac=crop_border_fac)
        crop_keypoints = posture_model.predict(crop_img.reshape((1, crop_img.shape[0], crop_img.shape[1], 1)))[0]
        crop_scale = crop_size / (max_ind - min_ind)
        nCam = m[nIm][0]
        # if (nCam==2) | (nCam==5):
        #     imCenter = np.array([2816 / 2, 1696 / 2])
        # else:
        #     imCenter = np.array([2816 / 2, 1408 / 2])
        keypoints = crop_keypoints
        keypoints[:, :2] = crop_keypoints[:, :2] / crop_scale + min_ind # - imCenter
        kData.append(keypoints)
        ind3D.append(m[nIm][2] + 25 * len(ptDict) * n)  # assumes each dataset has 25 images
        indCam.append(nCam)
        imgData.append(i[nIm])

kData = np.asarray(kData)
ind3D = np.asarray(ind3D)
indCam = np.asarray(indCam)
imgData = np.asarray(imgData)

#%%
#Optional, calibrate cameras only to selected points
ptSelect = np.array([0, 1, 8, 11, 12, 13, 16, 17])
tmp = np.zeros_like(kData)
tmp[:,ptSelect,:] = kData[:,ptSelect,:]

points_3d = np.full( (ind3D.max()+1, 3), 0.0)
points_2d = tmp.transpose([1, 0, 2]).reshape([-1, 3], order='F')
point_2dind3d = ind3D.reshape(-1, order='C')
camera_ind = np.tile(indCam, (len(ptDict),1)).reshape(-1, order='F')
cameraArray = getCameraArray()
sba = pySBA.PySBA(cameraArray, points_3d, points_2d[:,:2], camera_ind, point_2dind3d, points_2d[:,2])
sba.bundleAdjust_nocam()
sba.bundleAdjust_sharedcam(),
# sba.bundleAdjust(),
optCamArray = sba.cameraArray.copy()

#%%
points_3d = np.full( (ind3D.max()+1, 3), 0.0)
points_2d = kData.transpose([1, 0, 2]).reshape([-1, 3], order='F')
point_2dind3d = ind3D.reshape(-1, order='C')
camera_ind = np.tile(indCam, (len(ptDict),1)).reshape(-1, order='F')
cameraArray = optCamArray

sba = pySBA.PySBA(cameraArray, points_3d, points_2d[:,:2], camera_ind, point_2dind3d, points_2d[:,2])
sba.bundleAdjust_nocam(),
r = sba.project(sba.points3D[sba.point2DIndices], sba.cameraArray[sba.cameraIndices]) - sba.points2D
r = np.sqrt(np.sum(r**2, axis=1))
point_ind = np.mod(sba.point2DIndices, len(ptDict))


#%%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas

validInd = points_2d[:,2]>.25
stdPts = ['topBeak', 'botBeak','topHead','backHead','centerChest','centerBack','baseTail','tipTail',
            'leftEye','leftNeck','leftWing','leftAnkle','leftFoot',
           'rightEye','rightNeck','rightWing','rightAnkle','rightFoot']
data = pandas.DataFrame(data = {'rp_err': r,
                                'keypoint': np.array(stdPts)[point_ind]})

sns.catplot(x='rp_err',y='keypoint', data=data[validInd], kind='boxen', order=stdPts),
plt.title('Reproj. Error: %f; ' %np.mean(r[validInd]))
plt.xlim([-1, 25])
plt.show()

#%%
# Save images, 3d coordinates, and camera parameters
from scipy.io import savemat
savemat('training3d_shared.mat',{'images':imgData, 'coords3D':sba.points3D,
                     'camParam':sba.cameraArray, 'stdPts':stdPts})

#%%
idx = np.random.randint(len(imgData))
image = imgData[idx]

tmp = np.tile(indCam[idx],len(ptDict))
keypoints = sba.project(sba.points3D[ind3D[idx]], sba.cameraArray[tmp])
# if image.shape[0] > 1500:
#     imCenter = np.array([2816 / 2, 1696 / 2])
# else:
#     imCenter = np.array([2816 / 2, 1408 / 2])
# keypoints[:,:2] = keypoints[:,:2] + imCenter

plt.figure(figsize=(5,5))
plt.imshow(image, cmap='gray', interpolation='none')
plt.scatter(keypoints[:, 0], keypoints[:, 1], c=np.arange(keypoints.shape[0]),
            s=50, cmap=plt.cm.Set1, zorder=3)
centroid = np.median(keypoints,axis=0)[:2]
plt.xlim([centroid[0]-250, centroid[0]+250])
plt.ylim([centroid[1]+250, centroid[1]-250])
plt.show()