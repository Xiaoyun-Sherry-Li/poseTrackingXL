"""
Optionally can calibrate the camera using 2D pose tracking points instead of laser points.
To do so, update this code to work with your set-up...
"""
import numpy as np
import pySBA
from scipy.io import loadmat
import matplotlib.pyplot as plt
ptSelect = 0 # use top beak prediction only for self-calibration

tmp = loadmat('Z:\Selmaan\Birds\VLT186\VLT186_210211_100549\posture_2stage_3_11.mat')
pts = tmp['posture_rawpreds'][...,:2]
conf = tmp['posture_rawpreds'][...,2]
points_3d = tmp['posture_preds'][:,ptSelect,:] #initialize 3d positions w/ old triangulation

nPts = pts.shape[0]
nCams = pts.shape[1]
obs2d_pt = []
obs2d_ptInd = []
obs2d_cam = []
obs2d_conf = []
for nCam in range(nCams):
    obs2d_pt.append(pts[:,nCam,ptSelect])
    obs2d_ptInd.append(np.arange(nPts))
    obs2d_cam.append(np.full(nPts,nCam))
    obs2d_conf.append(conf[:,nCam,ptSelect])

obs2d_pt = np.concatenate(obs2d_pt)
obs2d_ptInd = np.concatenate(obs2d_ptInd)
obs2d_cam = np.concatenate(obs2d_cam)
obs2d_conf = np.concatenate(obs2d_conf)
cameraArray = pySBA.getCameraArray()

#%%
confThresh = 0.9
ptSelect = obs2d_conf>confThresh
sba = pySBA.PySBA(cameraArray, points_3d, obs2d_pt[ptSelect], obs2d_cam[ptSelect], obs2d_ptInd[ptSelect])
sba.bundleAdjust_nocam()
r = sba.project(sba.points3D[sba.point2DIndices], sba.cameraArray[sba.cameraIndices]) - sba.points2D
r = np.sqrt(np.sum(r**2, axis=1))
plt.hist(r[r<np.percentile(r, 95)]),plt.xlabel('Reprojection Error'),plt.title('no adjustment'),plt.show(),
inlierThresh = np.percentile(r, 80)
inlierInd = r < inlierThresh
sba2 = pySBA.PySBA(sba.cameraArray, sba.points3D, sba.points2D[inlierInd], sba.cameraIndices[inlierInd], sba.point2DIndices[inlierInd])
sba2.bundleAdjust_sharedcam(),
# sba2.bundleAdjust(),
optCamArray = sba2.cameraArray.copy()
sba = pySBA.PySBA(optCamArray, sba.points3D, sba.points2D, sba.cameraIndices, sba.point2DIndices)
sba.bundleAdjust_nocam()
r = sba.project(sba.points3D[sba.point2DIndices], sba.cameraArray[sba.cameraIndices]) - sba.points2D
r = np.sqrt(np.sum(r**2, axis=1))
plt.hist(r[r<np.percentile(r, 95)]),plt.xlabel('Reprojection Error'),plt.title('1st iteration'),plt.show(),
# rSum = np.zeros((nPts,1))
# for i in range(nPts):
#     rInd = obs2d_ptInd==i
#     rSum[i] = r[rInd].mean()

#%%
inlierThresh = np.percentile(r, 95)
inlierInd = r < inlierThresh
sba2 = pySBA.PySBA(sba.cameraArray, sba.points3D, sba.points2D[inlierInd], sba.cameraIndices[inlierInd], sba.point2DIndices[inlierInd])
sba2.bundleAdjust_sharedcam(),
optCamArray = sba2.cameraArray.copy()
r2 = sba2.project(sba2.points3D[sba2.point2DIndices], sba2.cameraArray[sba2.cameraIndices]) - sba2.points2D
r2 = np.sqrt(np.sum(r2**2, axis=1))
plt.hist(r2),plt.xlabel('Reprojection Error'),plt.title('2nd iteration'),plt.show(),

#%%
camInd = 1
ind2d = sba2.cameraIndices==camInd

plt.scatter(sba2.points2D[ind2d,0],-sba2.points2D[ind2d,1],6, r2[ind2d]),
plt.clim(0, 1)
plt.colorbar(),
plt.show()

# plt.scatter(obs2d_area[ind2d], r[ind2d], 6),
# plt.clim(0, 30)
# plt.colorbar(),
# plt.show()

#%%
from scipy.io import savemat

savemat('optCamArray_laserCalib_201125.mat',{'optCamArray':optCamArray})