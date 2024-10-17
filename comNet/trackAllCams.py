#%%
from dpk_utils import com_tracker, verify_video_metadata
from ffVideoReader import ffVideoReader
comNet = 'C:\\Users\\Selmaan\\PycharmProjects\\supervisely_dpk_training\\comNet\\comNet_Model.h5'

stdCams = ['lBack', 'lFront', 'lTop', 'rBack', 'rFront', 'rTop']
startTime = 0*60 #in seconds
nFrames = 120*60*60 #in frames at 60fps
path = "D:\SLV151_200730_131948"
chk, dat = verify_video_metadata(path)

allPreds = []
for i in range(len(stdCams)):
    cam = stdCams[i]
    print(cam)
    if cam == 'lTop' or cam == 'rTop':
        h=1696
    else:
        h=1408
    camPath = path+"\\"+cam+'.avi'
    obj = com_tracker(ffVideoReader(camPath, w=2816, h=h, queueSize=500, s=startTime), com_model=comNet)
    thesePreds = obj.track_video(nFrames=nFrames)
    allPreds.append(thesePreds)

#%%
import pySBA
import numpy as np
from scipy.io import loadmat, savemat

camParams = loadmat("./optCamArray_laserCalib_200729.mat")['optCamArray']
points2d = np.stack(allPreds,axis=2)[:,:,:,:2]
points2d = np.reshape(points2d, (-1,2),order='C')
cameraIndices = np.tile(np.arange(6),nFrames*3)
pointIndices = np.tile(np.arange(nFrames*3).reshape((-1,1)),(1,6)).ravel()

sba = pySBA.PySBA(camParams, np.zeros((nFrames*3,3)), points2d, cameraIndices, pointIndices)
sba.bundleAdjust_nocam()

#%%
import matplotlib.pyplot as plt

R = np.array([0.9960,0.0485,0.0751, -0.0332, 0.9807, -0.1925, -0.0830, 0.1892, 0.9784]).reshape((3,3))
t = np.array([-0.0099, 0.0318,-0.4038])

# reprojection error as automatic error metric
r = sba.project(sba.points3D[sba.point2DIndices], sba.cameraArray[sba.cameraIndices]) - sba.points2D
r = np.sqrt(np.sum(r**2, axis=1))
r = r.reshape((-1,3,6))
# r_frame = r.mean(axis=2)
r_frame = np.median(r, axis=2)

# inter-frame speeds as automatic error metric
p3 = (np.dot(sba.points3D.copy(), R.T) + t)/0.72
p3 = p3.reshape((-1,3,3))
v3 = p3[1:]-p3[:-1]
sp = np.sqrt(np.sum(v3**2, axis=1))

#
for i in range(3):
    ind = r_frame[:,i]<25
    print('excluding {:.4f} of data for bodypart {}'.format(1-np.mean(ind), i))
    thesePts = p3[ind,i]
    plt.scatter(thesePts[:,0],thesePts[:,1],1,thesePts[:,2]),plt.clim(0,0.15),plt.show(),

hVec =  p3[:,0]-p3[:,1]

#%%

savemat(path+"\\allTracks.mat",{"predictions": allPreds, "camNames": stdCams, "session": path,
                       "startTime": startTime, "nFrames": nFrames, "reprojError": r, "points3d": p3})

# points2d = np.reshape(allPreds[:,:,:,:2], (-1,2))

#%%
import numpy as np
from scipy.io import loadmat
from triangulation_utils import unDistortPoints, triangulate, camera_matrix

cam_params = loadmat('D:\SLV151_200730_131948\camParams_dannce.mat')['camParams']
f = loadmat('D:\SLV151_200730_131948\\allTracks.mat')
allPreds = np.transpose(f['predictions'], axes=[1,2,0,3])
R = np.array([0.9960,0.0485,0.0751, -0.0332, 0.9807, -0.1925, -0.0830, 0.1892, 0.9784]).reshape((3,3))
t = np.array([-0.0099, 0.0318,-0.4038])


for nCam in range(allPreds.shape[2]):
    thisParams = cam_params[nCam][0]
    K = thisParams['K'][0][0]
    RDistort = thisParams['RDistort'][0][0]
    thisPts = allPreds[:,:,nCam,:2]
    thisPts = np.reshape(thisPts,(-1,2))
    newPts = unDistortPoints(thisPts, K, RDistort)
    allPreds[:,:,nCam,:2] = np.reshape(newPts, (-1,3,2))

nPts = allPreds.shape[0]
nParts = allPreds.shape[1]
nCams = allPreds.shape[2]
nCombo = np.math.factorial(nCams)//np.math.factorial(nCams-2)//2
allP3 = np.full((nPts, 3, nCombo, nParts), np.NaN)
for nPart in range(nParts):
    nCombo = -1
    for iCam in range(nCams):
        for jCam in range(iCam+1, nCams):
            nCombo += 1
            print(nCombo)
            pts1 = allPreds[:,nPart,iCam,:2]
            pts2 = allPreds[:,nPart,jCam,:2]
            iParam = cam_params[iCam][0]
            jParam = cam_params[jCam][0]
            iMat = camera_matrix(iParam['K'][0][0], iParam['r'][0][0], iParam['t'][0][0])
            jMat = camera_matrix(jParam['K'][0][0], jParam['r'][0][0], jParam['t'][0][0])
            p3 = triangulate(pts1, pts2, iMat, jMat)
            allP3[:,:,nCombo,nPart] = (np.dot(R, p3).T + t)/0.72

med3 = np.transpose(np.median(allP3, axis=2), axes=[0, 2, 1])

#%%

#%%
from dpk_utils import com_tracker, verify_video_metadata
from ffVideoReader import ffVideoReader
import numpy as np
from scipy.io import savemat, loadmat
from triangulation_utils import unDistortPoints, triangulate, camera_matrix
netToUse = 'C:\\Users\\Selmaan\\PycharmProjects\\supervisely_dpk_training\\comNet\\comNet_Model.h5'
fn = "\\COMs_from_DPK.mat"
# netToUse = 'C:\\Users\\Selmaan\\PycharmProjects\\supervisely_dpk_training\\fullNet\\fullNet_Model.h5'
# fn = "\\test_posture.mat"
ds_fac = 4
path = "Z:\Selmaan\dannce-data\SLV151_200730_131948"
cam_params = loadmat('Z:\Selmaan\dannce-data\SLV151_200730_131948\\20200814_143127_Label3D_dannce.mat')['params']

stdCams = ['lBack', 'lFront', 'lTop', 'rBack', 'rFront', 'rTop']
startTime = 1*60 #in seconds
nFrames = 3*60*60 #in frames at 60fps
chk, dat = verify_video_metadata(path)

R = np.array([0.9960,0.0485,0.0751, -0.0332, 0.9807, -0.1925, -0.0830, 0.1892, 0.9784]).reshape((3,3))
t = np.array([-0.0099, 0.0318,-0.4038])

allPreds = []
for i in range(len(stdCams)):
    cam = stdCams[i]
    print(cam)
    if cam == 'lTop' or cam == 'rTop':
        h=1696
    else:
        h=1408
    camPath = path+"\\videos\\"+cam+'\\0.avi'
    obj = com_tracker(ffVideoReader(camPath, w=2816, h=h, s=startTime),
                      ds_fac=ds_fac, com_model=netToUse)
    thesePreds = obj.track_video(nFrames=nFrames)
    allPreds.append(thesePreds)

allPreds = np.stack(allPreds,axis=2)
nPts = allPreds.shape[0]
nParts = allPreds.shape[1]
nCams = allPreds.shape[2]

for nCam in range(nCams):
    thisParams = cam_params[nCam][0]
    K = thisParams['K'][0][0]
    RDistort = thisParams['RDistort'][0][0]
    thisPts = allPreds[:,:,nCam,:2]
    thisPts = np.reshape(thisPts,(-1,2))
    newPts = unDistortPoints(thisPts, K, RDistort)
    allPreds[:,:,nCam,:2] = np.reshape(newPts, (nPts,nParts,2))

nCombos = np.math.factorial(nCams)//np.math.factorial(nCams-2)//2
allP3 = np.full((nPts, 3, nCombos, nParts), np.NaN)
for nPart in range(nParts):
    nCombo = -1
    for iCam in range(nCams):
        for jCam in range(iCam+1, nCams):
            nCombo += 1
            print(nCombo)
            pts1 = allPreds[:,nPart,iCam,:2]
            pts2 = allPreds[:,nPart,jCam,:2]
            iParam = cam_params[iCam][0]
            jParam = cam_params[jCam][0]
            iMat = camera_matrix(iParam['K'][0][0], iParam['r'][0][0], iParam['t'][0][0])
            jMat = camera_matrix(jParam['K'][0][0], jParam['r'][0][0], jParam['t'][0][0])
            p3 = triangulate(pts1, pts2, iMat, jMat)
            allP3[:,:,nCombo,nPart] = p3.T
            # allP3[:,:,nCombo,nPart] = (np.dot(R, p3).T + t)/0.72

med3 = np.transpose(np.median(allP3, axis=2), axes=[0, 2, 1])

savemat(path+fn,{"predictions": allPreds, "camNames": stdCams, "session": path,
                       "startTime": startTime, "nFrames": nFrames, "points3d": med3})