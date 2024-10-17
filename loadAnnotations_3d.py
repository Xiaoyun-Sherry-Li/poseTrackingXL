from supervisely_utils import get_pt_dict
import glob
import numpy as np

ptDict = get_pt_dict()

#allDir = glob.glob('/Users/selmaan/PycharmProjects/bc_supervisely/Chickadee-1/*/')
d1 = '/Users/selmaan/PycharmProjects/bc_supervisely/Chickadee-1/IND32 200226 SideImages/'
d2 = '/Users/selmaan/PycharmProjects/bc_supervisely/Chickadee-1/IND32 200226 TopImages/'
allAnn = glob.glob(d1 + 'ann/*.json') + glob.glob(d2 + 'ann/*.json')

imCameras = []
imNumbers = []
for ann in allAnn:
    numStart = ann.rfind('_')+1
    numStop = ann.rfind('.bmp')
    imNumbers.append(ann[numStart:numStop])
    camStart = ann.rfind('_', 1, numStart-1)+1
    imCameras.append(ann[camStart:numStart-1])

#allCams = list(set(imCameras))
#allNumbers = list(set(imNumbers))
allCams = np.unique(imCameras)
allNumbers = np.unique(imNumbers)

#%%

cameraArray = np.load('shared_cam_array.npy')

#%%
# Prepare data into format with camera and point index for each observation
nIms = len(allNumbers)
nCams = len(allCams)
camera_ind = []
points_2d = []
point_node = []
point_2dind3d = []

indModulo = len(ptDict)
for nIm, ann in enumerate(allAnn):
    with open(ann, 'r') as read_file:
        data = json.load(read_file)
        read_file.close()
    thisCam = [i for i,e in enumerate(allCams) if e==imCameras[nIm]][0]
    thisIm = [i for i, e in enumerate(allNumbers) if e == imNumbers[nIm]][0]
    for nObj, obj in enumerate(data['objects']):
        for k in obj['nodes']:
            ptID = ptDict[k]['id']
            ptLoc = obj['nodes'][k]['loc']
            camera_ind.append(thisCam)
            point_node.append(ptID)
            points_2d.append(ptLoc)
            point_2dind3d.append(indModulo * thisIm + ptID)

camera_ind = np.array(camera_ind)
point_node = np.array(point_node)
point_2dind3d = np.array(point_2dind3d)
points_2d = np.array(points_2d) - [1400, 700] # 'zero' point locations at image center

# Given 2 annotators (this is for temporary example only!), require at least 4 observations of a 3d points
# u, c = np.unique(point_2dind3d, return_counts=True)
# validU = u[c>3]
# validInd = np.isin(point_2dind3d, validU)
# camera_ind = camera_ind[validInd]
# point_node = point_node[validInd]
# points_2d = points_2d[validInd]
# point_2dind3d = point_2dind3d[validInd]
# NOTE ABOVE CODE REQUIRES INDEX CONVERSION FOR 2dind3d TO WORK, may not be necessary anyway?

#%%
import matplotlib.pyplot as plt
import pySBA

points_3d = np.full((indModulo * len(allNumbers), 3), 0.0)
sba = pySBA.PySBA(cameraArray, points_3d, points_2d, camera_ind, point_2dind3d)

n = 9 * sba.cameraArray.shape[0] + 3 * sba.points3D.shape[0]
m = 2 * points_2d.shape[0]
print("n_cameras: {}".format(nCams))
print("n_points: {}".format(points_3d.shape[0]))
print("Total number of parameters: {}".format(n))
print("Total number of residuals: {}".format(m))

f0 = sba.getResiduals()
sba.bundleAdjust_nocam()
x1 = sba.points3D.copy()
f1 = sba.getResiduals()
tmp1 = sba.bundleAdjust_sharedcam()
x2 = sba.points3D.copy()
f2 = sba.getResiduals()
# tmp2 = sba.bundleAdjust()
# x3 = sba.points3D.copy()
# f3 = sba.getResiduals()

plt.plot(21*x1,'.'), plt.title('No Cam Residual: %d' %np.median(np.abs(f1))), plt.show()
plt.plot(21*x2,'.'), plt.title('Shared Cam Residual: %d' %np.median(np.abs(f2))), plt.show()

# camInd = []
# numInd = []
# for cam in allCams:
#     camInd.append([i for i, x in enumerate(imCameras) if x==cam])
# for num in allNumbers:
#     numInd.append([i for i, x in enumerate(imNumbers) if x==num])

# nIms = len(allNumbers)
# nCams = len(allCams)
# xLoc = np.full((len(ptDict), nIms, nCams, 2), np.NaN)
# yLoc = np.full((len(ptDict), nIms, nCams, 2), np.NaN)
# for nIm, ann in enumerate(allAnn):
#     with open(ann, 'r') as read_file:
#         data = json.load(read_file)
#         read_file.close()
#     thisCam = [i for i,e in enumerate(allCams) if e==imCameras[nIm]][0]
#     thisIm = [i for i,e in enumerate(allNumbers) if e==imNumbers[nIm]][0]
#     for nObj, obj in enumerate(data['objects']):
#         for k in obj['nodes']:
#             ptID = ptDict[k]['id']
#             ptLoc = obj['nodes'][k]['loc']
#             xLoc[ptID, thisIm, thisCam, nObj] = ptLoc[0]
#             yLoc[ptID, thisIm, thisCam, nObj] = ptLoc[1]


#%%
# import matplotlib.pyplot as plt
# xM = np.nanmean(xLoc, axis=3)
# yM = np.nanmean(yLoc, axis=3)
# plt.plot(np.mean(np.sum(~np.isnan(xM), axis=2), axis=1), '.'),
# plt.plot(np.median(np.sum(~np.isnan(xM), axis=2), axis=1), '.'),
# plt.plot(np.percentile(np.sum(~np.isnan(xM), axis=2), 10, axis=1), '.'),
# plt.show()



#%%
import scipy.spatial.distance as dst
import sklearn.decomposition as skd

nPts = len(ptDict)
nFrames = len(allNumbers)
pointLocs = sba.points3D.reshape((nPts, nFrames, 3), order='F')
allDist = np.full((nPts*(nPts-1)//2, nFrames), np.NaN)
for nFrame in range(nFrames):
    allDist[:,nFrame] = dst.pdist(pointLocs[:,nFrame,:])
normDist = allDist.T / np.std(allDist.T, axis=0)
pca = skd.PCA()
p = pca.fit_transform(normDist)

plt.plot(p[:,0],'.'), plt.show()