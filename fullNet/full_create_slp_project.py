'''
Create a SLP project using labeled frames from from Label3D - full model
--------------------
Takes labeled training data from Label3D and
packages it into a SLP project to train the full_net model.

Reformat 3D points from Label3D into an array of 2D points
- takes in a matfile from Label3D --> extracts video frames and 3D points
- reshapes the 3D points to (n_frames, n_nodes, 3)
- reprojects onto each camera view to get an array of shape (n_frames, n_cams, n_nodes, 2)

Then, pass through create_slp_project() to make the SLP project file.

These data should be used with the top-down SLP strategy.
With this method, SLP first finds the bird in downsampled video frames
and crops around it. Then, it learns the detailed keypoint locations in
these cropped, full-resolution video frames.

This is a built-in method that seems to closely match Selmaan's
com_net + posture_net approach.
'''
import numpy as np
import matplotlib.pyplot as plt
import csv
import mat73

import os 
import sys
sys.path.append("../utils/")
from slp_utils import create_slp_project
sys.path.append("../camera_calibration/")
import pySBA


''' Functions '''
def formatData(data3D):
    nFrames = data3D.shape[0]
    nParts = data3D.shape[1]//3
    data3D = np.reshape(data3D, (nFrames, nParts, 3))
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


''' Set paths '''
skeleton_file = './/full_skeleton_IL.csv'

# to save SLP project
slp_project_dir = '..//training_files/SLP/'
proj_date = input("input today's date (YYMMDD): ")
slp_project_file = f'{proj_date}_full_net.slp'
slp_project_path = f'{slp_project_dir}{slp_project_file}'

# Label3D training data
training_dir = '..//training_files/Label3D/'
training_files = []
for f in os.listdir(training_dir):
    training_files.append(f)

''' Get 3D points and reformat into an array of 2D points '''
# get the 2D points and frames for each Label3D file
all_labels = [] # (n_frames, n_cams, n_nodes, 2)
all_images = [] 
for fn in training_files:
    print(fn)
    file_path = f"{training_dir}{fn}"
    matfile = mat73.loadmat(file_path)
    labels = projectData(matfile)
    images = []
    for data in matfile['videos']:
        images.append(data[0])
    all_labels.append(labels)
    all_images.append(images)
    
# reformat
all_labels = np.concatenate(all_labels, axis=0)
n_cams = all_labels.shape[1]
all_cams = [] # list (len (n_cams,)) of arrays (w, h, n_frames)
for c in range(n_cams):
    these_images = np.concatenate([i[c] for i in all_images], axis=2)
    all_cams.append(np.squeeze(these_images))
del all_images

''' plot a frame to check output '''
ex_frame = 2

# fig params
f, ax = plt.subplots(n_cams//2, 2)
colors = ['xkcd:scarlet', 'xkcd:cobalt blue', 'xkcd:saffron']
node_idx = np.asarray([0, 4, 14]) # topBeak, centerBack, rightFoot - could change
x_idx = np.full(node_idx.shape[0], 0)
y_idx = np.full(node_idx.shape[0], 1)

# plot for each camera
for n_cam in range(n_cams):
    # get subplot index
    if n_cam < n_cams//2:
        r = n_cam
        c = 0
    else:
        r = n_cam - n_cams//2
        c = 1
    
    # label keypoints
    image = np.squeeze(images[n_cam])
    ax[r, c].imshow(image[:, :, ex_frame], cmap='gray')
    ax[r, c].scatter(labels[ex_frame, n_cam, node_idx, x_idx], 
                     labels[ex_frame, n_cam, node_idx, y_idx],
                     c=colors, marker='*', s=15)
plt.show()

''' concatenate across camera views to pass into SLP '''
label_data = [] 
image_data = [] 
for n_cam in range(n_cams):
    images = np.transpose(all_cams[n_cam], axes=[2, 0, 1]) 
    labels = all_labels[:, n_cam]
    n_frames = labels.shape[0]
    for f in range(n_frames):
        if np.isfinite(labels[f].sum()):
            label_data.append(labels[f])
            image_data.append(images[f])
label_data = np.asarray(label_data) # shape (total_frames, n_nodes, 2)
image_data = np.asarray(image_data) # shape (total_frames, ds_w, ds_h)

create_slp_project(images=image_data[:,:,:,np.newaxis], 
                   skeleton_file=skeleton_file,
                   keypoints=label_data,
                   slp_labels_file=slp_project_path)