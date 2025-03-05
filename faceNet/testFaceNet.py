import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model as tf_load

import os 
import sys
sys.path.append("..//utils/")
from slp_utils import crop_from_com
from triangulation_utils import unDistortPoints, camera_matrix, triangulate_confThresh_lowestErr
sys.path.append("..//camera_calibration/")
import pySBA
import mat73


''' cropping params '''
face_w3d=0.06 # scaling factor
face_crop_size=(128,128) # pixels
head_idx=np.asarray([7, 11])

''' load the training images and faceNet model '''
# set up GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

# Label3D training data
training_dir = '..//training_files/Label3D/'
training_files = []
for f in os.listdir(training_dir):
    if 'seed' in f:
        training_files.append(f)

# load the model
faceNet = "j4-il-v1.h5"
face_model = tf_load(faceNet, custom_objects={'tf': tf}, compile=True)


''' Functions '''
def formatData(data3D):
    nFrames = data3D.shape[0]
    nParts = data3D.shape[1]//3
    data3D = np.reshape(data3D, (nFrames, nParts, 3))
    return (data3D, nFrames, nParts)

def get_crop_info(matfile):
    # get the camera params for this file
    camParams = pySBA.convertParams(matfile['camParams'])
    nCams = camParams.shape[0]
    pt3d, nFrames, nParts = formatData(matfile['data_3D'])
    sba = pySBA.PySBA(camParams, np.NaN, np.NaN, np.NaN, np.NaN)
        
    # get the 3D distance from each camera for cropping scale
    head_COM = np.nanmean(pt3d[:, head_idx], axis=1)
    allCamScales = np.full((nFrames, nCams), np.NaN)
    allCentroids = np.full((nFrames, nCams, 2), np.NaN)
    for f in range(nFrames):
        this_COM = head_COM[f]
        allCentroids[f] = sba.project(np.tile(this_COM, (nCams, 1)), camParams)  # get reprojected centroid locations
        camDist = sba.rotate(np.tile(this_COM, (nCams, 1)), camParams[:, :3])  # rotate to camera coordinates
        camDist = camDist[:, 2] + camParams[:, 5]  # get z-axis distance ie along optical axis
        allCamScales[f] = camParams[:, 6] / camDist  # convert to focal length divided by distance
    
    return allCamScales, allCentroids


''' get the camera views, head centroids, and camera scales for each Label3D file '''
all_images = [] # list (len (n_cams,)) of arrays (w, h, n_frames)
all_scales = [] # (n_frames, n_cams)
all_centroids = [] # (n_frames, n_cams, 2)
print('\nloading training files...')
for fn in training_files:
    print(fn)
    file_path = f"{training_dir}{fn}"
    matfile = mat73.loadmat(file_path)
    scales, centroids = get_crop_info(matfile)
    images = []
    for data in matfile['videos']:
        images.append(data[0])
    all_images.append(images)
    all_scales.append(scales)
    all_centroids.append(centroids)
    
# reformat
all_scales = np.concatenate(all_scales, axis=0)
all_centroids = np.concatenate(all_centroids, axis=0)
n_cams = all_scales.shape[1]
all_cams = []
for c in range(n_cams):
    these_images = np.concatenate([i[c] for i in all_images], axis=3)
    all_cams.append(np.squeeze(these_images))
del all_images


''' crop and resize frames '''
n_frames = all_scales.shape[0]
face_images = np.zeros([n_frames, face_crop_size[1], face_crop_size[0], n_cams], dtype='uint8') # shape (total_frames, ds_h, ds_w)
for n_cam in range(n_cams):
    images = np.transpose(all_cams[n_cam], axes=[2, 0, 1]) 
    scales = all_scales[:, n_cam]
    centroids = all_centroids[:, n_cam]
    for f in range(n_frames):
        if np.isfinite(centroids[f].sum()):
            full_image = images[f]
            head_ctr = np.maximum(centroids[f], 0) # rough x-y head coords as in comNet
            head_ctr[0] = np.min([head_ctr[0], full_image.shape[1]])
            head_ctr[1] = np.min([head_ctr[1], full_image.shape[0]])
            half_width = np.nanmax([np.round(face_w3d * scales[f]), 15]) # minimum 31px image for head
            crop_img, _, _ = crop_from_com(full_image, head_ctr, half_width, face_crop_size)
            face_images[f, :, :, n_cam] = crop_img

# only keep frames where head is labeled in all views
label_idx = []
for f in range(n_frames):
    if np.isfinite(np.sum(all_centroids[f])):
        label_idx.append(f)
face_images = face_images[np.asarray(label_idx)]
n_labeled_frames = face_images.shape[0]


''' predict seed/no seed - score and view-specific weights '''
# # define new network with view-specific weights and make predictions
# print('\nreading and predicting...')
# weights_layer = [l for l in face_model.layers if l.name == 'view_weights'][0]
# jp_layer = [l for l in face_model.layers if l.name == 'joint_pred'][0]
# pred_model = tf.keras.Model(inputs=face_model.input, outputs=[jp_layer.output, weights_layer.output])
# facePreds = []
# for face_img in face_images:
#     thisPrediction = pred_model.predict_on_batch(face_img[None, :, :, :])
#     facePreds.append(thisPrediction.copy())


# ''' plot 10 random frames to check output '''
# example_idx = np.random.randint(n_labeled_frames, size=10)
# for ex in example_idx:
#     # get the weight and score
#     val = facePreds[ex]
#     weights = np.squeeze(val[1])
#     score = np.squeeze(val[0])

#     # plot the face and label with score
#     f, ax = plt.subplots(2, 2, figsize=(4, 4))
#     ax[0, 0].imshow(face_images[ex, :, :, 0], cmap='gray')
#     ax[0, 1].imshow(face_images[ex, :, :, 1], cmap='gray')
#     ax[1, 0].imshow(face_images[ex, :, :, 2], cmap='gray')
#     ax[1, 1].imshow(face_images[ex, :, :, 3], cmap='gray')
#     for i in range(2):
#         for j in range(2):
#             ax[i, j].set_xticks([])
#             ax[i, j].set_yticks([])

#     # label with weight and score
#     ax[0, 0].set_title(f'weight = {weights[0]:.1f}')
#     ax[0, 1].set_title(f'weight = {weights[1]:.1f}')
#     ax[1, 0].set_title(f'weight = {weights[2]:.1f}')
#     ax[1, 1].set_title(f'weight = {weights[3]:.1f}')
#     f.suptitle(f'score = {score:.3f}')
#     plt.show()


''' predict seed/no seed - score only '''
# define new network with view-specific weights and make predictions
print('\nreading and predicting...')
jp_layer = [l for l in face_model.layers if l.name == 'joint_pred'][0]
pred_model = tf.keras.Model(inputs=face_model.input, outputs=jp_layer.output)
facePreds = []
for face_img in face_images:
    thisPrediction = pred_model.predict_on_batch(face_img[None, :, :, :])
    facePreds.append(thisPrediction.copy())


''' plot 10 random frames to check output '''
example_idx = np.random.randint(n_labeled_frames, size=10)
for ex in example_idx:
    # get the weight and score
    val = facePreds[ex]
    score = np.squeeze(val)

    # plot the face and label with score
    f, ax = plt.subplots(2, 2, figsize=(4, 4))
    ax[0, 0].imshow(face_images[ex, :, :, 0], cmap='gray')
    ax[0, 1].imshow(face_images[ex, :, :, 1], cmap='gray')
    ax[1, 0].imshow(face_images[ex, :, :, 2], cmap='gray')
    ax[1, 1].imshow(face_images[ex, :, :, 3], cmap='gray')
    for i in range(2):
        for j in range(2):
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])

    # label with the score
    f.suptitle(f'score = {score:.3f}')
    plt.show()