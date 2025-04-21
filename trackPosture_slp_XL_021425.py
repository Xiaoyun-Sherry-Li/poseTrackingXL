#%%
import sys
sys.path.append("C:/Users/xl313/OneDrive/Documents/GitHub/poseTrackingXL/utils")
sys.path.append("C:/Users/xl313/OneDrive/Documents/GitHub/poseTrackingXL/faceNet")
import numpy as np
import tensorflow as tf
import cProfile
import pstats
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress warnings and info, show only errors

''' set up for this run '''
# set up GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

from tensorflow.keras.models import load_model as tf_load
import matplotlib.pyplot as plt
import cv2
import faceNetArchitectures
from load_matlab_data import loadmat_sbx
from slp_utils_XL import posture_tracker, create_slp_project, crop_from_com
import scipy.io
#%%
''' UPDATE data params as appropriate'''
# cam params
cam_ids = ['blue_cam', 'green_cam', 'red_cam', 'yellow_cam'] # check the input order
im_w = 2200
im_h = 650
# video params
start_frame = 0 # in frames at 50fps # (XL,SLV123_110824_wEphys, 5m - 1h35m)
nFrames = 1500  # in frames at 50fps # takes 1 min

''' UPDATE paths as needed '''
# videos
root_dir = "Z:/Sherry/acquisition/"
vid_root = f"{root_dir}LVN4_040725/"
# camera params
cam_params = loadmat_sbx("Z:/Sherry/poseTrackingXL/calibration_files/all_opt_arrays/102324_negated_camParams")['camParams_negateR'] #['camParams']

# to save
pred_date = "042025_test"
save_file_py = f'{pred_date}_posture.npy' # python
save_file_mat = f'{pred_date}_posture.mat' # matlab
save_path_py = f"{vid_root}{save_file_py}"
save_path_mat = f"{vid_root}{save_file_mat}"
#%%
# models
comNet = "Z:/Sherry/poseTrackingXL/training_files/SLP/models/032225250322_000322.single_instance.n=752"
postureNet = "Z:/Sherry/poseTrackingXL/training_files/SLP/models/032225250322_002442.single_instance.n=752"
faceNet = "C:/Users/xl313/OneDrive/Documents/GitHub/poseTrackingXL/faceNet/j5-xl-041925.keras"
# cocoNet = "C:/Users/xl313/OneDrive/Documents/GitHub/poseTrackingXL/faceNet/cocoNet-041725.keras"

# if running face model, otherwise set to None
joint_model = tf_load(faceNet, custom_objects={'tf': tf}, compile=True) # load the complete model
jp_layer = [l for l in joint_model.layers if l.name == 'joint_pred'][0] # extract out "joint_pred" layer from the model
face_model = tf.keras.Model(inputs=joint_model.input, outputs=jp_layer.output) # a new model that only output the "joint_pred" layer

# cocoNet_tmp = tf_load(cocoNet, custom_objects={'tf': tf}, compile=True) # load the complete model
# coco_layer = [l for l in cocoNet_tmp.layers if l.name == 'joint_pred'][0] # extract out "joint_pred" layer from the model
# cocoNet = tf.keras.Model(inputs=cocoNet_tmp.input, outputs=coco_layer.output)
# a new model that only output the "joint_pred" layer
# face_model = None
#%%
# define the video reader for each camera
all_readers = []
for i in range(len(cam_ids)):
    cam = cam_ids[i]
    print(cam)
    camPath = f"{vid_root}{cam}.avi"
    # define the video reader obj and settings
    api_id = cv2.CAP_FFMPEG
    reader = cv2.VideoCapture(camPath, api_id)
    if start_frame > 0:
        reader.set(cv2.CAP_PROP_FRAME_COUNT, start_frame)
    all_readers.append(reader)
#%%
''' track posture '''
obj = posture_tracker(all_readers, cam_params,
                        com_model=comNet,
                        posture_model=postureNet,
                        face_model=face_model)
                        # cocoNet = None)

results = obj.track_video(start_frame=start_frame,
                            nFrames=nFrames)
cProfile.run(
    "results = obj.track_video(start_frame=start_frame, nFrames=nFrames)",
    "profile_stats.prof")
stats = pstats.Stats("profile_stats.prof")
stats.strip_dirs()  # Optional, to clean up long directory paths in the output
stats.sort_stats("cumulative")  # Sort by cumulative time
stats.print_stats()

#%%
''' save file '''
# # for python
save_dict = {"results": results,
            "camNames": cam_ids,
            "session": vid_root,
            "start_frame": start_frame,
            "n_frames": nFrames,
            "cam_params": cam_params
}
np.save(save_path_py, save_dict)

# for matlab
results_struct = {
    "posture_preds": results['posture_preds'],
    "posture_reproj": results['posture_rep_err'],
    "posture_rawpreds": results['posture_rawpred'],
    "com_preds": results['com_preds'],
    "com_reproj": results['com_rep_err'],
    "posture_conf": results['posture_conf'],
    "com_conf": results['com_conf'],
    "face_preds": results['face_preds'],
    # "coco_preds": results['coco_preds'],  # Uncomment if needed
    "camNames": cam_ids,
    "session": vid_root,
    "nFrames": nFrames,
    "camParams": cam_params,
    # "rawPostures": sleap_raw_predicted_points_scale_back,  # Uncomment if needed
}
# Save the struct to a .mat file
scipy.io.savemat(save_path_mat, {"results": results_struct})