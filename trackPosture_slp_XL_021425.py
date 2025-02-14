#%%
import sys
sys.path.append("C:/Users/xl313/OneDrive/Documents/GitHub/poseTrackingXL/utils")
sys.path.append("C:/Users/xl313/OneDrive/Documents/GitHub/bird_pose_tracking/faceNet")
import numpy as np
import tensorflow as tf

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
start_frame = 0 # in frames at 50fps # (XL, 010825: an exampler coconut caching/eating + drinking water snippet: 11:50 - 13:00 min in SLV123_110824_wEphys) 
nFrames = 90000 # in frames at 50fps # takes 0.5 hour = 1 * 30 * 60 * 50 

''' UPDATE paths as needed '''
# videos
root_dir = "Z:/Sherry/poseTrackingXL/training_files/raw_acquisition_copy/"
vid_root = f"{root_dir}RBY52_012025/"
# camera params
cam_params = loadmat_sbx("Z:/Sherry/poseTrackingXL/calibration_files/all_opt_arrays/102324_negated_camParams")['camParams_negateR'] #['camParams']

# to save
pred_date = "021325"
save_file = f'{pred_date}_posture_2stage.npy' # python
# save_file = f'{pred_date}_posture_2stage_faceNet.mat' # matlab
save_path = f"{vid_root}{save_file}"
#%%
# models
comNet = "Z:/Sherry/poseTrackingXL/training_files/SLP/models/010725_com250107_235615.single_instance.n=460" 
postureNet = "Z:/Sherry/poseTrackingXL/training_files/SLP/models/010825_postureNet250108_164045.single_instance.n=460"
faceNet = "C:/Users/xl313/OneDrive/Documents/GitHub/bird_pose_tracking/faceNet/j4-xl-v1.h5"
# if running face model, otherwise set to None
# joint_model = tf_load(faceNet, custom_objects={'tf': tf}, compile=True) # load the complete model
# jp_layer = [l for l in joint_model.layers if l.name == 'joint_pred'][0] # extract out "joint_pred" layer from the model
# face_model = tf.keras.Model(inputs=joint_model.input, outputs=jp_layer.output) # a new model that only output the "joint_pred" layer
face_model = None
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
results = obj.track_video(start_frame=start_frame,
                            nFrames=nFrames)
#%%
''' save file '''
# for python
save_dict = {"results": results,
            "camNames": cam_ids,
            "session": vid_root,
            "start_frame": start_frame,
            "n_frames": nFrames,
            "cam_params": cam_params
}
np.save(save_path, save_dict)
#%%
# for matlab 
scipy.io.savemat(save_path,{"posture_preds": results['posture_preds'], "posture_reproj": results['posture_rep_err'],
                     "posture_rawpreds": results['posture_rawpred'], "com_preds": results['com_preds'], "com_reproj": results['com_rep_err'],
                     "posture_conf":results['posture_conf'], "com_conf":results['com_conf'], #  "face_preds":results['face_preds'], "startTime": startTime,
                     "camNames": cam_ids, "session": vid_root, "nFrames": nFrames,
                     "camParams": cam_params, }) # "rawPostures":sleap_raw_predicted_points_scale_back