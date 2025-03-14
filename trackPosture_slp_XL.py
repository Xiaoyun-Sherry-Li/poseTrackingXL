import numpy as np
import tensorflow as tf
import cv2
# from tensorflow.keras.models import load_model as tf_load
import matplotlib.pyplot as plt

import sys
sys.path.append("C:/Users/xl313/OneDrive/Documents/GitHub/poseTrackingXL/utils")
from load_matlab_data import loadmat_sbx
from slp_utils import posture_tracker
import scipy.io


''' UPDATE data params as appropriate'''
# cam params
cam_ids = ['blue_cam', 'green_cam', 'red_cam', 'yellow_cam'] # check the input order
im_w = 2200
im_h = 650

# video params
start_frame = 0 # in frames at 50fps # SHERRY: start at 0 min
nFrames = 5 # in frames at 50fps # SHERRY: take only 100 frames, which is 2 sec

''' UPDATE paths as needed '''
# videos
root_dir = "Z:/Sherry/acquisition/"
vid_root = f"{root_dir}AMB155_031025/"

# camera params
cam_param_dir = "Z:/Sherry/camera_calibration/"  
cam_param_file = "092124_camOptArrayDA_XL.mat" # UPDATE if-needed to match session
cam_params = scipy.io.loadmat("Z:/Sherry/camera_calibration/092124_camOptArrayDA_XL")['optCamArrayXL'] #['camParams']
# K format: array([[1852,    0,    0],
       #[   0, 1852,    0],
       #[ 952,  497,    1]])

# models
comNet = "Z:/Sherry/poseTrackingXL/training_files/SLP/models/240924_155057.single_instance.n=80/" 
postureNet = "Z:/Sherry/poseTrackingXL/training_files/SLP/models/240924_162008.single_instance.n=80/"
# faceNet = "Z:\Selmaan\DPK-transfer\j4-v4"

# to save
pred_date = "101724"
# save_file = f'{pred_date}_posture_2stage.npy' # python
save_file = f'{pred_date}_posture_2stage.mat' # matlab
save_path = f"{vid_root}{save_file}"

''' set up for this run '''
# set up GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

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

# if running face model, otherwise set to None
# face_model = tf_load(faceNet, custom_objects={'tf': tf})
face_model = None

''' track posture '''
obj = posture_tracker(all_readers, cam_params,
                        com_model=comNet,
                        posture_model=postureNet,
                        face_model=face_model)
results = obj.track_video(start_frame=start_frame,
                            nFrames=nFrames)
# results = obj.track_video_com(start_frame=start_frame,
#                                 nFrames=nFrames)

''' save file '''
# for python
# save_dict = {"results": results,
#             "camNames": cam_ids,
#             "session": vid_root,
#             "start_frame": start_frame,
#             "n_frames": nFrames,
#             "cam_params": cam_params
# }
# np.save(save_path, save_dict)

# for matlab 
scipy.io.savemat(save_path,{"posture_preds": results['posture_preds'], "posture_reproj": results['posture_rep_err'],
                     "posture_rawpreds": results['posture_rawpred'], "com_preds": results['com_preds'], "com_reproj": results['com_rep_err'],
                     "posture_conf":results['posture_conf'], "com_conf":results['com_conf'], #  "face_preds":results['face_preds'], "startTime": startTime,
                     "camNames": cam_ids, "session": vid_root, "nFrames": nFrames,
                     "camParams": cam_params})