import numpy as np
import tensorflow as tf
import cv2
# from tensorflow.keras.models import load_model as tf_load

import sys
sys.path.append(".//utils/")
from load_matlab_data import loadmat_sbx
from slp_utils import posture_tracker

''' UPDATE data params as appropriate'''
# cam params
cam_ids = ['red_cam', 'yellow_cam', 'green_cam', 'blue_cam']
im_w = 1896
im_h = 640

# video params
start_frame = 0 # in frames at 50fps
nFrames = ((2*60 + 50)*60 + 9)*50 # in frames at 50fps

''' UPDATE paths as needed '''
# videos
root_dir = "Z:/Isabel/data/"
vid_root = f"{root_dir}behavior/SLV124/SLV124_240906/"

# camera params
cam_param_dir = ".//calibration_files/all_opt_arrays/"
cam_param_file = "240903_aligned_opt_cam_array.mat" # UPDATE if-needed to match session
cam_params = loadmat_sbx(f"{cam_param_dir}{cam_param_file}")['optCamArray']

# models
comNet = ".//comNet/com_models/com_240909_142349.single_instance.n=104/"
postureNet = ".//postureNet/posture_models/posture_240909_183000.single_instance.n=104/"
# faceNet = "Z:\Selmaan\DPK-transfer\j4-v4"

# to save
pred_date = input("input today's date (YYMMDD): ")
save_file = f'{pred_date}_posture_2stage.npy' # python
# save_file = f'{pred_date}_posture_2stage.mat' # matlab
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
save_dict = {"results": results,
            "camNames": cam_ids,
            "session": vid_root,
            "start_frame": start_frame,
            "n_frames": nFrames,
            "cam_params": cam_params
}
np.save(save_path, save_dict)

# for matlab 
# savemat(save_path,{"posture_preds": results['posture_preds'], "posture_reproj": results['posture_rep_err'],
#                      "posture_rawpreds": results['posture_rawpred'], "com_preds": results['com_preds'], "com_reproj": results['com_rep_err'],
#                      "posture_conf":results['posture_conf'], "com_conf":results['com_conf'], "face_preds":results['face_preds'],
#                      "camNames": cam_ids, "session": vid_root, "startTime": startTime, "nFrames": nFrames,
#                      "camParams": cam_params})