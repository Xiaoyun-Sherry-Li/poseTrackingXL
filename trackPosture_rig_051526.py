#%%
import sys
sys.path.append('C:/Users/User/Documents/GitHub/poseTrackingXL')
sys.path.append('C:/Users/User/Documents/GitHub/poseTrackingXL/utils')
sys.path.append('C:/Users/User/Documents/GitHub/poseTrackingXL/faceNet')
import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress warnings and info, show only errors
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # forcing tensorflow to use GPU
import cv2

''' set up for this run '''
# set up GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

from tensorflow.keras.models import load_model as tf_load
import faceNetArchitectures
from load_matlab_data import loadmat_sbx
import mat73
import pySBA
from slp_utils_XL import posture_tracker, create_slp_project, crop_from_com
import scipy.io

#%%
''' UPDATE data params as appropriate'''
nFrames = 194 * 60 * 50  # in frames at 50fps # takes 1 min total 178mins
# videos
root_dir = "Z:/Sherry/acquisition/"
vid_root = f"{root_dir}RBY97_050626/"
# to save
pred_date = "051526"
# video params
start_frame = 0
# cam params
cam_ids = ['blue_cam', 'green_cam', 'red_cam', 'yellow_cam'] # check the input order
im_w = 2200
im_h = 650
# camera params
cam_params = loadmat_sbx("Z:/Sherry/poseTrackingXL/calibration_files/all_opt_arrays/102324_negated_camParams")['camParams_negateR'] #['camParams']
#camParamCells = loadmat_sbx("Z:/Sherry/poseTrackingXL/calibration_files/all_opt_arrays/031926_calibration_reformatted.mat")['allParams'] #['camParams']
#data = mat73.loadmat("Z:/Sherry/poseTrackingXL/calibration_files/all_opt_arrays/031926_calibration_reformatted.mat")
#camParamCells = data['allParams']
#cam_params = pySBA.convertParams(camParamCells)
save_file_py = f'raw_{pred_date}.npy' # python
save_file_mat = f'raw_{pred_date}.mat' # matlab
save_path_py = f"{vid_root}{save_file_py}"
save_path_mat = f"{vid_root}{save_file_mat}"

#%%
# models
# working model
# comNet = "Z:/Sherry/poseTrackingXL/training_files/SLP/models/comNet250430_222637.single_instance.n=1684"
# postureNet = "Z:/Sherry/poseTrackingXL/training_files/SLP/models/posture250430_230225.single_instance.n=1684"
# IL + XL simple combined model
# comNet = "Z:/Sherry/poseTrackingXL/training_files/SLP/models/250814_161111.single_instance.n=3756"
# postureNet = "Z:/Sherry/poseTrackingXL/training_files/SLP/models/250815_002509.single_instance.n=3756"
# IL foundation + XL model
# postureNet = "Z:/Sherry/poseTrackingXL/training_files/SLP/models/260112_222537.single_instance.n=1684"
# RH + XL model
# comNet = "Z:/Roman/poseTracking_RH/training_files/SLP/models/251013_145719.single_instance.n=2804"
# postureNet = "Z:/Roman/poseTracking_RH/training_files/SLP/models/251013_172819.single_instance.n=2800"
# postureNet = "Z:/Sherry/poseTrackingXL/training_files/SLP/models/260121_133117.single_instance.n=1684" # with resume training based on IL data

# successfully trained XL + IL model
comNet = "Z:/Sherry/poseTrackingXL/training_files/SLP/models/ILbaseCom260122_095351.single_instance.n=1684"
# postureNet = "Z:/Sherry/poseTrackingXL/training_files/SLP/models/XLILbase260121_184939.single_instance.n=1684"
postureNet = "Z:/Sherry/poseTrackingXL/training_files/SLP/models/042126_2536FramesTotal260421_194809.single_instance.n=344"
# postureNet = "Z:/Sherry/poseTrackingXL/training_files/SLP/models/ROS100260215_133136.single_instance.n=400" # building on an existing model
faceNet = "C:/Users/User/Documents/GitHub/poseTrackingXL/faceNet/j5-xl-041925.keras"

# if running face model, otherwise set to None
with tf.device('/GPU:0'):  # Explicitly place model on GPU # added by sherry 072725
    joint_model = tf_load(faceNet, custom_objects={'tf': tf}, compile=True) # load the complete model
    jp_layer = [l for l in joint_model.layers if l.name == 'joint_pred'][0] # extract out "joint_pred" layer from the model
    face_model = tf.keras.Model(inputs=joint_model.input, outputs=jp_layer.output) # a new model that only output the "joint_pred" layer
    # Verify model is on GPU
    print("Model device:", face_model.weights[0].device)

#%%
# define the video reader for each camera
all_readers = []
for i in range(len(cam_ids)):
    cam = cam_ids[i]
    print(cam)
    camPath = f"{vid_root}{cam}.avi"
    # define the video reader obj and settings
    api_id = cv2.CAP_FFMPEG
    reader = cv2.VideoCapture(camPath, api_id) # 063025: sherry commented it out to debug failed to read frame in a session
    # reader = cv2.VideoCapture(camPath)
    if start_frame > 0:
        ## reader.set(cv2.CAP_PROP_FRAME_COUNT, start_frame) # 063025: sherry commented it out to debug failed to read frame in a session
        reader.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    all_readers.append(reader)

#%%
''' track posture '''
obj = posture_tracker(all_readers, cam_params,
                        ds_fac=4,
                        w3d=80,
                        crop_size=(320,320),
                        com_model=comNet,
                        posture_model=postureNet,
                        face_model = face_model)
                        # cocoNet = None)

results = obj.track_video(start_frame=start_frame,
                            nFrames=nFrames)

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