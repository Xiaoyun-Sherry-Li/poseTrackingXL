#%%
from dpk_utils import posture_tracker
from ffVideoReader import ffVideoReader
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model as tf_load
from scipy.io import loadmat, savemat
gpu = 1
ds_fac = 4
path = "/media/selmaan/Locker/Selmaan/Birds/SPP7/SPP7_220822_104636"
camParams = loadmat("./optCamArray_laserCalib_220621.mat")['optCamArray']
comNet = "/media/selmaan/Locker/Selmaan/DPK-transfer/comNet_Model_06.h5"
postureNet = "/media/selmaan/Locker/Selmaan/DPK-transfer/postureNet_Model_14.h5"
faceNet = "/media/selmaan/Locker/Selmaan/DPK-transfer/j4-v6"
fn_out = "/posture_2stage_6_14.mat"

stdCams = ['lBack', 'lFront', 'lTop', 'rBack', 'rFront', 'rTop']
startTime = 0 #in seconds
nFrames = 500*60*60 #in frames at 60fps

allReaders = []
for i in range(len(stdCams)):
    cam = stdCams[i]
    print(cam)
    if cam == 'lTop' or cam == 'rTop':
        h=1696
    else:
        h=1408
    # camPath = path+"/videos/"+cam+'/0.avi'
    camPath = path + "/" + cam + ".avi"
    allReaders.append(ffVideoReader(camPath, w=2816, h=h, s=startTime))

gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[gpu],'GPU')
tf.config.experimental.set_memory_growth(gpus[gpu], True)

face_model = tf_load(faceNet, custom_objects={'tf': tf}, compile=False)
obj = posture_tracker(allReaders, camParams, com_model=comNet, posture_model=postureNet, face_model=face_model)
# obj = posture_tracker(allReaders, camParams, com_model=comNet, posture_model=postureNet)
results = obj.track_video(nFrames)

# kalman filter
from pkm_utils import kf_outer
repThresh = 12
preds = results['posture_preds']
reproj = results['posture_rep_err']
smPos, smVel = kf_outer(preds, reproj, repThresh)

savemat(path+fn_out, {"posture_preds": results['posture_preds'], "posture_reproj": results['posture_rep_err'],
                     "posture_rawpreds": results['posture_rawpred'], "com_preds": results['com_preds'], "com_reproj": results['com_rep_err'],
                     "posture_conf":results['posture_conf'], "com_conf":results['com_conf'], "face_preds":results['face_preds'],
                     "camNames": stdCams, "session": path, "startTime": startTime, "nFrames": nFrames,
                     "camParams": camParams, "smPos": smPos, "smVel": smVel})