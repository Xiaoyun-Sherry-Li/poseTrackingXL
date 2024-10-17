#%%
from dpk_utils import posture_tracker
from ffVideoReader import ffVideoReader
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model as tf_load
from scipy.io import loadmat, savemat
path = "Z:\Selmaan\Birds\IND88\IND88_210420_105623"
camParams = loadmat("./optCamArray_laserCalib_210505_old.mat")['optCamArray']
comNet = "Z:\Selmaan\DPK-transfer\comNet_Model_04.h5"
postureNet = "Z:\Selmaan\DPK-transfer\postureNet_Model_12c.h5"
faceNet = "Z:\Selmaan\DPK-transfer\j4-v4"
fn_out = '\\posture_2stage_4_12c.mat'

stdCams = ['lBack', 'lFront', 'lTop', 'rBack', 'rFront', 'rTop']
startTime = 0 #19*60 + 45 #in seconds
nFrames = 300*60*60 #in frames at 60fps

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

allReaders = []
for i in range(len(stdCams)):
    cam = stdCams[i]
    print(cam)
    if cam == 'lTop' or cam == 'rTop':
        h=1696
    else:
        h=1408
    camPath = path+"\\"+cam+".avi"
    allReaders.append(ffVideoReader(camPath, w=2816, h=h, s=startTime))

face_model = tf_load(faceNet, custom_objects={'tf': tf})
obj = posture_tracker(allReaders, camParams, com_model=comNet, posture_model=postureNet, face_model=face_model)
results = obj.track_video(nFrames)

savemat(path+fn_out,{"posture_preds": results['posture_preds'], "posture_reproj": results['posture_rep_err'],
                     "posture_rawpreds": results['posture_rawpred'], "com_preds": results['com_preds'], "com_reproj": results['com_rep_err'],
                     "posture_conf":results['posture_conf'], "com_conf":results['com_conf'], "face_preds":results['face_preds'],
                     "camNames": stdCams, "session": path, "startTime": startTime, "nFrames": nFrames,
                     "camParams": camParams})