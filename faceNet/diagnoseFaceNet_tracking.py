#%%
# Diagnostic for the faceNet step of the TRACKING pipeline.
#
# Shows exactly what reaches the model during a real tracking run - the 4 RGB
# head crops per frame - alongside each view's 3-class softmax, its consensus
# contribution (w, f), and the final frame score. Use this when the tracked
# face predictions look worse than they did in testFaceNet_072026.ipynb.
#
# The notebook crops from Label3D ground-truth 3D points; the tracker crops from
# live comNet/postureNet predictions. If the crops here look mis-framed compared
# to the notebook, that framing difference is the problem, not the classifier.
#
# COMPARE_CENTERING=True runs the face crop BOTH ways on the same frames:
#   comNet   - best_com[com_head_ind]        (posture_tracker default)
#   posture  - mean of best_posture[parts]   (how the training crops were made,
#                                             pos_center_ind=[0,1] in
#                                             faceNet/label_training_data_XL.ipynb)
import sys
import os
# These must all be set before tensorflow initialises - `import tensorflow` reads
# TF_CPP_MIN_LOG_LEVEL, and the CUDA driver reads CUDA_CACHE_MAXSIZE at context
# creation. Setting them afterwards (as this script previously did) is a no-op.
#
# TF 2.7 ships no sm_120 (Blackwell / RTX 5080) kernel binaries, so every kernel
# is JIT-compiled from PTX ("could take 30 minutes or longer"). That result is
# cached on disk, but the 1 GiB default cap overflows and thrashes, so the cost
# recurs every run. 4 GiB (the maximum) makes it a genuine one-time cost.
os.environ['CUDA_CACHE_MAXSIZE'] = '4294967296'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append('C:/Users/User/Documents/GitHub/poseTrackingXL')
sys.path.append('C:/Users/User/Documents/GitHub/poseTrackingXL/utils')
sys.path.append('C:/Users/User/Documents/GitHub/poseTrackingXL/faceNet')
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

from tensorflow.keras.models import load_model as tf_load
from load_matlab_data import loadmat_sbx
from slp_utils_XL import posture_tracker

#%% parameters - keep these in sync with trackPosture_bench_072026.py
N_DIAG_FRAMES = 50          # how many frames to inspect (keep small - one figure per frame)
start_frame = 34600
root_dir = "Z:/Sherry/acquisition/"
vid_root = f"{root_dir}ROS108/ROS108_062526/"
cam_ids = ['blue_cam', 'green_cam', 'red_cam', 'yellow_cam']
cam_params = loadmat_sbx("Z:/Sherry/poseTrackingXL/calibration_files/all_opt_arrays/102324_negated_camParams")['camParams_negateR']

FACE_W3D = 8
FACE_CENTER_PARTS = [0, 1, 7, 11]      # posture keypoints used for the training crops
MIN_TOTAL_WEIGHT = 1.0
COMPARE_CENTERING = True        # render comNet-centred vs posture-centred side by side

comNet = "Z:/Sherry/poseTrackingXL/training_files/SLP/models/ILbaseCom260122_095351.single_instance.n=1684"
postureNet = "Z:/Sherry/poseTrackingXL/training_files/SLP/models/042126_2536FramesTotal260421_194809.single_instance.n=344"
faceNet = "C:/Users/User/Documents/GitHub/poseTrackingXL/faceNet/j5-xl-071926-zoom.h5"
faceNet_classes = "C:/Users/User/Documents/GitHub/poseTrackingXL/faceNet/j5-xl-071926-zoom_classes.npy"

with tf.device('/GPU:0'):
    face_model = tf_load(faceNet, compile=True)
class_names = np.load(faceNet_classes, allow_pickle=True)
names = list(class_names)
i_has, i_occ, i_no = names.index('hasFood'), names.index('occluded'), names.index('noFood')


#%%
class DiagnosticTracker(posture_tracker):
    '''
    Keeps every head crop it feeds to the model, plus the per-view softmax, so
    the inputs can be inspected after the run instead of only the final score.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.crops = []   # list of (crop_h, crop_w, 3, n_cams) uint8, one per frame
        self.probs = []   # list of (n_cams, 3) softmax
        self._face_fn = None   # traced forward pass, built on first use

    def _predict_face(self, face_mdl, face_img_rgb):
        self.crops.append(face_img_rgb.copy())   # copy: the buffer is reused each frame
        batch = np.ascontiguousarray(np.transpose(face_img_rgb, (3, 0, 1, 2)))
        # direct call, not predict_on_batch - see trackPosture_bench_072026.py:
        # Keras 2.7's predict_on_batch builds a new tf.data iterator per call
        if self._face_fn is None:
            self._face_fn = tf.function(lambda x: face_mdl(x, training=False))
        probs = np.asarray(self._face_fn(tf.convert_to_tensor(batch)))
        self.probs.append(probs.copy())
        return 0.0   # score is recomputed below; the base class just needs something


def consensus(probs, min_total_weight=MIN_TOTAL_WEIGHT):
    w_vis = 1.0 - probs[:, i_occ]
    f_food = probs[:, i_has] / (probs[:, i_has] + probs[:, i_no] + 1e-9)
    sum_w = w_vis.sum()
    score = float((w_vis * f_food).sum() / sum_w) if sum_w >= min_total_weight else np.nan
    return w_vis, f_food, sum_w, score


def run(face_center_parts, tag):
    readers = []
    for cam in cam_ids:
        rd = cv2.VideoCapture(f"{vid_root}{cam}.avi", cv2.CAP_FFMPEG)
        if start_frame > 0:
            rd.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        readers.append(rd)
    obj = DiagnosticTracker(readers, cam_params,
                            ds_fac=4, w3d=80, crop_size=(320, 320),
                            com_model=comNet, posture_model=postureNet,
                            face_model=face_model,
                            face_w3d=FACE_W3D,
                            face_center_parts=face_center_parts)
    obj.track_video(start_frame=start_frame, nFrames=N_DIAG_FRAMES)
    for rd in readers:
        rd.release()
    print(f'[{tag}] captured {len(obj.crops)} frames')
    return obj


runs = [(run(FACE_CENTER_PARTS, 'posture-centred'), 'posture')]
if COMPARE_CENTERING:
    runs.append((run(None, 'comNet-centred'), 'comNet'))
runs = runs[::-1] if COMPARE_CENTERING else runs   # show comNet first (the current default)

#%% per-frame view of what the model actually sees
n_cams = len(cam_ids)
for f in range(N_DIAG_FRAMES):
    fig, ax = plt.subplots(len(runs), n_cams, figsize=(3.1 * n_cams, 3.5 * len(runs)),
                           squeeze=False)
    for r, (obj, tag) in enumerate(runs):
        if f >= len(obj.crops):
            continue
        probs = obj.probs[f]
        w_vis, f_food, sum_w, score = consensus(probs)
        for c in range(n_cams):
            ax[r, c].imshow(obj.crops[f][:, :, :, c])
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            k = int(np.argmax(probs[c]))
            colour = '0.6' if w_vis[c] < 0.5 else 'k'
            ax[r, c].set_title(f'{tag} cam{c}: {class_names[k]} {probs[c, k]:.2f}\n'
                               f'w={w_vis[c]:.2f} f={f_food[c]:.2f}',
                               fontsize=8, color=colour)
        ax[r, 0].set_ylabel(tag, fontsize=10)
        note = 'NaN (too occluded)' if np.isnan(score) else f'{score:.3f}'
        ax[r, n_cams - 1].set_title(ax[r, n_cams - 1].get_title() +
                                    f'\nSCORE={note} (sw={sum_w:.2f})', fontsize=8)
    fig.suptitle(f'video frame {start_frame + f}')
    plt.tight_layout()
    plt.show()

#%% summary: how much does the centring choice change the scores?
if COMPARE_CENTERING and len(runs) == 2:
    scores = {}
    for obj, tag in runs:
        scores[tag] = np.array([consensus(p)[3] for p in obj.probs])
    a, b = scores['comNet'], scores['posture']
    both = np.isfinite(a) & np.isfinite(b)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(a, label='comNet-centred', marker='o', ms=3)
    ax[0].plot(b, label='posture-centred', marker='o', ms=3)
    ax[0].axhline(0.5, color='r', ls='--', lw=0.8)
    ax[0].set_xlabel('frame (offset from start_frame)'); ax[0].set_ylabel('carry_score')
    ax[0].legend(fontsize=8); ax[0].set_title('score by crop centring')
    ax[1].scatter(a[both], b[both], s=12)
    ax[1].plot([0, 1], [0, 1], 'r--', lw=0.8)
    ax[1].set_xlabel('comNet-centred'); ax[1].set_ylabel('posture-centred')
    ax[1].set_title('per-frame agreement')
    plt.tight_layout(); plt.show()

    flips = int((((a > 0.5) != (b > 0.5)) & both).sum())
    print(f'frames where the carrying call flips: {flips}/{int(both.sum())}')
    print(f'NaN (too occluded)  comNet: {int(np.isnan(a).sum())}   posture: {int(np.isnan(b).sum())}')
    print(f'mean |difference|: {np.abs(a[both] - b[both]).mean():.3f}')
