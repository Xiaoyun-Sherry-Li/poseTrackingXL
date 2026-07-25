#%%
# Based on trackPosture_bench_042126.py. Two changes from that script:
#
#   1. FIX: the .mat save dict referenced `sleap_raw_predicted_points_scale_back`,
#      a name that is never assigned anywhere in the original script -> NameError
#      on save. It should be `results['posture_rawpred']` (the untriangulated
#      per-camera SLEAP points) - that's what's used below.
#
#   2. faceNet swapped from the legacy grayscale joint model (j5-xl-041925.keras)
#      to the newer single-view RGB classifier (j5-xl-071926-zoom.h5, 3-class
#      softmax over ['hasFood', 'occluded', 'noFood']), combined across cameras
#      with the confidence-weighted consensus method from
#      faceNet/testFaceNet_072026.ipynb:
#          w_i = 1 - P(occluded)_i                              (visibility weight)
#          f_i = P(hasFood)_i / (P(hasFood)_i + P(noFood)_i)     (food evidence | visible)
#          carry_score = sum(w_i * f_i) / sum(w_i)
#      This is a convex combination, so it's bounded in [min(f_i), max(f_i)]
#      regardless of how small the w_i are - shrinking every view's weight by
#      the same factor doesn't change the ratio. So sum(w_i) is tracked
#      separately as a reliability gate: frames where sum(w_i) < MIN_TOTAL_WEIGHT
#      don't have enough total visibility across cameras to trust, and are
#      marked NaN in 'face_preds' rather than forced into a score.
import sys
import os
# CUDA JIT cache size - MUST be set before tensorflow initialises the CUDA
# context, hence before `import tensorflow`.
# TF 2.7 ships no sm_120 (Blackwell / RTX 5080) kernel binaries, so every kernel
# is JIT-compiled from PTX ("could take 30 minutes or longer"). That result is
# cached on disk, but the 1 GiB default cap overflows and thrashes, so the cost
# recurs every run. 4 GiB (the maximum) makes it a genuine one-time cost.
os.environ['CUDA_CACHE_MAXSIZE'] = '4294967296'
sys.path.append('C:/Users/User/Documents/GitHub/poseTrackingXL')
sys.path.append('C:/Users/User/Documents/GitHub/poseTrackingXL/utils')
sys.path.append('C:/Users/User/Documents/GitHub/poseTrackingXL/faceNet')
import numpy as np
import tensorflow as tf
# NOTE: these two were previously set *after* `import tensorflow`, where they
# have no effect - both are read at import/context-init. Move them above the
# tensorflow import if you re-enable them.
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress warnings and info, show only errors
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # forcing tensorflow to use GPU
import cv2

''' set up for this run '''
# set up GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

from tensorflow.keras.models import load_model as tf_load
from load_matlab_data import loadmat_sbx
import pySBA
from slp_utils_XL import posture_tracker, create_slp_project, crop_from_com
from sleap import load_model as slp_load
import scipy.io

#%% parameters
''' UPDATE data params as appropriate'''
# NFRAMES: how many frames to track per session, starting at start_frame.
#   None -> auto-detect from each session's videos (shortest camera minus
#           start_frame, so track_video never reads past the end of any view).
#           Use this for full-session runs.
#   int  -> hard cap applied to every session, e.g. a quick test
#           (187*60*50 = one 187-min session at 50fps; a full run took ~178 min).
NFRAMES = 2000

# ---------------------------------------------------------------------------
# SESSIONS TO PROCESS - edit this list. One vid_root per line; each must be a
# folder holding blue_cam.avi / green_cam.avi / red_cam.avi / yellow_cam.avi.
# The batch runs them top to bottom, SKIPS any that already have output (see
# SAVE_TAG), and keeps going if one session fails. Comment a line out with # to
# leave that session out of the run.
# ---------------------------------------------------------------------------
root_dir = "Z:/Sherry/acquisition/"
SESSIONS = [
    f"{root_dir}ROS108/ROS108_070126/",
    # f"{root_dir}ROS108/ROS108_070326/",
    # f"{root_dir}ROS108/ROS108_070726/",
    # f"{root_dir}ROS108/ROS108_070926/",
    # f"{root_dir}ROS108/ROS108_071426/",
    # f"{root_dir}ROS108/ROS108_071826/",
    # f"{root_dir}ROS108/ROS108_071926/",
    # f"{root_dir}ROS108/ROS108_072126/",
    # f"{root_dir}ROS108/ROS108_072326/",
    # f"{root_dir}RBY97/RBY97_061926",
    # f"{root_dir}RBY97/RBY97_062226",
    # f"{root_dir}RBY97/RBY97_062426",
    # f"{root_dir}RBY97/RBY97_062626",
    # f"{root_dir}RBY97/RBY97_063026",
    # f"{root_dir}RBY97/RBY97_070226",
]

# Output files written into each session folder: <SAVE_TAG>.npy and <SAVE_TAG>.mat.
# The batch skips a session when its <SAVE_TAG>.mat already exists, so this name
# is also what makes "resume where it left off" work - keep it fixed across a run.
SAVE_TAG = "raw_pred_V4"

# video params
start_frame = 0
# cam params
cam_ids = ['blue_cam', 'green_cam', 'red_cam', 'yellow_cam'] # check the input order
im_w = 2200
im_h = 650
# if True, also save which 3 cameras won each triangulation
# (0-based indices into cam_ids; add 1 when indexing in MATLAB)
return_tri_cams = False
# camera params
cam_params = loadmat_sbx("Z:/Sherry/poseTrackingXL/calibration_files/all_opt_arrays/102324_negated_camParams")['camParams_negateR'] #['camParams']

# faceNet crop + consensus params
# Both values below were chosen by eye in faceNet/diagnoseFaceNet_tracking.py -
# keep the two scripts in sync if you retune either one.
#
# face_w3d controls how tight the head crop is before it goes to the face model
# (separate from w3d above, which sets the body crop for postureNet). Smaller =
# more zoomed in. The training crops used 7 (faceNet/label_training_data_XL.ipynb);
# 8 framed the head better on this rig in the diagnostic. Note this is NOT the
# posture_tracker default of 20, which suited the old j5-xl-041925.keras model.
FACE_W3D = 8
# Which 3D point the head crop is centred on: the mean of these triangulated
# posture keypoints. MUST NOT be left as None - that falls back to the coarse
# comNet head COM, which frames the head quite differently from anything the
# model saw in training and visibly degrades the predictions.
# The training notebook used pos_center_ind = [0, 1]; [0, 1, 7, 11] (the head_idx
# from testFaceNet_072026.ipynb) looked better in the diagnostic.
FACE_CENTER_PARTS = [0, 1, 7, 11]
MIN_TOTAL_WEIGHT = 1.0  # ~one fully-visible-equivalent view; below this, mark the frame NaN

#%%
# models
comNet = "Z:/Sherry/poseTrackingXL/training_files/SLP/models/comNet260720_203202.single_instance.n=2656"
# comNet = "Z:/Sherry/poseTrackingXL/training_files/SLP/models/ILbaseCom260122_095351.single_instance.n=1684"
# postureNet = "Z:/Sherry/poseTrackingXL/training_files/SLP/models/042126_2536FramesTotal260421_194809.single_instance.n=344"
postureNet = "Z:/Sherry/poseTrackingXL/training_files/SLP/models/260722_175335.single_instance.n=3326"
# single-view RGB classifier, input (128,128,3), 3-class softmax over
# ['hasFood', 'occluded', 'noFood'] - replaces the legacy grayscale joint model.
# NOTE: unlike comNet/postureNet (on Z:), this model currently only exists in the
# local repo - it was never copied to Z:/Sherry/poseTrackingXL/faceNet/. If you
# want it on Z: for consistency, copy both files there and switch these paths.
faceNet = "C:/Users/User/Documents/GitHub/poseTrackingXL/faceNet/j5-xl-071926-zoom.h5"
faceNet_classes = "C:/Users/User/Documents/GitHub/poseTrackingXL/faceNet/j5-xl-071926-zoom_classes.npy"

with tf.device('/GPU:0'):  # Explicitly place model on GPU # added by sherry 072725
    face_model = tf_load(faceNet, compile=True)
    print("Model device:", face_model.weights[0].device)
class_names = np.load(faceNet_classes, allow_pickle=True)

# Load the SLEAP models HERE, before opening the video readers below. All the
# slow work (model load + first-use JIT compilation) then happens while no video
# file handles are held, so the readers are opened immediately before track_video
# reads them. track_video accepts these pre-loaded models as well as path strings.
print("loading comNet and postureNet...")
com_mdl = slp_load([comNet], peak_threshold=0)
posture_mdl = slp_load([postureNet], peak_threshold=0)


class RGBConsensusPostureTracker(posture_tracker):
    '''
    posture_tracker with the face-scoring step swapped for the single-view RGB
    classifier, combined across cameras via confidence-weighted consensus
    (derivation in faceNet/testFaceNet_072026.ipynb):
        w_i = 1 - P(occluded)_i
        f_i = P(hasFood)_i / (P(hasFood)_i + P(noFood)_i)
        carry_score = sum(w_i * f_i) / sum(w_i)

    carry_score is a convex combination of the f_i, so it's always bounded in
    [min(f_i), max(f_i)] - regardless of how small the w_i are. sum(w_i) is
    therefore tracked separately as a reliability signal: frames where
    sum(w_i) < min_total_weight don't have enough total visibility across
    cameras to trust the score, and are marked NaN instead.

    'face_preds' in the results dict returned by track_video() holds this
    per-frame carry_score (float, NaN where unreliable). Per-camera softmax +
    weights for every frame are additionally kept on self.face_raw_history for
    later inspection (not part of the base class's results dict).
    '''
    def __init__(self, *args, class_names, min_total_weight=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        names = list(class_names)
        self.i_has, self.i_occ, self.i_no = names.index('hasFood'), names.index('occluded'), names.index('noFood')
        self.min_total_weight = min_total_weight
        self.face_raw_history = []  # list of (n_cams, 3) softmax arrays, one per frame
        self._face_fn = None        # traced forward pass, built on first use

    def _predict_face(self, face_mdl, face_img_rgb):
        # face_img_rgb: (crop_h, crop_w, 3, n_cams) -> (n_cams, crop_h, crop_w, 3)
        # ascontiguousarray: the transpose of the reused crop buffer is a
        # non-contiguous view, and the model wants a real batch-major array
        batch = np.ascontiguousarray(np.transpose(face_img_rgb, (3, 0, 1, 2)))
        # NOT predict_on_batch: in Keras 2.7 that builds a fresh tf.data iterator
        # on every call (data_adapter.single_batch_iterator, see
        # keras/engine/training.py), so calling it per frame churns datasets and
        # threads - the source of the repeated "Optimization loop failed:
        # CANCELLED" warnings. Calling the model directly skips tf.data entirely;
        # tf.function traces it once (shape and dtype are constant per run).
        if self._face_fn is None:
            self._face_fn = tf.function(lambda x: face_mdl(x, training=False))
        probs = np.asarray(self._face_fn(tf.convert_to_tensor(batch)))  # (n_cams, 3) softmax
        self.face_raw_history.append(probs.copy())

        w_vis = 1.0 - probs[:, self.i_occ]
        f_food = probs[:, self.i_has] / (probs[:, self.i_has] + probs[:, self.i_no] + 1e-9)
        sum_w = w_vis.sum()
        if sum_w >= self.min_total_weight:
            return float((w_vis * f_food).sum() / sum_w)
        return np.nan  # too occluded across all cameras to trust a score


#%%
def process_session(vid_root):
    """Run the full tracking pipeline on one session folder and save outputs.

    Reuses the models already loaded above (the slow part), so batching many
    sessions only pays the load/JIT cost once. Video readers are always released
    at the end - even on failure - so no stale handle is left on the Z: share for
    the next session or a retry.
    """
    save_path_py = f"{vid_root}{SAVE_TAG}.npy"
    save_path_mat = f"{vid_root}{SAVE_TAG}.mat"

    # define the video reader for each camera
    all_readers = []
    try:
        for cam in cam_ids:
            camPath = f"{vid_root}{cam}.avi"
            reader = cv2.VideoCapture(camPath, cv2.CAP_FFMPEG)
            # Fail loudly here if a video won't open, instead of much later with an
            # opaque cvtColor(None) error deep inside track_video. A reader can fail
            # to open if the file is missing, or if a previous crashed run still
            # holds a handle on it (esp. on the Z: network share).
            if not reader.isOpened():
                raise RuntimeError(f"failed to open {camPath} - is another run still holding it?")
            print(f"{cam}: opened, {int(reader.get(cv2.CAP_PROP_FRAME_COUNT))} frames")
            if start_frame > 0:
                reader.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            all_readers.append(reader)

        # Per-session nFrames: honour a hard cap when NFRAMES is set, otherwise
        # auto-detect from the shortest camera (cameras occasionally differ by a
        # frame or two) minus start_frame, so track_video never runs past the end
        # of any view.
        if NFRAMES is None:
            frame_counts = [int(r.get(cv2.CAP_PROP_FRAME_COUNT)) for r in all_readers]
            nFrames = min(frame_counts) - start_frame
            if nFrames <= 0:
                raise RuntimeError(
                    f"auto-detected nFrames={nFrames} (shortest cam {min(frame_counts)} "
                    f"frames, start_frame {start_frame}) - nothing to track")
            print(f"nFrames auto-detected: {nFrames} "
                  f"(cam frame counts {frame_counts}, start_frame {start_frame})")
        else:
            nFrames = NFRAMES

        ''' track posture '''
        obj = RGBConsensusPostureTracker(all_readers, cam_params,
                                ds_fac=4,
                                w3d=80,
                                crop_size=(320,320),
                                com_model=com_mdl,
                                posture_model=posture_mdl,
                                face_model=face_model,
                                face_w3d=FACE_W3D,
                                face_center_parts=FACE_CENTER_PARTS,
                                class_names=class_names,
                                min_total_weight=MIN_TOTAL_WEIGHT,
                                return_cams=return_tri_cams)

        results = obj.track_video(start_frame=start_frame, nFrames=nFrames)
        face_rawpred = np.stack(obj.face_raw_history)  # (n_frames, n_cams, 3) per-camera softmax, for diagnostics
    finally:
        # Always release the readers, even if tracking above raised, so the next
        # session (and any retry of this one) can reopen these files cleanly.
        for r in all_readers:
            r.release()

    ''' save file '''
    # for python
    save_dict = {"results": results,
                "camNames": cam_ids,
                "session": vid_root,
                "start_frame": start_frame,
                "n_frames": nFrames,
                "cam_params": cam_params,
                "face_class_names": class_names,
                "face_rawpred": face_rawpred,
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
        "face_preds": results['face_preds'],  # per-frame carry_score, NaN where too occluded to trust
        "face_rawpred": face_rawpred,  # per-camera softmax (n_frames, n_cams, 3), for diagnostics
        "face_class_names": class_names,
        "camNames": cam_ids,
        "session": vid_root,
        "nFrames": nFrames,
        # 0-based index of the first tracked video frame, so the MATLAB side can map
        # results back to absolute video frames without hardcoding it:
        #   matlabFrame = start_frame + (1:nFrames)
        "start_frame": start_frame,
        "camParams": cam_params,
        "rawPostures": results['posture_rawpred'],  # fixed: was the undefined name sleap_raw_predicted_points_scale_back
    }
    if return_tri_cams:
        # 0-based camera indices (order of cam_ids); add 1 when indexing in MATLAB
        results_struct["com_tri_cams"] = results['com_tri_cams']
        results_struct["posture_tri_cams"] = results['posture_tri_cams']
    # Save the struct to a .mat file. Written last, so the presence of this file
    # is what the batch loop treats as "session fully done".
    scipy.io.savemat(save_path_mat, {"results": results_struct})
    print(f"saved {save_path_py} and {save_path_mat}")


#%%
''' run the batch over every session in SESSIONS '''
import traceback
done, skipped, failed = [], [], []
for vid_root in SESSIONS:
    out_mat = f"{vid_root}{SAVE_TAG}.mat"
    print("\n" + "=" * 72)
    print(f"SESSION: {vid_root}")
    if os.path.exists(out_mat):
        print(f"  already has {SAVE_TAG}.mat - skipping")
        skipped.append(vid_root)
        continue
    try:
        process_session(vid_root)
        done.append(vid_root)
        print(f"  DONE: {vid_root}")
    except Exception as e:
        # One bad session (missing/locked video, decode error, ...) must not abort
        # the whole vacation batch - log the full traceback and move to the next.
        print(f"  FAILED: {vid_root}\n{traceback.format_exc()}")
        failed.append((vid_root, repr(e)))

print("\n" + "=" * 72)
print(f"BATCH COMPLETE: {len(done)} done, {len(skipped)} skipped, {len(failed)} failed")
for v in done:      print(f"  done     {v}")
for v in skipped:   print(f"  skipped  {v}")
for v, e in failed: print(f"  FAILED   {v}   ({e})")
