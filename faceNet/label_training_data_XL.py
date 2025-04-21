#%%
import numpy as np
import cv2
import sys
sys.path.append("C:/Users/xl313/OneDrive/Documents/GitHub/poseTrackingXL")
sys.path.append("C:/Users/xl313/OneDrive/Documents/GitHub/poseTrackingXL/utils")
from load_matlab_data import loadmat_sbx
sys.path.append("C:/Users/xl313/OneDrive/Documents/GitHub/poseTrackingXL/camera_calibration")
from pySBA import PySBA
import matplotlib.pyplot as plt
import notebook
from scipy.io import savemat
from datetime import datetime

# %matplotlib
notebook
#%%
''' Data Params '''
# define file paths
bird_id = 'LVN4'
session_date = '040425_3rdpart'
session_root = f'Z:/Sherry/acquisition'
session_dir = f'{bird_id}_{session_date}'
vid_path = f'{session_root}/{session_dir}/'
pred_file = f'040525_posture_face.npy'
pred_path = f"{vid_path}{pred_file}"

# to save training images
current_datetime = datetime.now().strftime('%Y%m%d%H%M')
save_dir = 'Z:/Sherry/poseTrackingXL/SeedCarryingLabeling/LabeledData/'
save_file = f'seedLabel_{session_dir}_{current_datetime}_XL.mat'
# save_file = f'seedLabel_{session_dir}.npz'
save_path = f'{save_dir}{save_file}'

# camera params
cam_params = loadmat_sbx("Z:/Sherry/poseTrackingXL/calibration_files/all_opt_arrays/102324_negated_camParams")['camParams_negateR']
cam_ids = ['blue_cam', 'green_cam', 'red_cam', 'yellow_cam'] # make sure it is in order 
n_cams = len(cam_ids)

# video params
n_images = 1000 # 1000 # number of frames to label

#%%
''' Get the frame index - random frames near interactions '''
# # load the site interactions
# annotated_seed_path = f'{vid_path}/annotatedSeeds.mat'
# seed_struct = loadmat_sbx(annotated_seed_path)['annotatedSeeds']
# count_data = seed_struct['countData']
# # get interaction frames
# all_int_start = count_data['newSite']
# all_int_end = count_data['endSite']
# # take surrounding frames
# int_frames = np.concatenate((all_int_start - 5, all_int_end + 5))
# near_idx = np.arange(6, 15)
# for i in near_idx:
#     int_frames = np.concatenate((int_frames, all_int_start-i))
#     int_frames = np.concatenate((int_frames, all_int_end+i))
# # get caches + retrievals frames
# all_int_changes = np.sum(seed_struct['seedChanges'], axis=1)
# cache_onsets = all_int_start[all_int_changes > 0]
# cache_offsets = all_int_end[all_int_changes > 0]
# ret_onsets = all_int_start[all_int_changes < 0]
# ret_offsets = all_int_end[all_int_changes < 0]
# # take more surrounding frames
# near_idx = np.arange(15, 50)
# for i in near_idx:
#     int_frames = np.concatenate((int_frames, cache_onsets - i, cache_offsets + i,
#                                  ret_onsets - i, ret_offsets + i))
# # select random frames near interactions to label
# frame_idx = np.random.choice(int_frames, size=n_images, replace=False)
# frame_idx = np.sort(frame_idx)

#%% Get frame Indices
# frame no. to select
coconut_frame1 = np.random.randint(8 * 50 * 60, (8 * 60 + 20) * 50, 300)
coconut_frame2 = np.random.randint((8 * 60 + 25) * 50, (8 * 60 + 58) * 50, 400)
coconut_frame3 = np.random.randint((10 * 60 + 32) * 50, (10 * 60 + 43) * 50, 300)
frame_idx = np.concatenate(
    (coconut_frame1, coconut_frame2, coconut_frame3), axis=0
)

#%%
''' For cropping '''
def crop_from_com(img, centroid, half_width, crop_size = (320,320)): # used to be (320,320)
    '''
    Crops an image around a given centroid (crop dims defined by half_width)
    and resizes to the specified crop_size.
    '''
    ctr = np.round(centroid).astype(int)
    half_width = np.round(half_width).astype(int)
    img_h,img_w = img.shape[0],img.shape[1] # used to be img_h, img_w = img.shape
    
    xmin = np.min([np.max([ctr[0] - half_width, 0]), img_w - 1])
    xmax = np.max([np.min([ctr[0] + half_width + 1, img_w]), 1])
    ymin = np.min([np.max([ctr[1] - half_width, 0]), img_h - 1])
    ymax = np.max([np.min([ctr[1] + half_width + 1, img_h]), 1])
    
    cropped_img = img[ymin:ymax, xmin:xmax]
    
    crop_img = cv2.resize(cropped_img, crop_size, interpolation=cv2.INTER_AREA)
    min_ind = np.array([xmin, ymin])
    max_ind = np.array([xmax, ymax])
    crop_scale = crop_size / (max_ind - min_ind)
    
    return crop_img, min_ind, crop_scale

face_crop_size = (128,128) # pixels
face_w3d = 20 # scaling factor # IL: 0.08
com_head_ind = 0
face_images = np.full((face_crop_size[1], face_crop_size[0], 3, n_cams, n_images), 0, dtype='uint8') # added 3
#%%
''' Create image set '''
# load the COM predictions to find the face
print('Loading COM predictions')
results_dict = np.load(pred_path, allow_pickle=True).item()
results = results_dict['results']
com_preds = results['com_preds'] # shape (n_frames, n_keypoints, 3)
n_frames_total = com_preds.shape[0] # total number of frames in video

# define the video reader for each camera
all_readers = []
for i in range(len(cam_ids)):
    cam = cam_ids[i]
    print(cam)
    camPath = f"{vid_path}{cam}.avi"

    # define the video reader obj and settings
    api_id = cv2.CAP_FFMPEG
    reader = cv2.VideoCapture(camPath, api_id)
    all_readers.append(reader)

# flag that video reading failed
stopReading = False
#%%
# crop and save each frame
print('Saving cropped images')
sba = PySBA(cam_params, np.nan, np.nan, np.nan, np.nan)
im_idx = 0
for n_frame in range(n_frames_total):
    # Read and downsample
    full_img = []
    for nCam in range(n_cams):
        if n_frame in frame_idx:
            all_readers[nCam].set(cv2.CAP_PROP_POS_FRAMES, n_frame)
            flag, img = all_readers[nCam].read()
            if img is None:
                stopReading = True
                break
            full_img.append(img) # used to be img[:,:,0]
            if full_img[nCam] is None:
                stopReading = True
                break
    # If reading for any video failed, terminate tracking
    if stopReading:
        print(f'Terminated Reading on Frame {n_frame}')
        break

    if n_frame in frame_idx:
        print(f'Reading Frame {im_idx + 1}/{n_images}')
        # get the 3D distance from each camera for cropping scale
        head_COM = com_preds[n_frame, com_head_ind]
        head_reproj = sba.project(np.tile(head_COM, (n_cams, 1)),
                                    cam_params)  # get reprojected body centroid location for each camera
        camDist = sba.rotate(np.tile(head_COM, (n_cams, 1)),
                             cam_params[:, :3])  # rotate to camera coordinates
        camDist = camDist[:, 2] + cam_params[:, 5]  # get z-axis distance ie along optical axis
        camScale = cam_params[:, 6] / camDist  # convert to focal length divided by distance
        half_width = camScale * face_w3d

        # save the cropped image for each camera
        min_ind = np.full((n_cams, 1, 2), np.nan)
        crop_scale = np.full((n_cams, 1, 2), np.nan)
        for nCam in range(n_cams):
            thisCom = np.maximum(head_reproj[nCam], 0)
            thisCom[0] = np.minimum(thisCom[0], full_img[nCam].shape[1])  # x limit is shape[1]
            thisCom[1] = np.minimum(thisCom[1], full_img[nCam].shape[0])  # y limit is shape[0]
            thisHalfWidth = np.maximum(half_width[nCam], 15)  # minimum 31px image for head
            face_images[:, :,:, nCam, im_idx], _, _ = crop_from_com(full_img[nCam],
                                                                    thisCom,
                                                                    thisHalfWidth,
                                                                    face_crop_size)
        im_idx += 1
    if im_idx == n_images:
        break

#%%
face_images_rgb = np.zeros_like(face_images)  # Initialize an array to store RGB images
face_images_bw = np.zeros((face_crop_size[1], face_crop_size[0], n_cams, n_images),dtype='uint8')
for cam in range(face_images.shape[3]):  # Iterate over cameras
    for img_idx in range(face_images.shape[4]):  # Iterate over images
        face_images_rgb[:, :, :, cam, img_idx] = cv2.cvtColor(face_images[:, :, :, cam, img_idx], cv2.COLOR_BGR2RGB)
        face_images_bw[:, :, cam, img_idx] = cv2.cvtColor(face_images_rgb[:, :, :, cam, img_idx], cv2.COLOR_RGB2GRAY)

#%%
def label_images(images):
    seed_labels = []
    color_labels = []

    def on_key(event):
        if event.key in ['0', '2','3','9']:
            print(f"Key pressed: {event.key}")
            if event.key == '0':
                seed_labels.append(int(event.key))
                color_labels.append(int(event.key))
            elif event.key == '2':
                seed_labels.append(1)
                color_labels.append(int(event.key))
            elif event.key == '3':
                seed_labels.append(1)
                color_labels.append(int(event.key))
            elif event.key == '9':
                print(f"skipping to the next frame")
                seed_labels.append(np.nan)
                color_labels.append(np.nan)
            idx = len(seed_labels)
            if idx < images.shape[-1]: # present the next image
                ax[0, 0].imshow(images[:, :, :, 0, idx], cmap='gray')
                ax[1, 0].imshow(images[:, :, :, 1, idx], cmap='gray')
                ax[0, 1].imshow(images[:, :, :, 2, idx], cmap='gray')
                ax[1, 1].imshow(images[:, :, :, 3, idx], cmap='gray')
                fig.suptitle(f"Image {idx+1}/{images.shape[-1]}: Press 0 (no seed), 2 (white peanut), 3 (blue coconut), 9 (skip)")
                fig.canvas.draw()
            else:
                print("No more images to process.")
                plt.close()
        else:
            print("Invalid key pressed. Please press either '0' or '1'.")

    # display image
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].imshow(images[:, :, :, 0, 0], cmap='gray')
    ax[1, 0].imshow(images[:, :, :, 1, 0], cmap='gray')
    ax[0, 1].imshow(images[:, :, :, 2, 0], cmap='gray')
    ax[1, 1].imshow(images[:, :, :, 3, 0], cmap='gray')
    fig.suptitle(f"Image 1/{images.shape[-1]}: Press 0 (no seed), 2 (white peanut), 3 (blue coconut), 9 (skip)")
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

    return np.asarray(color_labels), np.asarray(seed_labels)


color_labels, seed_labels = label_images(face_images_rgb)
#%%
save_dict = {
    "allIms": face_images_bw,
    "allIms_RGB": face_images_rgb,
    "seedLabels": seed_labels,
    "colorlabels": color_labels,
    "frameTimes": frame_idx.T
}

print(f'Saving file to {save_path}')
savemat(f'{save_path}', save_dict)
# # save as a dict
# save_dict = {
#     "all_images": face_images,
#     "seed_labels": seed_labels,
#     "color_labels": color_labels
# }
# print(f'Saving file to {save_path}')
# np.savez_compressed(f'{save_path}', **save_dict)
