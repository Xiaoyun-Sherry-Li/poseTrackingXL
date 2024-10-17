import os
import numpy as np
import matplotlib.pyplot as plt

# define file paths
tmp_local_root = "C:/Users/ilow1/Documents/code/bird_pose_tracking/"
root_dir = f"{tmp_local_root}calibration_files/raw_centroids/"
cam_ids = ['red_cam', 'yellow_cam', 'green_cam', 'blue_cam']
img_date = input("input date of image acquisition (YYMMDD): ")
centroid_files = []
for cam_id in cam_ids:
    centroid_files.append(f"{img_date}_{cam_id}_centroids.npz")

# data params
# determine the minimum number of images across cameras
n_images = 10000
for f in centroid_files:
    with np.load(f"{root_dir}{f}") as data:
        areas = data['arr_0']
        n_images = np.min((n_images, areas.shape[0]))
n_cams = len(cam_ids) # number of cameras

# to store final centroids
pts = np.empty(shape=(n_images, 2, n_cams))
nan_pts = np.zeros(shape=(n_images, n_cams), dtype=bool)


fig, axs = plt.subplots(n_cams, 1, sharex=True)
for i, f in enumerate(centroid_files):
    # load this camera's centroids
    filename = f"{root_dir}{f}"
    with np.load(filename) as data:
        areas = data['arr_0'][:n_images]
        centroids = data['arr_1'][:n_images]

    # plot centroid locations for each camera
    pts[:, :, i] = centroids.copy()
    nan_pts[:, i] = np.isnan(areas)
    colors = np.linspace(0, 1, len(areas))
    axs[i].scatter(centroids[:, 1], centroids[:, 0],\
                    s=10, c=colors, alpha=0.5)
    axs[i].title.set_text(cam_ids[i])

# POSSIBLY CHANGE THIS
# find frames where there was no point or multiple points for 1 or more of the cameras
# keep only single points visible to all cameras
in_pts = np.zeros(shape=(n_images,), dtype=bool)
for i in range(n_images):
    in_pts[i] = (not any(nan_pts[i, :]))
good_pts = pts[in_pts, :, :] 

# save the preprocessed centroid locations
outfile = f"{tmp_local_root}calibration_files/preprocessed_centroids/{img_date}_centroids"
np.savez(outfile, good_pts)
fig.savefig(f"{outfile}_each_cam.png", dpi=600, bbox_inches='tight')
plt.show()