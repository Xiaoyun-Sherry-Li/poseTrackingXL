import numpy as np
from scipy.spatial.transform import Rotation as R
import pySBA
import matplotlib.pyplot as plt
import calibration_functions as fxns


'''
------
Step 0
------
Set file paths, initialize camera array.
'''
''' Adjust these params as needed '''
# base file paths
tmp_local_root = "C:/Users/ilow1/Documents/code/bird_pose_tracking/"
root_dir = f"{tmp_local_root}calibration_files/"

# camera array file paths
cam_file_path = root_dir
init_array_folder = 'init_cam_arrays/' # to load initial estimates
opt_array_file = '240327_opt_cam_array_zeroes_init.npy' # to load optimized array

# known 3D points file paths
known_pt_folder = f"{root_dir}known_points/"

# to save output figures
date_id = input("input today's date (YYMMDD): ")
save_dir = f"{tmp_local_root}figures/camera_calibration_figs/{date_id}/"
if os.path.exists(save_dir):
    print("warning, save folder already exists!")
else:
    os.mkdir(save_dir)

# camera params
cam_ids = ['red_cam', 'yellow_cam', 'green_cam', 'blue_cam']

''' initialize the camera array from estimates or a previous array '''
# load camera parameters - initial estimates
# cam_array, cam_array_fields = pySBA.getCameraArray(f'{cam_file_path}{init_array_folder}',
#                                                         camera_ids=cam_ids)


# load camera parameters - saved file
cam_array, cam_array_fields = pySBA.getCameraArray(cam_file_path,
                                                    camera_ids=cam_ids,
                                                    load_opt_array=True, 
                                                    opt_file_name=opt_array_file)

'''
------
Step 1
------
Optimize the camera extrinsics using points with known 3D coordinates.
Hold the point locations and camera intrinsics fixed.

This step ensures that the optimization begins in the right reference frame,
relative to the arena.
'''
# load the points and their indexing variables
img_date = input("input date of image acquisition for known points (YYMMDD): ")
pts_3d_known = np.load(f'{known_pt_folder}points_3d_simple.npy')
pts_2d_known = np.load(f'{known_pt_folder}{img_date}_points_2d.npy')
n_pts = pts_2d_known.shape[0]
known_pts_camera_idx = np.load(f'{known_pt_folder}{img_date}_camera_ind.npy')
known_pts_idx = np.load(f'{known_pt_folder}point_ind.npy')

# reformat to remove extraneous dim
known_pts_camera_idx = np.squeeze(known_pts_camera_idx.astype(int))
known_pts_idx = np.squeeze(known_pts_idx.astype(int))

# initialize the SBA object
sba = pySBA.PySBA(cam_array, pts_3d_known, pts_2d_known, 
                  known_pts_camera_idx, known_pts_idx)

# display the initial camera params
fxns.display_cam_array(sba, cam_array_fields)

# plot the point locations to double check them
colors = ['black', 'red', 'yellow', 'green', 'blue', 
          'red', 'yellow', 'green', 'blue']

# plot the points in 3D
fxns.plot_3D_points(pts_3d=pts_3d_known, colors=colors,
                    title=f'world coords, init pts\nred cam at (1, 1)',
                    save_path=f'{save_dir}init_pts_3d.png')

# plot the intial reprojection vs. the 2D views
# TODO
fxns.plot_2D_points(sba, colors,
                    title, save_path)

# get reprojected points
est_pts = sba.project(sba.points3D[sba.point2DIndices], sba.cameraArray[sba.cameraIndices])

# fig params
fig, axs = plt.subplots(n_cams, 2, sharex=True, figsize=(8, 8))

# selected points
for i in range(n_cams):
    cam_idx = known_pts_camera_idx==i
    axs[i, 1].scatter(pts_2d_known[cam_idx, 0],
                   pts_2d_known[cam_idx, 1],
                   c=colors, alpha=0.5, s=30)
    axs[i, 1].title.set_text(f'selected 2D pts\n{cam_ids[i]}')
    axs[i, 1].invert_yaxis()
    
# reprojected points, original extrinsics
for i in range(n_cams):
    ind2d = sba.cameraIndices==i
    axs[i, 0].scatter(est_pts[ind2d, 0],\
                    est_pts[ind2d, 1],\
                    s=30, c=colors, alpha=0.5)
    if i == 0:
        axs[i, 0].title.set_text(f'reprojection w/ est. extrinsics\n{cam_ids[i]}')
    else:
        axs[i, 0].title.set_text(f'{cam_ids[i]}')    
    axs[i, 0].invert_yaxis()
       
plt.show()
fig.savefig(f'{save_dir}init_pts_2d.png', dpi=600, bbox_inches='tight')

# optimize the camera extrinsics, holding the intrinsics fixed, using known 3D points
sba.bundleAdjust_just_extrinsics()

# plot the reprojection error
r = sba.project(sba.points3D[sba.point2DIndices], sba.cameraArray[sba.cameraIndices]) - sba.points2D
r = np.sqrt(np.sum(r**2, axis=1))

fig, ax = plt.subplots(1, 1)
ax.hist(r)
ax.set_xlabel('reprojection error (pixels)')
ax.set_title('only extrinsics adjusted')
plt.show()
fig.savefig(f'{save_dir}reproj_error_step1.png', dpi=600, bbox_inches='tight')

# display the updated camera params
x = PrettyTable()
x.field_names = cam_array_fields.copy()
for row in sba.cameraArray:
    x.add_row(np.round(row, 2))
print(x)













# load centroids found on all cameras
tmp_local_root = "C:/Users/ilow1/Documents/Python Scripts/bird_pose_tracking/"
root_dir = f"{tmp_local_root}calibration_files/preprocessed_centroids/"
img_date = input("input date of calibration image acquisition (YYMMDD): ")
centroid_file = f"{img_date}_centroids.npz"
with np.load(f"{root_dir}{centroid_file}") as data:
    pts = data['arr_0']

# flip xy (regionprops orders) <-- check this makes sense
pts = np.flip(pts, axis=1)

# data params
n_pts = pts.shape[0]
n_cams = pts.shape[2]
cam_ids = ['red_cam', 'yellow_cam', 'green_cam', 'blue_cam']
cam_array_fields = [
    'rot_1', 'rot_2', 'rot_3',
    'trans_1', 'trans_2', 'trans_3',
    'focal dist', 'distort_1', 'distort_2',
    'pt_x', 'pt_y'
]

# load camera data for pySBA
cam_array = pySBA.getCameraArray(cam_ids)

# intialize the 3D world points variable
# for roughly the right orientation and scale (red cam at (1, 1) and green cam at (-1, -1))
# we can take the red cam points and center/normalize them
points_3d = np.zeros(shape=(n_pts, 3))
red_pts = pts[:, :, 0].copy()
norm_x = (red_pts[:, 0] - np.min(red_pts[:, 0]))
norm_x = norm_x - (np.max(norm_x)/2)
norm_x = norm_x / np.max(norm_x)
norm_y = (red_pts[:, 1] - np.min(red_pts[:, 1]))
norm_y = norm_y - (np.max(norm_y)/2)
norm_y = norm_y / np.max(norm_y)
points_3d[:, 0] = norm_x
points_3d[:, 1] = norm_y

# plot the 2D points
fig, axs = plt.subplots(n_cams, 1, sharex=True)
plt.title('2D points found on all cameras')
for i in range(n_cams):
    colors = np.linspace(0, 1, n_pts)
    axs[i].scatter(pts[:, 0, i], pts[:, 1, i],\
                    s=10, c=colors, alpha=0.5)
    axs[i].title.set_text(f'{cam_ids[i]}')
    axs[i].invert_yaxis()
plt.show()

# create indexing variables
camera_ind = np.zeros(shape=(n_pts*n_cams,), dtype=int)
point_ind = np.zeros(shape=(n_pts*n_cams,), dtype=int)
points_2d = np.zeros(shape=(n_pts*n_cams, 2), dtype=float)
for i in range(n_cams):
    for j in range(n_pts):
        ind = (i*n_pts) + j
        camera_ind[ind] = i
        point_ind[ind] = j
        points_2d[ind, :] = pts[j, :, i].copy()


"""
Initialize the SBA object with points and calibration
(using an old calibration or just general ballpark calculated manually). 
"""
sba = pySBA.PySBA(cam_array, points_3d, points_2d, camera_ind, point_ind)

# display the initial camera params
x = PrettyTable()
x.field_names = cam_array_fields.copy()
for row in sba.cameraArray:
    x.add_row(np.round(row, 2))
print(x)

"""
Optimize for the 3d positions holding all camera parameters fixed
"""
sba.bundleAdjust_nocam()

# plot the reprojection error
r = sba.project(sba.points3D[sba.point2DIndices], sba.cameraArray[sba.cameraIndices]) - sba.points2D
r = np.sqrt(np.sum(r**2, axis=1))
plt.hist(r[r < np.percentile(r, 99)])
plt.xlabel('reprojection error')
plt.title('no param adjustment')
plt.show()

"""
TRY THIS:
Optimize for the camera parameters holding known 3D positions fixed
"""

"""
Given the updated 3d positions jointly optimize the camera parameters and 3d positions to minimize reconstruction errors.  
Use sba.bundleAdjust() if you want each camera to have separate intrinsics.
sba.bundleAdjust_sharedcam() uses shared intrinsics but with different image centroids used for radial distortion.
"""
sba.bundleAdjust_sharedcam()
# sba.bundleAdjust()
opt_cam_array = sba.cameraArray.copy()

# display the updated camera params
x = PrettyTable()
x.field_names = cam_array_fields.copy()
for row in sba.cameraArray:
    x.add_row(np.round(row, 2))
print(x)

# plot the reprojection error
r = sba.project(sba.points3D[sba.point2DIndices], sba.cameraArray[sba.cameraIndices]) - sba.points2D
r = np.sqrt(np.sum(r**2, axis=1))
plt.hist(r[r < np.percentile(r, 99)])
plt.xlabel('reprojection error')
plt.title('shared intrinsics')
plt.show()

"""
Rescale translation vector to a standard value
"""
targScale = 1.5
thisScale = np.mean(np.sqrt(np.sum(opt_cam_array[:,3:6]**2, axis=1)))
opt_cam_array[:, 3:6] = opt_cam_array[:, 3:6] * (targScale/thisScale)
sba_scaled = pySBA.PySBA(opt_cam_array, sba.points3D, sba.points2D,\
                            sba.cameraIndices, sba.point2DIndices)
sba_scaled.bundleAdjust_nocam()

# display the updated camera params
x = PrettyTable()
x.field_names = cam_array_fields.copy()
for row in sba_scaled.cameraArray:
    x.add_row(np.round(row, 2))
print(x)

# plot the reprojection error
r = sba_scaled.project(sba_scaled.points3D[sba_scaled.point2DIndices], sba_scaled.cameraArray[sba_scaled.cameraIndices]) - sba_scaled.points2D
r = np.sqrt(np.sum(r**2, axis=1))
plt.hist(r[r < np.percentile(r, 99)])
plt.xlabel('reprojection error')
plt.title('rescaled extrinsics')
plt.show()

# plot the 2D points
fig, axs = plt.subplots(1, n_cams, sharey=True)
plt.title('updated 2D point locations')
for i in range(n_cams):
    ind2d = sba.cameraIndices==i
    colors = np.linspace(0, 1, n_pts)
    axs[i].scatter(sba.points2D[ind2d, 0],\
                    sba.points2D[ind2d, 1],\
                    s=10, c=colors, alpha=0.5)
    axs[i].title.set_text(f'{cam_ids[i]}')
    axs[i].invert_yaxis()
plt.show()

"""
Save the optimized camera array
"""
opt_array_file = f"{tmp_local_root}/calibration_files/{img_date}_opt_cam_array.npy"
np.save(opt_array_file, opt_cam_array)