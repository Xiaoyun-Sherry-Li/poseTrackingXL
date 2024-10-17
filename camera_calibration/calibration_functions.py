import numpy as np
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from scipy.io import savemat
import pySBA

''' Optimize extrinsics only holding known point locations fixed '''
def optimize_extrinsics(known_pt_folder, \
                        cam_array, cam_ids, \
                        save_figs, save_files, \
                        show_plots=True, save_plots=True):
    # known points are defined by the date they were acquired
    img_date = input("input date of known point image acquisition (YYMMDD): ")
    
    # load the 3D world coordinates for the known points
    pts_3d_known = np.load(f'{known_pt_folder}points_3d_simple.npy')
    # pts_3d_known = np.load(f'{known_pt_folder}{img_date}_points_3d.npy')

    # load the corresponding 2D points (hand labeled for each camera using Label3D)
    pts_2d_known = np.load(f'{known_pt_folder}{img_date}_points_2d.npy')
    n_pts = pts_2d_known.shape[0]

    # load the camera and point indexing variables
    known_pts_camera_idx = np.load(f'{known_pt_folder}{img_date}_camera_ind.npy')
    known_pts_idx = np.load(f'{known_pt_folder}{img_date}_point_ind.npy')
    known_pts_camera_idx = np.squeeze(known_pts_camera_idx.astype(int))
    known_pts_idx = np.squeeze(known_pts_idx.astype(int))

    # label the fields for the camera array
    cam_array_fields = [
                        'rot_1', 'rot_2', 'rot_3',
                        'trans_1', 'trans_2', 'trans_3',
                        'focal dist', 'distort_1', 'distort_2',
                        'pt_x', 'pt_y'
                        ]
    
    # initialize the bundle adjustment object
    sba = pySBA.PySBA(cam_array, pts_3d_known, pts_2d_known, 
                      known_pts_camera_idx, known_pts_idx)

    # display the initial camera params
    print('\ninitial camera params')
    display_cam_array(sba, cam_array_fields)
    
    if show_plots:
        # show the 3D points
        known_colors = ['black', 'red', 'yellow', 'green', 'blue', 
                        'red', 'yellow', 'green', 'blue']

        f, ax = plot_3D_points(pts_3d=pts_3d_known,
                                colors=known_colors,
                                title=f'world coords, init pts\nred cam at roughly (2, 2)',
                                pt_size=100)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-0.5, 0.5)
        plt.show()

        if save_plots:
            f.savefig(f'{save_figs}init_pts_3d.png', dpi=600, bbox_inches='tight')
            
        # compare the initial reprojected points to the selected points
        # reprojection points are colored by error
        titles = []
        titles.append('reprojection w/ original camera params')
        titles.append('triangulated 2D points')

        f, ax = plot_reproj_points(sba, titles, cam_ids, pt_size=30)
        if save_plots:
            f.savefig(f'{save_figs}init_pts_2d.png', dpi=600, bbox_inches='tight')
            
    # optimize the camera extrinsics, holding the intrinsics fixed, using known 3D points
    sba.bundleAdjust_just_extrinsics()

    # plot the reprojection error
    f, ax = plot_reproj_error(sba, title='only extrinsics adjusted (known points)', n_bins=50)
    if save_plots:
        f.savefig(f'{save_figs}reproj_error_step1.png', dpi=600, bbox_inches='tight')
        
    # display the updated camera params
    print('\nupdated camera params')
    display_cam_array(sba, cam_array_fields)
    
    if show_plots:
        # compare the new reprojected points to the selected points
        titles = []
        titles.append('optimized extrinsics, fixed 3D pts')
        titles.append('triangulated 2D points')

        f, ax = plot_reproj_points(sba, titles, cam_ids, pt_size=30)
        if save_plots:
            f.savefig(f'{save_figs}known_3d_pts.png', dpi=600, bbox_inches='tight')

    # save the updated camera array
    opt_cam_array = sba.cameraArray
    opt_array_file = f"{save_files}opt_cam_array_extrinsics_only.npy"
    np.save(opt_array_file, opt_cam_array)
    
    return opt_cam_array


''' Optimize the 3D world coordinates of the laser points holding the camera params fixed '''
def optimize_3d_coords(centroid_dir, \
                       cam_array, cam_ids, \
                       save_figs, save_files, \
                       init_3D='zeros_init', \
                       show_plots=True, save_plots=True):
    '''
    init_3D : string; how to initialize the 3D coordinates of the laser points
                'zeros_init' (default) uses an array of zeros
                'estimated' uses an estimated 3D array created in extract_preprocess_centroids.ipynb
                'optimized' uses the optimized 3D points saved in the save_files folder
    '''
    # load the preprocessed centroid locations and indices
    img_date = input("input date of calibration video acquisition (YYMMDD): ")

    file_centroids = f'{img_date}_centroids.npy'
    points_2d = np.load(f'{centroid_dir}{file_centroids}')

    file_camera_ind = f'{img_date}_camera_ind.npy'
    camera_ind = np.load(f'{centroid_dir}{file_camera_ind}')

    file_point_ind = f'{img_date}_point_ind.npy'
    point_ind = np.load(f'{centroid_dir}{file_point_ind}')
    
    n_pts = np.max(point_ind) + 1
    print(f'analyzing {n_pts} centroids')
    
    # initialize the 3D points
    if init_3D=='zeros_init':
        points_3d = np.zeros(shape=(n_pts, 3))
    elif init_3D=='z_estimated':
        file_init_3d = f'{img_date}_init_3d.npy'
        points_3d = np.load(f'{centroid_dir}{file_init_3d}')
        points_3d[:, :2] = np.zeros(shape=(n_pts, 2))
    elif init_3D=='optimized':
        points_3d = np.load(f"{save_files}opt_3d_points.npy")
    else:
        print('Warning: 3D initialization param not recognized! Initializing with zeros.')
        points_3d = np.zeros(shape=(n_pts, 3))
        
    # label the fields for the camera array
    cam_array_fields = [
                        'rot_1', 'rot_2', 'rot_3',
                        'trans_1', 'trans_2', 'trans_3',
                        'focal dist', 'distort_1', 'distort_2',
                        'pt_x', 'pt_y'
                        ]
    
    # initialize the bundle adjustment object
    sba = pySBA.PySBA(cam_array, points_3d, points_2d, 
                      camera_ind, point_ind)

    # display the initial camera params
    print('\ninitial camera params')
    display_cam_array(sba, cam_array_fields)
    
    if show_plots:
        # compare the initial reprojected points to the centroid locations
        # reprojection points are colored by error
        titles = []
        titles.append('reprojection w/ initial camera params')
        titles.append('2D centroid locations')

        f, ax = plot_reproj_points(sba, titles, cam_ids, pt_size=5)
        if save_plots:
            f.savefig(f'{save_figs}original_pts.png', dpi=600, bbox_inches='tight')
            
    # Optimize the 3d point locations holding all camera parameters fixed
    sba.bundleAdjust_nocam()

    # plot the reprojection error
    f, ax = plot_reproj_error(sba, title='point coordinates adjusted')
    if save_plots:
        f.savefig(f'{save_figs}reproj_error_step2.png', dpi=600, bbox_inches='tight')
    f, ax = plot_reproj_error(sba, title='point coordinates adjusted', zoom=True)
    if save_plots:
        f.savefig(f'{save_figs}reproj_error_step2_zoom.png', dpi=600, bbox_inches='tight')
        
    # display the updated camera params
    print('\nupdated camera params')
    display_cam_array(sba, cam_array_fields)
    
    if show_plots:
        # compare the new reprojected points to the selected points
        titles = []
        titles.append('optimized 3D pts, fixed camera params')
        titles.append('2D centroid locations')
        f, ax = plot_reproj_points(sba, titles, cam_ids, pt_size=5)
        if save_plots:
            f.savefig(f'{save_figs}cam_params_fixed.png', dpi=600, bbox_inches='tight')

        # plot the updated 3D point locations
        pts_3d = sba.points3D
        n_pts = pts_3d.shape[0]
        colors = np.linspace(0, 1, n_pts)
        f, ax = plot_3D_points(pts_3d=pts_3d,
                                 colors=colors,
                                 title=f'world coords, init pts\nred cam at roughly (2, 2)',
                                 pt_size=5
                                )
        ax.view_init(azim=45, elev=15)
        plt.show()
        if save_plots:
            f.savefig(f'{save_figs}updated_pts_3d.png', dpi=600, bbox_inches='tight')
            
    
    # get the updated 3D points locations
    opt_3d_pts = sba.points3D

    # save the updated camera array
    opt_cam_array = sba.cameraArray
    opt_array_file = f"{save_files}opt_cam_array_pts_only.npy"
    np.save(opt_array_file, opt_cam_array)
    
    return opt_cam_array, opt_3d_pts


'''
Jointly optimize the camera parameters and 3D positions to minimize reconstruction errors.
Optionally exclude points with high error and re-optimize.
'''
def optimize_pts_cam(centroid_dir, \
                     cam_array, cam_ids, \
                     save_figs, save_files, \
                     init_3D='zeros_init', \
                     sharedcam=True, remove_high_error=False, \
                     show_plots=True, save_plots=True
                    ):
    # load the preprocessed centroid locations and indices
    img_date = input("input date of calibration video acquisition (YYMMDD): ")

    file_centroids = f'{img_date}_centroids.npy'
    points_2d = np.load(f'{centroid_dir}{file_centroids}')

    file_camera_ind = f'{img_date}_camera_ind.npy'
    camera_ind = np.load(f'{centroid_dir}{file_camera_ind}')

    file_point_ind = f'{img_date}_point_ind.npy'
    point_ind = np.load(f'{centroid_dir}{file_point_ind}')

    n_pts = np.max(point_ind) + 1
    print(f'analyzing {n_pts} centroids')

    # initialize using the optimized 3D point locations
    if init_3D=='zeros_init':
        points_3d = np.zeros(shape=(n_pts, 3))
    elif init_3D=='z_estimated':
        file_init_3d = f'{img_date}_init_3d.npy'
        points_3d = np.load(f'{centroid_dir}{file_init_3d}')
        points_3d[:, :2] = np.zeros(shape=(n_pts, 2))
    elif init_3D=='optimized':
        points_3d = np.load(f"{save_files}opt_3d_points.npy")
    else:
        print('Warning: 3D initialization param not recognized! Initializing with zeros.')
        points_3d = np.zeros(shape=(n_pts, 3))

    # label the fields for the camera array
    cam_array_fields = [
                        'rot_1', 'rot_2', 'rot_3',
                        'trans_1', 'trans_2', 'trans_3',
                        'focal dist', 'distort_1', 'distort_2',
                        'pt_x', 'pt_y'
                        ]
    # initialize the bundle adjustment object
    sba = pySBA.PySBA(cam_array, points_3d, points_2d,
                      camera_ind, point_ind)

    # display the initial camera params
    print('\ninitial camera params')
    display_cam_array(sba, cam_array_fields)

    # jointly optimize the 3D points and camera parameters
    if sharedcam:
        sba.bundleAdjust_sharedcam()
    else:
        sba.bundleAdjust()
        
    # plot the reprojection error
    f, ax = plot_reproj_error(sba, title='camera params and point locations adjusted')
    if save_plots:
        f.savefig(f'{save_figs}reproj_error_step3.png', dpi=600, bbox_inches='tight')

    # plot without the 99th percentile and near zero values
    f, ax = plot_reproj_error(sba, title='camera params and point locations adjusted', zoom=True)
    if save_plots:
        f.savefig(f'{save_figs}reproj_error_step3_zoom.png', dpi=600, bbox_inches='tight')

    # display the updated camera params
    print('\nupdated camera params')
    display_cam_array(sba, cam_array_fields)

    if show_plots:
        # compare the new reprojected points to the selected points
        titles = []
        titles.append('optimized 3D pts and camera params')
        titles.append('2D centroid locations')
        f, ax = plot_reproj_points(sba, titles, cam_ids, pt_size=5)
        if save_plots:
            f.savefig(f'{save_figs}optimized_pts_2d.png', dpi=600, bbox_inches='tight')

        # plot the updated 3D point locations
        pts_3d = sba.points3D
        n_pts = pts_3d.shape[0]
        colors = np.linspace(0, 1, n_pts)
        f, ax = plot_3D_points(pts_3d=pts_3d,
                                 colors=colors,
                                 title=f'world coords\nred cam at roughly (2, 2)',
                                 pt_size=5
                                )
        plt.show()
        if save_plots:
            f.savefig(f'{save_figs}optimized_pts_3d.png', dpi=600, bbox_inches='tight') 

    # Optionally exclude points with high error
    if remove_high_error:
        print(f"\n\nfiltering out points with greater than 99th percentile error...")

        # get the optimized camera array and 3D point locations
        opt_cam_array = sba.cameraArray
        opt_3d_pts = sba.points3D

        # calculate the reprojection error
        est_pts = sba.project(sba.points3D[sba.point2DIndices], \
                                sba.cameraArray[sba.cameraIndices])
        r = est_pts - sba.points2D
        reproj_error = np.sqrt(np.sum(r**2, axis=1)) # reprojection error

        # get the indices for points with high reprojection error
        r_thresh = np.percentile(reproj_error, 99)
        trouble_pts = np.unique(point_ind[reproj_error > r_thresh])

        # get the 3D low error points
        good_idx_3d = np.setdiff1d(np.unique(point_ind), trouble_pts)
        filtered_pts_3d = opt_3d_pts[good_idx_3d]
        n_pts = opt_3d_pts.shape[0]
        n_good_pts = filtered_pts_3d.shape[0]

        # total number of good centroids in view across the 4 cameras
        n_good_total = points_2d.shape[0] - point_ind[reproj_error > r_thresh].shape[0]
        n_now_total = 0
        ind_end=0

        # get the indices and 2D points for low error points
        filtered_point_ind = np.zeros(shape=[n_good_total,], dtype=int)
        filtered_camera_ind = np.zeros(shape=[n_good_total,], dtype=int)
        filtered_pts_2d = np.zeros(shape=[n_good_total, 2], dtype=float)
        for i in range(n_cams):
            # get the good point index for this camera
            ind2d = camera_ind==i
            cam_pt_idx = point_ind[ind2d]
            cam_pts = points_2d[ind2d]
            _, idx1, idx2 = np.intersect1d(good_idx_3d, cam_pt_idx, return_indices=True)
            
            # set the index range
            n_pts_now = idx1.shape[0]
            n_now_total += n_pts_now
            ind_start = ind_end
            ind_end = ind_start + n_pts_now
            
            # get the points values and indices
            filtered_point_ind[ind_start:ind_end] = idx1
            filtered_camera_ind[ind_start:ind_end] = i
            filtered_pts_2d[ind_start:ind_end] = cam_pts[idx2].squeeze()

        # remove unused slots (not sure why these totals are different)    
        filtered_point_ind = filtered_point_ind[:n_now_total].astype(int)
        filtered_camera_ind = filtered_camera_ind[:n_now_total].astype(int)
        filtered_pts_2d = filtered_pts_2d[:n_now_total]

        print(f'filtered out {trouble_pts.shape[0]} points with error > {np.round(r_thresh, 2)}')
        print(f'{n_good_pts} points remain')

        # re-initialize the bundle adjustment object with the filtered variables
        sba = pySBA.PySBA(opt_cam_array,
                          filtered_pts_3d, filtered_pts_2d,
                          filtered_camera_ind, filtered_point_ind)

        # sanity check - plot the reprojection error between the filtered 3D and 2D points
        f, ax = plot_reproj_error(sba, title='high error points removed')

        # ensure filtering isn't deleting points from just one part of the FOV
        if show_plots:
            # compare the filtered reprojected points to the selected points
            titles = []
            titles.append('high error points removed')
            titles.append('2D centroid locations')
            f, ax = plot_reproj_points(sba, titles, cam_ids, pt_size=5)
            
            # plot the filtered 3D point locations
            n_pts = filtered_pts_3d.shape[0]
            colors = np.linspace(0, 1, n_pts)
            f, ax = plot_3D_points(pts_3d=filtered_pts_3d,
                                     colors=colors,
                                     title=f'world coords, high error removed\nred cam at roughly (2, 2)',
                                     pt_size=5
                                    )
            plt.show()

        # jointly optimize the 3D points and camera parameters
        if sharedcam:
            sba.bundleAdjust_sharedcam()
        else:
            sba.bundleAdjust()
            
        # plot the reprojection error
        f, ax = plot_reproj_error(sba, title='camera params and point locations adjusted')
        if save_plots:
            f.savefig(f'{save_figs}reproj_error_step3-5.png', dpi=600, bbox_inches='tight')
            
        # plot without the 99th percentile
        f, ax = plot_reproj_error(sba, title='camera params and point locations adjusted', zoom=True)
        if save_plots:
            f.savefig(f'{save_figs}reproj_error_step3-5_zoom.png', dpi=600, bbox_inches='tight')

        # display the updated camera params
        print('\nupdated camera params')
        display_cam_array(sba, cam_array_fields)

        if show_plots:
            # compare the new reprojected points to the selected points
            titles = []
            titles.append('optimized 3D pts and camera params')
            titles.append('2D centroid locations')
            f, ax = plot_reproj_points(sba, titles, cam_ids, pt_size=5)
            if save_plots:
                f.savefig(f'{save_figs}optimized_pts_2d.png', dpi=600, bbox_inches='tight')

            # plot the updated 3D point locations
            pts_3d = sba.points3D
            n_pts = pts_3d.shape[0]
            colors = np.linspace(0, 1, n_pts)
            f, ax = plot_3D_points(pts_3d=pts_3d,
                                     colors=colors,
                                     title=f'world coords, init pts\nred cam at roughly (2, 2)',
                                     pt_size=5
                                    )
            plt.show()
            if save_plots:
                f.savefig(f'{save_figs}optimized_pts_3d.png', dpi=600, bbox_inches='tight')

    # get the updated 3D points locations
    opt_3d_pts = sba.points3D

    # save the updated camera array
    opt_cam_array = sba.cameraArray
    opt_array_file = f"{save_files}opt_cam_array_pts_cam.npy"
    np.save(opt_array_file, opt_cam_array)

    return opt_cam_array, opt_3d_pts

''' Utils '''
def display_cam_array(sba, cam_array_fields):
    x = PrettyTable()
    x.field_names = cam_array_fields
    for row in sba.cameraArray:
        x.add_row(np.round(row, 2))
    print(x)

def save_cam_array_matlab(cam_array, save_files):
    cam_dict = {"optCamArray":cam_array}
    date_id = input("input today's date (YYMMDD): ")
    savemat(f"{save_files}{date_id}_opt_cam_array.mat", cam_dict)

def sanity_check_label3D_pts(cam_array, cam_ids, known_pt_folder):
    # known points are defined by the date they were acquired
    img_date = input("input date of known point image acquisition (YYMMDD): ")

    # load the 3D world coordinates for the known points
    pts_3d_known = np.load(f'{known_pt_folder}{img_date}_points_3d.npy')

    # load the corresponding 2D points (hand labeled for each camera using Label3D)
    pts_2d_known = np.load(f'{known_pt_folder}{img_date}_points_2d.npy')
    n_pts = pts_2d_known.shape[0]

    # load the camera and point indexing variables
    known_pts_camera_idx = np.load(f'{known_pt_folder}{img_date}_camera_ind.npy')
    known_pts_idx = np.load(f'{known_pt_folder}{img_date}_point_ind.npy')
    known_pts_camera_idx = np.squeeze(known_pts_camera_idx.astype(int))
    known_pts_idx = np.squeeze(known_pts_idx.astype(int))

    # label the fields for the camera array
    cam_array_fields = [
                        'rot_1', 'rot_2', 'rot_3',
                        'trans_1', 'trans_2', 'trans_3',
                        'focal dist', 'distort_1', 'distort_2',
                        'pt_x', 'pt_y'
                        ]

    # initialize the bundle adjustment object
    sba = pySBA.PySBA(cam_array, pts_3d_known, pts_2d_known, 
                      known_pts_camera_idx, known_pts_idx)

    # plot the reprojection error and reprojections
    f, ax = plot_reproj_error(sba, title='reprojection of Label3D 3D points onto 2D',\
                                n_bins=20, zoom=False)
    plt.show()
    
    titles = ['Label3D 3D pts proj.', 'Label3D 2D pts']
    f, ax = plot_reproj_points(sba, titles, cam_ids, pt_size=30)
    plt.show()

''' Figures '''
# plot the points in 3D
def plot_3D_points(pts_3d, colors, title, pt_size=5):
    fig = plt.figure(figsize=(8, 4))
    ax = plt.axes([0, 0, .6, 1.2], projection='3d')
    x_pts = pts_3d[:, 0].copy()
    y_pts = pts_3d[:, 1].copy()
    z_pts = pts_3d[:, 2].copy()

    ax.scatter(
            x_pts, y_pts, z_pts,
            c=colors, alpha=0.5, lw=0, s=pt_size)
    # ax.set_xlim(-1, 1)
    # ax.set_ylim(-1, 1)
    ax.set_zlim(-0.5, 0.5)

    ax.set_xlabel('world x')
    ax.set_ylabel('world y')
    ax.set_zlabel('world z')

    ax.view_init(azim=45, elev=15)
    ax.set_title(title)

    return fig, ax

# plot the points in 2D
def plot_2D_points(fig, axs, points_2d,\
                    camera_ind, cam_ids, \
                    title, colors, col_idx=0, pt_size=5):
    n_cams = axs.shape[0]

    for i in range(n_cams):
        # data params
        ind2d = camera_ind==i
        cam_points = points_2d[ind2d].squeeze()
        n_pts = cam_points.shape[0]
        
        # plot the points
        axs[i, col_idx].scatter(cam_points[:, 0],
                       cam_points[:, 1],
                       s=pt_size, lw=0, 
                       c=colors[ind2d], alpha=0.5)
        axs[i, col_idx].invert_yaxis()
        if i == 0:
            axs[i, col_idx].title.set_text(f'{title}\n{cam_ids[i]}')
        else:
            axs[i, col_idx].title.set_text(f'{cam_ids[i]}') 

    return fig, axs

# compare the reprojected and original points
def plot_reproj_points(sba, titles, cam_ids, pt_size=5):
    # fig params
    n_cams = np.max(sba.cameraIndices) + 1
    fig, axs = plt.subplots(n_cams, 2, sharex=True, figsize=(8, 8))

    # data params
    camera_ind = sba.cameraIndices
    est_pts = sba.project(sba.points3D[sba.point2DIndices], \
                            sba.cameraArray[sba.cameraIndices])
    r = est_pts - sba.points2D
    reproj_error = np.sqrt(np.sum(r**2, axis=1)) # reprojection error
    
    # plot the reprojected points
    fig, axs = plot_2D_points(fig, axs,
                                points_2d=est_pts,
                                camera_ind=camera_ind,
                                cam_ids=cam_ids,
                                title=titles[0],
                                colors=reproj_error,
                                col_idx=0, pt_size=pt_size)

    # plot the original points
    colors = np.zeros(sba.points2D.shape[0])
    for i in range(n_cams):
        ind2d = camera_ind==i
        n_pts = np.sum(ind2d)
        colors[ind2d] = np.linspace(0, 1, n_pts)
    fig, axs = plot_2D_points(fig, axs,
                                points_2d=sba.points2D,
                                camera_ind=camera_ind,
                                cam_ids=cam_ids,
                                title=titles[1],
                                colors=colors,
                                col_idx=1, pt_size=pt_size)
    plt.show()
    return fig, axs

# plot the reprojection error
def plot_reproj_error(sba, title, n_bins=100, zoom=False):
    r = sba.project(sba.points3D[sba.point2DIndices], sba.cameraArray[sba.cameraIndices]) - sba.points2D
    r = np.sqrt(np.sum(r**2, axis=1))

    fig, ax = plt.subplots(1, 1)

    # plot reprojection error
    # for better visualization, optionally exclude 99th percentile and near 0 values
    if zoom:
        r_zoom = r[r < np.percentile(r, 99)]
        r_zoom = r_zoom[r_zoom > 2e-6]
        ax.hist(r_zoom, bins=n_bins)
    else: 
        ax.hist(r, bins=n_bins)

    # indicate the 99th percentile
    ymin, ymax = ax.get_ylim()
    ax.vlines(np.percentile(r, 99), ymin, ymax, \
                colors='red', linestyles='dashed')

    ax.set_xlabel('reprojection error (pixels)')
    ax.set_title(f'{title}')
    plt.show()

    return fig, ax