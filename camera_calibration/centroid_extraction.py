import os 
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage import measure
from scipy.ndimage import gaussian_filter

def get_image_folders(calibration_path):
    # find the folder of images associated with each camera
    cam_folders = []
    for d in os.listdir(calibration_path):
        if os.path.isdir(os.path.join(calibration_path, d)):
            cam_folders.append(d)
    cam_folders = sorted(cam_folders)
    print(cam_folders)

    return cam_folders

def get_centroid_info(img, img_label, thresh=0.4):
    '''
    Processes each image to extract the number of points for each image.
    If there is one clean point, computes the weighted centroid and area of that point.

    Params:
    -------
    img : flattened image; shape (n_rows, n_columns)
    img_label : string defining the image (e.g. [camera]_[image num])
    thresh : pixel value below which the normalized image is set to 0

    Returns:
    -------
    the number of centroids, and if 1 centroid, its size and xy location for each image
    '''
    # normalize and threshold the image
    norm_img = (img - np.min(img)) / np.max(img - np.min(img))
    filt_img = gaussian_filter(norm_img, sigma=1)
    blobs_binary = filt_img > thresh
    
    # extract continuous blobs
    blob_idx, num_blobs = measure.label(blobs_binary, background=0, return_num=True)
    
    # only proceed if there is just 1 blob
    if num_blobs == 0:
        # print(f"{img_label} - no region found")
        area = np.nan
        centroid = [np.nan, np.nan]
    elif num_blobs > 1:
        # print(f"{img_label} - more than one region found")
        area = np.nan
        centroid = [np.nan, np.nan]

    # get the area and the weighted centroid
    elif num_blobs == 1:
        props = measure.regionprops(blob_idx)
        area = props[0].area
        
        # make a grid to compute the weights
        r,c = np.shape(img)
        r_ = np.linspace(0, r, r+1)
        c_ = np.linspace(0, c, c+1)
        x_m, y_m = np.meshgrid(c_, r_, sparse=False, indexing='xy')
        
        # extract an ROI around the blob
        (y_min, x_min, y_max, x_max) = props[0].bbox
        weights = norm_img[y_min:y_max, x_min:x_max]
        grid_x = x_m[y_min:y_max, x_min:x_max]
        grid_y = y_m[y_min:y_max, x_min:x_max]
        
        # get the weighted centroid
        weighted_x = weights * grid_x
        weighted_y = weights * grid_y
        centroid = np.zeros(2)
        centroid[1] = np.sum(weighted_x) / np.sum(weights)
        centroid[0] = np.sum(weighted_y) / np.sum(weights)
        
    return num_blobs, centroid, area

def plot_centroid(img, centroid, img_label):
    x_start = (centroid[1] - 20).astype(int)
    x_end = (centroid[1] + 20).astype(int)
    y_start = (centroid[0] - 20).astype(int)
    y_end = (centroid[0] + 20).astype(int)

    # normalize the image
    norm_img = (img - np.min(img)) / np.max(img - np.min(img))

    # plot it
    plt.imshow(norm_img, 
                cmap='gray', aspect='auto')
    plt.scatter(centroid[1], centroid[0], 
                marker="+", c="red", s=10, lw=0.5)
    plt.xlim(x_start, x_end)
    plt.ylim(y_start, y_end)
    plt.title(f"{img_label} (normalized)")
    plt.show()

# get the centroid info for each camera
def extract_centroids(calibration_path, cam_folder,\
                        laser_rgb_idx=1, show_plots=True):
    # get the camera ID
    parts = cam_folder.split('_')
    cam_id = f"{parts[0]}_{parts[1]}"
    
    # get the image files and sort them
    folder_path = os.path.join(calibration_path, cam_folder)
    image_files = []
    for f in os.listdir(folder_path):
        image_files.append(f)
    image_files = sorted(image_files)
    if image_files[0] == 'Thumbs.db':
        image_files = image_files[1:]
    n_images = len(image_files)
    print(f"processing {n_images} images for {cam_id}...")

    # get the size, xy location, and number of centroids for each image
    areas = np.empty(shape=(n_images,))
    centroids = np.empty(shape=(n_images, 2))
    blob_count = np.empty(shape=(n_images,))
    for i, img_file in enumerate(image_files):
        # load the image and take a slice if needed
        img_path = folder_path + '/' + img_file
        img = mpimg.imread(img_path)
        if len(img.shape) > 2:
            img = img[:, :, laser_rgb_idx]            
        
        # extract the centroid info and store it
        blob_count[i], centroids[i], areas[i] = get_centroid_info(img, img_file)

        # plot to check
        if show_plots:
            if i/1000 == i//1000:
                if blob_count[i] == 1:
                    plot_centroid(img, centroids[i], img_file)

    # print the number of good images with 1 centroid
    n_bad_centroids = np.sum(np.isnan(areas))
    print(f"{n_bad_centroids} excluded")
    print(f"{cam_id}: found {n_images-n_bad_centroids} well-defined centroids\n\n")

    return cam_id, areas, centroids, blob_count