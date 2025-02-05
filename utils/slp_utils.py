#%%
import numpy as np
import csv
import cv2

import sleap
from sleap import Labels, Video, LabeledFrame, Instance, Skeleton
from sleap import load_model as slp_load
from sleap.instance import Point
from sleap.skeleton import Node
import scipy.io
import matplotlib.pyplot as plt
import os
#%%


import sys
sys.path.append(".//camera_calibration/")
sys.path.append("..//camera_calibration/")
sys.path.append("c:\\Users\\xl313\\OneDrive\\Documents\\GitHub\\poseTrackingXL")
from pySBA import unconvertParams, PySBA
sys.path.append(".//utils/")
sys.path.append("..//utils/")
from triangulation_utils import unDistortPoints, camera_matrix, triangulate_confThresh_lowestErr

def create_slp_project(vid_path, skeleton_file, keypoints, slp_labels_file):
    '''
    Takes in video, 2D keypoints, and skeleton nodes/edges and creates a SLP project.
    Basic protocol from:
        https://github.com/talmolab/sleap/discussions/1534

    Params
    ------
    vid_path : path to numpy array of images; array shape (total_frames, image_h, image_w, 1)
    skeleton_file : string, path to csv file with skeleton nodes and edges
    keypoints : ndarray, shape (total_frames, n_nodes, 2)
    slp_labels_file : string, path to save slp project and file name (.slp)

    '''
    # data params
    n_frames = keypoints.shape[0]

    # create a video object
    video = Video.from_filename(vid_path)

    # get the skeleton info
    nodes = []
    edges = []
    symmetries = []

    with open(f"{skeleton_file}") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if len(row['name']) == 0:
                break

            # add the node
            node = row['name']
            nodes.append(node)

            # if applicable, add an edge
            if len(row['parent']) > 0:
                parent = row['parent']
                edges.append([node, parent])

            # if applicable, add symmetry constraints
            if row['swap'] is not None:
                if len(row['swap']) > 0:
                    swap = row['swap']
                    symmetries.append([node, swap])

    # create a skeleton
    skeleton = Skeleton(name="bird")
    skeleton.add_nodes(nodes)
    for edge in edges:
        skeleton.add_edge(edge[0], edge[1])
    for symmetry in symmetries:
        try:
            skeleton.add_symmetry(symmetry[0], symmetry[1])
        except ValueError as inst:
            # will raise a value error if the symmetry already exists
            print(inst)

    # make a list of labeled frames
    labeled_frames = []

    for f in range(n_frames):
        # create an instance
        instances = []
        points = dict()
        for i, node in enumerate(nodes):
            pt_x = keypoints[f, i, 0]
            pt_y = keypoints[f, i, 1]
            points[node] = Point(pt_x, pt_y)
        instance = Instance(skeleton=skeleton, points=points)
        instances.append(instance) # not sure if this needs to be a list to be interpreted 

        # label the frame with that instance
        frame = LabeledFrame(video=video, frame_idx=f, instances=instances)
        labeled_frames.append(frame)

    # create the labels object and save it to a SLP file
    labels_obj = Labels(labeled_frames)
    labels_obj.add_video(video)
    labels_obj.save(slp_labels_file)


def resize_and_pad_rows(img, ds_size):
    '''
    Downsamples by interpolating across ds_size pixels
    appends rows at bottom of image if needed to maintain aspect ratio
    '''
    imShape = img.shape
    if imShape[1] / imShape[0] == ds_size[0] / ds_size[1]: # if aspect ratio is the same
        ds_img = cv2.resize(img, ds_size, interpolation=cv2.INTER_AREA)
    else: # pad rows to preserve aspect ratio
        ds_fac = imShape[1] / ds_size[0]
        ds_row = int(np.round(imShape[0] / ds_fac))
        ds_img = cv2.resize(img, (ds_size[0], ds_row), interpolation=cv2.INTER_AREA)
        fill_row = np.full((ds_size[1]-ds_row, ds_size[0]), 0, dtype='uint8')
        ds_img = np.concatenate((ds_img,fill_row), axis=0)
    return ds_img

# def crop_from_com(img, centroid, half_width, crop_size = (500,500)):
#     '''
#     Crops an image around a given centroid (crop dims defined by half_width)
#     and resizes to the specified crop_size.
#     '''
#     ctr = np.round(centroid).astype(int)
#     half_width = np.round(half_width).astype(int)
#     img_h,img_w = img.shape[0],img.shape[1]
#     xmin = np.min([np.max([ctr[0] - half_width, 0]), img_w - 100])
#     xmax = np.max([np.min([ctr[0] + half_width + 100, img_w]), 100])
#     ymin = np.min([np.max([ctr[1] - half_width, 0]), img_h - 100])
#     ymax = np.max([np.min([ctr[1] + half_width + 100, img_h]), 100])
#
#     # crop_img = cv2.resize(img[ymin:ymax, xmin:xmax], crop_size, cv2.INTER_AREA)
#     # ctr = np.round(body_ctr).astype(int)
#     # half_width = np.round(half_width).astype(int)
#     # img_h,img_w = img.shape[0],img.shape[1]
#     # xmin = np.min([np.max([ctr[0] - half_width - 100, 0]), img_w - 100])
#     # xmax = np.max([np.min([ctr[0] + half_width + 100, img_w]), 100])
#     # ymin = np.min([np.max([ctr[1] - half_width -100, 0]), img_h - 100])
#     # ymax = np.max([np.min([ctr[1] + half_width + 100, img_h]), 100])
# #crop_img = cv2.resize(images[4][0:419,387:1007], crop_size, cv2.INTER_AREA)
#     crop_img = cv2.resize(img[ymin:ymax, xmin:xmax], crop_size, cv2.INTER_AREA)
#     #crop_img = cv2.resize(img[0:419,387:1007], crop_size, cv2.INTER_AREA)
#     min_ind = np.array([xmin, ymin])
#     max_ind = np.array([xmax, ymax])
#     crop_scale = crop_size / (max_ind - min_ind)
#
#     return crop_img, min_ind, crop_scale

def crop_from_com(img, centroid, half_width, crop_size=(500, 500)): # used to be 320, 320
    '''
    Crops an image around a given centroid (crop dims defined by half_width)
    and resizes to the specified crop_size.
    '''
    ctr = np.round(centroid).astype(int)
    half_width = np.round(half_width).astype(int)
    img_h,img_w = img.shape[0],img.shape[1] # used to be img_h, img_w = img.shape

    xmin = np.min([np.max([ctr[0] - half_width, 0]), img_w - 1]) # not really understand why -1/+1
    xmax = np.max([np.min([ctr[0] + half_width + 1, img_w]), 1])
    ymin = np.min([np.max([ctr[1] - half_width, 0]), img_h - 1])
    ymax = np.max([np.min([ctr[1] + half_width + 1, img_h]), 1])

    crop_img = cv2.resize(img[ymin:ymax, xmin:xmax], crop_size, cv2.INTER_AREA)
    min_ind = np.array([xmin, ymin])
    max_ind = np.array([xmax, ymax])
    crop_scale = crop_size / (max_ind - min_ind)

    return crop_img, min_ind, crop_scale


class posture_tracker:

    def __init__(self, readers, camParams,
                    ds_fac=4, w3d=80, crop_size=(500,500),
                    nParts=15, nCOMs=3, com_body_ind=1,
                    com_model=None, posture_model=None, face_model=None,
                    face_w3d=0.08, com_head_ind=0, face_crop_size=(128,128)):
        
        # input a list of video reader objects
        self.readers = readers
        self.nCams = len(readers)
        assert self.nCams == camParams.shape[0]
        self.camParams = camParams
        self.com_model = com_model
        self.posture_model = posture_model
        self.face_model = face_model
        self.ds_fac = ds_fac
        self.ds_size = (2200//ds_fac, 650//ds_fac) # UPDATE to match your image size in pixels
        self.w3d = w3d
        self.crop_size = crop_size
        self.nParts = nParts
        self.nCOMs = nCOMs
        self.com_body_ind = com_body_ind
        self.face_w3d = face_w3d
        self.com_head_ind = com_head_ind
        self.face_crop_size = face_crop_size


    def crop_bird(crop_img, COM,
                    this_w3d=80,
                    min_px=25,
                    this_crop_size=(500,500)): # used to be 320, 320. XL changed it
        '''
        Crops around the full bird or the face given a centroid defined by COM
        Default params are for the full bird crop

        this_w3d : float, defines the relative cropping scale
        min_px : int, defines the minimum pixel size to crop
        this_crop_size : tuple of ints, defines the dimensions of the cropped image
        '''
        # get the 3D distance of the bird from each camera to determine cropping scale
        com_reproj = sba.project(np.tile(COM, (self.nCams, 1)), self.camParams) # get reprojected body centroid location for each camera
        camDist = sba.rotate(np.tile(COM, (self.nCams, 1)), self.camParams[:,:3]) # rotate to camera coordinates
        camDist = camDist[:, 2] + self.camParams[:,5] # get z-axis distance ie along optical axis
        camScale = self.camParams[:, 6] / camDist  # convert to focal length divided by distance
        half_width = camScale * this_w3d
        
        # save the cropped image, min index, and crop scale for each camera
        min_ind = np.full((self.nCams,1,2), np.NaN)
        crop_scale = np.full((self.nCams,1,2), np.NaN)
        for nCam in range(self.nCams):
            thisCom = np.maximum(com_reproj[nCam],0)
            thisCom[0] = np.minimum(thisCom[0], full_img[nCam].shape[1]) # x limit is shape[1]
            thisCom[1] = np.minimum(thisCom[1], full_img[nCam].shape[0]) # y limit is shape[0]
            thisHalfWidth = np.maximum(half_width[nCam], min_px) # minimum 51px image for body
            crop_img[nCam,:,:,0], min_ind[nCam], crop_scale[nCam] = crop_from_com(full_img[nCam],
                                                                                    thisCom,
                                                                                    thisHalfWidth,
                                                                                    )

        return crop_img, min_ind, crop_scale

    def track_video_com(self, base_dir, start_frame=0, nFrames=1000):
        '''
        pared down inference script for just getting the COM model predictions
        '''
        ''' load the models '''
        com_mdl = slp_load([self.com_model], peak_threshold=0)
        #posture_mdl = slp_load([self.posture_model], peak_threshold=0)
        #face_mdl = self.face_model

        ''' Collect camera parameters '''
        print('Collecting Camera Parameters')
        cameraDicts = []
        cameraMats = []
        for nCam in range(self.nCams):  # range would be 0,1,2,3
            theseParams = unconvertParams(self.camParams[nCam])
            cameraDicts.append(theseParams)
            cameraMats.append(camera_matrix(theseParams['K'], theseParams['R'], theseParams['t'].reshape((1, 3))))
        print("cameraMats: ", cameraMats)
        sba = PySBA(self.camParams, np.NaN, np.NaN, np.NaN, np.NaN)

        ''' Read in video and predict keypoints '''
        print('Reading and Predicting')

        # preallocate results lists (all frames)
        comPred = []
        comReproj = []
        comConf = []

        # preallocate image arrays
        ds_img = np.full((self.nCams, self.ds_size[1], self.ds_size[0], 3), 0,
                         dtype='uint8')  # SHERRY: the last dimension used to be 1, needs to make it 3
        crop_img = np.full((self.nCams, self.crop_size[1], self.crop_size[0], 3), 0,
                           dtype='uint8')  # SHERRY: the last dimension used to be 1, need to make it 3
        face_img = np.full((1, self.face_crop_size[1], self.face_crop_size[0], self.nCams), 0,
                           dtype='uint8')  # note different shape w/ n=1 and channels = nCams

        # preallocate results variables (each frame)
        best_com = np.full((self.nCOMs, 3), np.NaN)
        com_reproj = np.full((self.nCOMs), np.NaN)
        com_conf = np.full((self.nCOMs), np.NaN)

        # flag that video reading failed
        stopReading = False

        # read and predict each frame
        end_frame = start_frame + nFrames
        for nFrame in range(end_frame):
            # Read and downsample
            full_img = []
            for nCam in range(self.nCams):
                flag, img = self.readers[nCam].read()
                img = cv2.cvtColor(img, ### SHERRY: temporaily disable it
                                  cv2.COLOR_BGR2RGB)  # SHERRY added this bcs cv2 image reader reads in BGR, need to convert to RGB to read in the images correctly

                if nFrame < start_frame:
                    continue
                full_img.append(img)  #### SHERRY: used to be [:,:,0], but now it is taking all RGB channels
                if full_img[nCam] is None:
                    stopReading = True
                    break
                ds_img[nCam] = cv2.resize(full_img[nCam], self.ds_size,
                                          interpolation=cv2.INTER_AREA)  # SHERRY: used to be [nCam,:,:,0], need to change this so that it is taking in all RGB dimension
            if nFrame < start_frame:
                continue
            elif np.mod(nFrame, 1000) == 0:
                print('Reading Frame {}'.format(nFrame))

            # If reading for any video failed, terminate tracking
            if stopReading:
                print('Terminated Reading on Frame {}'.format(nFrame))
                break

            # Predict coarse keypoints using COM model
            preds = com_mdl.inference_model.predict_on_batch(ds_img, numpy=True)
            COM = np.squeeze(preds['instance_peaks']) * self.ds_fac  # node locations, shape (n_cams, n_keypoints, 2)
            conf = np.squeeze(preds['instance_peak_vals'])  # confidence scores, shape (n_cams, n_keypoints)
            # undistort com for each camera
            for nCam in range(self.nCams):
                COM[nCam] = unDistortPoints(COM[nCam], cameraDicts[nCam]['K'], cameraDicts[nCam]['d'])
            # triangulate all COM keypoints, select best triplet and its reprojection error
            for nCom in range(self.nCOMs):
                com_results = triangulate_confThresh_lowestErr(COM[:, nCom],
                                                               cameraMats,
                                                               conf[:, nCom])
                best_com[nCom], com_reproj[nCom], com_conf[nCom] = com_results
            # collect COM results
            comPred.append(best_com.copy())
            comReproj.append(com_reproj.copy())
            comConf.append(com_conf.copy())

            # get the 3D distance of the bird from each camera to determine cropping scale
            body_COM = best_com[self.com_body_ind]
            body_reproj = sba.project(np.tile(body_COM, (self.nCams, 1)),
                                      self.camParams)  # get reprojected body centroid location for each camera
            camDist = sba.rotate(np.tile(body_COM, (self.nCams, 1)),
                                 self.camParams[:, :3])  # rotate to camera coordinates
            camDist = camDist[:, 2] + self.camParams[:, 5]  # get z-axis distance ie along optical axis
            camScale = self.camParams[:, 6] / camDist  # convert to focal length divided by distance
            half_width = camScale * self.w3d

            # base_dir = 'Z:/Sherry/poseTrackingXL/training_files/posture_vids/unseen_images'  # Base directory for cropped images
            ncams = 4  # Number of cameras
            camera_dirs = [os.path.join(base_dir, f'cam{i + 1}') for i in range(ncams)]

            # Create directories if they don't exist
            for dir in camera_dirs:
                if not os.path.exists(dir):
                    os.makedirs(dir)


            # save the cropped image, min index, and crop scale for each camera
            min_ind = np.full((self.nCams, 1, 2), np.NaN)
            crop_scale = np.full((self.nCams, 1, 2), np.NaN)
            for nCam in range(self.nCams):
                thisCom = np.maximum(body_reproj[nCam], 0)
                thisCom[0] = np.minimum(thisCom[0], full_img[nCam].shape[1])  # x limit is shape[1]
                thisCom[1] = np.minimum(thisCom[1], full_img[nCam].shape[0])  # y limit is shape[0]
                thisHalfWidth = np.maximum(half_width[nCam], 25)  # minimum 51px image for body
                crop_img[nCam], min_ind[nCam], crop_scale[nCam] = crop_from_com(full_img[nCam],
                                                                                thisCom,
                                                                                thisHalfWidth,
                                                                                )                # SHERRY: used to be crop_img[nCam,:,:,0], need to get all RGB
                filename = f'img{nFrame}.png'  # Name of the file to save
                crop_img_rgb = cv2.cvtColor(crop_img[nCam], cv2.COLOR_BGR2RGB)
                cv2.imwrite(os.path.join(camera_dirs[nCam], filename),crop_img_rgb)
                #crop_images.append(crop_img)

        # the return call here executes if loop finishes naturally or is broken when reading fails
        return {'com_preds':np.stack(comPred),
                'com_rep_err':np.stack(comReproj),
                'com_conf':np.stack(comConf),
                'read_status': stopReading,
                'cropped_unseen_images':crop_img}

    def track_video(self, start_frame=0, nFrames=1000):
        '''
        Get coarse and fine keypoint predictions
        TODO add face model / seed predictions
        '''
        ''' load the models '''
        com_mdl = slp_load([self.com_model], peak_threshold=0)
        posture_mdl = slp_load([self.posture_model], peak_threshold=0)
        face_mdl = self.face_model

        ''' Collect camera parameters '''
        print('Collecting Camera Parameters')
        cameraDicts = []
        cameraMats = []
        for nCam in range(self.nCams): # range would be 0,1,2,3
            theseParams = unconvertParams(self.camParams[nCam])
            cameraDicts.append(theseParams)
            cameraMats.append(camera_matrix(theseParams['K'], theseParams['R'], theseParams['t'].reshape((1,3))))
        print("cameraMats: ",cameraMats)
        sba = PySBA(self.camParams, np.NaN, np.NaN, np.NaN, np.NaN)

        ''' Read in video and predict keypoints '''
        print('Reading and Predicting')

        # preallocate results lists (all frames)
        comPred = []
        comReproj = []
        comConf = []
        posturePred = []
        postureReproj = []
        postureConf = []
        rawPosturePreds = []
        facePreds = []

        # preallocate image arrays
        ds_img = np.full((self.nCams, self.ds_size[1], self.ds_size[0], 3), 0, dtype='uint8')  # SHERRY: the last dimension used to be 1, needs to make it 3
        crop_img = np.full((self.nCams, self.crop_size[1], self.crop_size[0], 3), 0, dtype='uint8') # SHERRY: the last dimension used to be 1, need to make it 3
        face_img = np.full((1, self.face_crop_size[1], self.face_crop_size[0], self.nCams), 0, dtype='uint8') # note different shape w/ n=1 and channels = nCams
        
        # preallocate results variables (each frame)
        best_com = np.full((self.nCOMs, 3), np.NaN)
        com_reproj = np.full((self.nCOMs), np.NaN)
        com_conf = np.full((self.nCOMs), np.NaN)
        best_posture = np.full((self.nParts, 3), np.NaN)
        posture_reproj = np.full((self.nParts), np.NaN)
        posture_conf = np.full((self.nParts), np.NaN)

        # flag that video reading failed
        stopReading = False

        # read and predict each frame
        end_frame = start_frame + nFrames
        for nFrame in range(end_frame):
            # Read and downsample
            full_img = []
            for nCam in range(self.nCams):
                flag, img = self.readers[nCam].read()
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # SHERRY added this bcs cv2 image reader reads in BGR, need to convert to RGB to read in the images correctly

                if nFrame < start_frame:
                    continue
                full_img.append(img) #### SHERRY: used to be [:,:,0], but now it is taking all RGB channels
                if full_img[nCam] is None:
                    stopReading = True
                    break
                ds_img[nCam] = cv2.resize(full_img[nCam], self.ds_size,
                                                interpolation=cv2.INTER_AREA)  #SHERRY: used to be [nCam,:,:,0], need to change this so that it is taking in all RGB dimension
            if nFrame < start_frame:
                continue
            elif np.mod(nFrame, 1000)==0:
                print('Reading Frame {}'.format(nFrame))

            # If reading for any video failed, terminate tracking
            if stopReading:
                print('Terminated Reading on Frame {}'.format(nFrame))
                break
            
            # Predict coarse keypoints using COM model
            preds = com_mdl.inference_model.predict_on_batch(ds_img, numpy=True)
            COM = np.squeeze(preds['instance_peaks']) * self.ds_fac # node locations, shape (n_cams, n_keypoints, 2)
            conf = np.squeeze(preds['instance_peak_vals']) # confidence scores, shape (n_cams, n_keypoints)
            # undistort com for each camera
            for nCam in range(self.nCams):
                COM[nCam] = unDistortPoints(COM[nCam], cameraDicts[nCam]['K'], cameraDicts[nCam]['d'])
            # triangulate all COM keypoints, select best triplet and its reprojection error
            for nCom in range(self.nCOMs):
                com_results = triangulate_confThresh_lowestErr(COM[:, nCom],
                                                            cameraMats,
                                                            conf[:, nCom])
                best_com[nCom], com_reproj[nCom], com_conf[nCom] = com_results
            # collect COM results
            comPred.append(best_com.copy())
            comReproj.append(com_reproj.copy())
            comConf.append(com_conf.copy())
            
            # get the 3D distance of the bird from each camera to determine cropping scale
            body_COM = best_com[self.com_body_ind]
            body_reproj = sba.project(np.tile(body_COM, (self.nCams, 1)), self.camParams) # get reprojected body centroid location for each camera
            camDist = sba.rotate(np.tile(body_COM, (self.nCams, 1)), self.camParams[:,:3]) # rotate to camera coordinates
            camDist = camDist[:, 2] + self.camParams[:,5] # get z-axis distance ie along optical axis
            camScale = self.camParams[:, 6] / camDist  # convert to focal length divided by distance
            half_width = camScale * self.w3d

            # save the cropped image, min index, and crop scale for each camera
            min_ind = np.full((self.nCams, 1, 2), np.NaN)
            crop_scale = np.full((self.nCams, 1, 2), np.NaN)
            for nCam in range(self.nCams):
                thisCom = np.maximum(body_reproj[nCam], 0)
                thisCom[0] = np.minimum(thisCom[0], full_img[nCam].shape[1]) # x limit is shape[1]
                thisCom[1] = np.minimum(thisCom[1], full_img[nCam].shape[0]) # y limit is shape[0]
                thisHalfWidth = np.maximum(half_width[nCam], 25) # minimum 51px image for body
                crop_img[nCam], min_ind[nCam], crop_scale[nCam] = crop_from_com(full_img[nCam],
                                                                                        thisCom,
                                                                                        thisHalfWidth,
                                                                                        ) # SHERRY: used to be crop_img[nCam,:,:,0], need to get all RGB


            # Predict posture and convert to full image pixel coordinates
            crop_preds = posture_mdl.inference_model.predict_on_batch(crop_img, numpy=True)
            raw_posture = np.squeeze(crop_preds['instance_peaks']) / crop_scale + min_ind # node locations, shape (n_cams, n_keypoints, 2) # this is where it gets converted back to original image size.
            print("raw posture shape", raw_posture.shape)
            rawPosturePreds.append(raw_posture)
            posture_2d = raw_posture
            conf = np.squeeze(crop_preds['instance_peak_vals']) # confidence scores, shape (n_cams, n_keypoints)
            # undistort posture for each camera
            for nCam in range(self.nCams):
                posture_2d[nCam] = unDistortPoints(posture_2d[nCam], cameraDicts[nCam]['K'], cameraDicts[nCam]['d'])
            print("posture_2d value: ",posture_2d)
            # triangulate posture
            for nPart in range(self.nParts):
                pos_results = triangulate_confThresh_lowestErr(posture_2d[:, nPart],
                                                                cameraMats,
                                                                conf[:, nPart])
                best_posture[nPart], posture_reproj[nPart], posture_conf[nPart] = pos_results
                print("best posture shape", best_posture.shape)
            print("posturePred value: ", best_posture)
            # collect posture results
            posturePred.append(best_posture.copy())
            postureReproj.append(posture_reproj.copy())
            postureConf.append(posture_conf.copy())
            full_img_sleap = np.stack(full_img, axis=0)


        # the return call here executes if loop finishes naturally or is broken when reading fails
        return {'posture_preds':np.stack(posturePred),
                'posture_rep_err':np.stack(postureReproj),
                'posture_rawpred':np.stack(rawPosturePreds),
                'com_preds':np.stack(comPred),
                'com_rep_err':np.stack(comReproj),
                'com_conf':np.stack(comConf),
                'posture_conf':np.stack(postureConf),
                'read_status': stopReading,
                'unseen_images':full_img_sleap,
                'sleap_raw_predicted_points_scale_back': raw_posture,
                'cropped_unseen_images':crop_img} # SHERRY added the sleap raw output