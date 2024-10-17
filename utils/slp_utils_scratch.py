import h5py
import numpy as np
# from deepposekit.models import load_model as dpk_load
from sleap import load_model as slp_load
import cv2
from triangulation_utils import unDistortPoints, camera_matrix, triangulate_confThresh_lowestErr
from pySBA import unconvertParams, PySBA


''' TODO 
- look up SLEAP predict code and replace
- check image dims, orientation
- check triangulation_utils
'''



''' 9/4/24 - WORKING ON MODIFYING FROM DPK UTILS '''
class posture_tracker:

    def __init__(self, readers, camParams,
                    ds_fac=4, w3d=0.25, crop_size=(320,320),
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
        self.ds_size = (1896//ds_fac, 640//ds_fac) # UPDATE to match your image size in pixels
        self.w3d = w3d
        self.crop_size = crop_size
        self.nParts = nParts
        self.nCOMs = nCOMs
        self.com_body_ind = com_body_ind
        self.face_w3d = face_w3d
        self.com_head_ind = com_head_ind
        self.face_crop_size = face_crop_size

    def crop_video(self, nFrames = 1000):
        '''
        IS THIS FUNCTION STILL USED?
        '''
        com_mdl = dpk_load(self.com_model, compile=False)
        print('Collecting Camera Parameters')
        cameraDicts = []
        cameraMats = []
        for nCam in range(self.nCams):
            theseParams = unconvertParams(self.camParams[nCam])
            cameraDicts.append(theseParams)
            cameraMats.append(camera_matrix(theseParams['K'], theseParams['R'], theseParams['t'].reshape((1, 3))))
        sba = PySBA(self.camParams, np.NaN, np.NaN, np.NaN, np.NaN)

        ds_img = np.full((self.nCams, self.ds_size[1], self.ds_size[0], 1), 0, dtype='uint8')
        crop_img = np.full((self.nCams, self.crop_size[1], self.crop_size[0], 1), 0, dtype='uint8')
        crop_vid = np.full((self.nCams, self.crop_size[1], self.crop_size[0], nFrames), 0, dtype='uint8')
        all_min_ind = np.full((self.nCams, 2, nFrames), np.NaN)
        all_crop_scale = np.full((self.nCams, 2, nFrames), np.NaN)
        for nFrame in range(nFrames):
            if np.mod(nFrame,1000)==0:
                print('Reading Frame {}'.format(nFrame))
            # Read, resize, and predict COM
            full_img = []
            for nCam in range(self.nCams):
                full_img.append(self.readers[nCam].read())
                ds_img[nCam,:,:,0] = resize_and_pad_rows(full_img[nCam], self.ds_size)
            preds = com_mdl.predict_on_batch(ds_img)
            COM = preds[:,:,:2] * self.ds_fac
            # Undistort com for each camera
            for nCam in range(self.nCams):
                COM[nCam] = unDistortPoints(COM[nCam], cameraDicts[nCam]['K'], cameraDicts[nCam]['d'])
            # triangulate all COMS, select best triplet and its reprojection error
            body_COM, com_reproj, com_conf = triangulate_confThresh_lowestErr(COM[:, self.com_body_ind], cameraMats, preds[:, self.com_body_ind, 2])
            #calculate 3d scale and crop
            body_reproj = sba.project(np.tile(body_COM, (self.nCams, 1)), self.camParams) # get reprojected body centroid location for each camera
            camDist = sba.rotate(np.tile(body_COM, (self.nCams, 1)), self.camParams[:,:3]) # rotate to camera coordinates
            camDist = camDist[:, 2] + self.camParams[:,5] # get z-axis distance ie along optical axis
            camScale = self.camParams[:,6] / camDist  # convert to focal length divided by distance
            half_width = camScale * self.w3d
            min_ind = np.full((self.nCams,2), np.NaN)
            crop_scale = np.full((self.nCams,2), np.NaN)
            for nCam in range(self.nCams):
                crop_img[nCam,:,:,0], min_ind[nCam], crop_scale[nCam] = crop_from_com(
                    full_img[nCam], np.round(body_reproj[nCam]), np.round(half_width[nCam]), self.crop_size)
            crop_vid[:,:,:,nFrame] = crop_img[:,:,:,0]
            all_min_ind[:,:,nFrame] = min_ind
            all_crop_scale[:,:,nFrame] = crop_scale

        return crop_vid, all_min_ind, all_crop_scale

    def crop_bird(crop_img, COM,
                    this_w3d=self.w3d,
                    min_px=25,
                    this_crop_size=self.crop_size):
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


    def track_video(self, start_frame=0, nFrames=1000):
        ''' load the models '''
        com_mdl = slp_load(self.com_model)
        posture_mdl = slp_load(self.posture_model)
        face_mdl = self.face_model

        ''' Collect camera parameters '''
        print('Collecting Camera Parameters')
        cameraDicts = []
        cameraMats = []
        for nCam in range(self.nCams):
            theseParams = unconvertParams(self.camParams[nCam])
            cameraDicts.append(theseParams)
            cameraMats.append(camera_matrix(theseParams['K'], theseParams['R'], theseParams['t'].reshape((1,3))))
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
        ds_img = np.full((self.nCams, self.ds_size[1], self.ds_size[0], 1), 0, dtype='uint8')
        crop_img = np.full((self.nCams, self.crop_size[1], self.crop_size[0], 1), 0, dtype='uint8')
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
                if nFrame < start_frame:
                    continue
                full_img.append(img[:,:,0])
                if full_img[nCam] is None:
                    stopReading = True
                    break
                ds_img[nCam,:,:,0] = cv2.resize(full_img[nCam], ds_size,
                                                interpolation=cv2.INTER_AREA)
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
            # TODO: this produces a `Labels` object containing predictions
            # `Labels(labeled_frames=100, videos=1, skeletons=1, tracks=2)`
            # this should be a similar format to the obj create in `create_slp_project`
            COM = preds[:,:,:2] * self.ds_fac
            # undistort com for each camera
            for nCam in range(self.nCams):
                COM[nCam] = unDistortPoints(COM[nCam], cameraDicts[nCam]['K'], cameraDicts[nCam]['d'])
            # triangulate all COMS, select best triplet and its reprojection error
            for nCom in range(self.nCOMs):
                best_com[nCom], com_reproj[nCom], com_conf[nCom] = triangulate_confThresh_lowestErr(COM[:,nCom],
                                                                                                    cameraMats,
                                                                                                    preds[:,nCom,2])
            # collect bodyCOM results
            comPred.append(best_com.copy())
            comReproj.append(com_reproj.copy())
            comConf.append(com_conf.copy())

            # calculate 3d scale and crop around the bird
            body_COM = best_com[self.com_body_ind]
            crop_img[:,:,:,0], min_ind, crop_scale = crop_bird(crop_img, body_COM)
            
            # Predict posture and convert to full image pixel coordinates
            crop_preds = posture_mdl.inference_model.predict_on_batch(crop_img, numpy=True)
            # TODO update to match sleap formating (see above)
            crop_preds[:,:,:2] = crop_preds[:,:,:2] / crop_scale + min_ind
            rawPosturePreds.append(crop_preds)
            posture_2d = crop_preds[:,:,:2]
            # undistort posture for each camera
            for nCam in range(self.nCams):
                posture_2d[nCam] = unDistortPoints(posture_2d[nCam], cameraDicts[nCam]['K'], cameraDicts[nCam]['d'])
            # triangulate posture
            for nPart in range(self.nParts):
                best_posture[nPart], posture_reproj[nPart], posture_conf[nPart] = triangulate_confThresh_lowestErr(posture_2d[:,nPart],
                                                                                                                    cameraMats,
                                                                                                                    crop_preds[:,nPart,2])
            # collect posture results
            posturePred.append(best_posture.copy())
            postureReproj.append(posture_reproj.copy())
            postureConf.append(posture_conf.copy())

            # calculate 3d scale and crop around head
            if face_mdl is not None:
                head_COM = best_com[self.com_head_ind]
                face_img[0, :, :, nCam], _, _ crop_bird(crop_img, COM,
                                                        this_w3d=self.face_w3d,
                                                        min_px=15,
                                                        this_crop_size=self.face_crop_size)

                # make prediction on multichannel head data
                thisPrediction = face_mdl.predict_on_batch(face_img)[0]
                facePreds.append(thisPrediction.copy())
            else:
                facePreds.append(None)

        # the return call here executes if loop finishes naturally or is broken when reading fails
        return {'posture_preds':np.stack(posturePred),
                'posture_rep_err':np.stack(postureReproj),
                'posture_rawpred':np.stack(rawPosturePreds),
                'com_preds':np.stack(comPred),
                'com_rep_err':np.stack(comReproj),
                'com_conf':np.stack(comConf),
                'posture_conf':np.stack(postureConf),
                'read_status': stopReading,
                'face_preds': np.stack(facePreds)}


    def track_video_com(self, start_frame=0, nFrames=1000):
        '''
        pared down inference script for just getting the COM model predictions
        '''
        ''' load the models '''
        com_mdl = slp_load(self.com_model)

        ''' Collect camera parameters '''
        print('Collecting Camera Parameters')
        cameraDicts = []
        cameraMats = []
        for nCam in range(self.nCams):
            theseParams = unconvertParams(self.camParams[nCam])
            cameraDicts.append(theseParams)
            cameraMats.append(camera_matrix(theseParams['K'], theseParams['R'], theseParams['t'].reshape((1,3))))
        sba = PySBA(self.camParams, np.NaN, np.NaN, np.NaN, np.NaN)

        ''' Read in video and predict keypoints '''
        print('Reading and Predicting')

        # preallocate results lists (all frames)
        comPred = []
        comReproj = []
        comConf = []

        # preallocate image arrays
        ds_img = np.full((self.nCams, self.ds_size[1], self.ds_size[0], 1), 0, dtype='uint8')
        
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
                if nFrame < start_frame:
                    continue
                full_img.append(img[:,:,0])
                if full_img[nCam] is None:
                    stopReading = True
                    break
                ds_img[nCam,:,:,0] = cv2.resize(full_img[nCam], ds_size,
                                                interpolation=cv2.INTER_AREA)
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
            conf = np.squeeze(pred['instance_peak_vals']) # confidence scores, shape (n_cams, n_keypoints)

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

        # the return call here executes if loop finishes naturally or is broken when reading fails
        return {'com_preds':np.stack(comPred),
                'com_rep_err':np.stack(comReproj),
                'com_conf':np.stack(comConf),
                'posture_conf':np.stack(postureConf),
                'read_status': stopReading}
