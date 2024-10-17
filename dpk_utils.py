#%%
import h5py
import numpy as np
from deepposekit.models import load_model as dpk_load
import cv2
from ffVideoReader import ffVideoReader
from triangulation_utils import unDistortPoints, camera_matrix, triangulate_confThresh_lowestErr
from pySBA import unconvertParams, PySBA

#%%
def verify_video_metadata(path, targetRate=60, startTol=.001, rateTol=.01, ifiTol=.001, discard_frames=1):

    cams = ['lTop','lFront','lBack','rBack','rFront','rTop']

    f = []
    for c in cams:
        f.append(h5py.File(path + '\\' + c + '_metadata.h5', 'r'))
    timestamps = []
    systimes = []
    IFI = []
    t0 = []
    for c in f:
        timestamps.append(c['timestamp'])
        t0.append(c['timestamp'][0])
        systimes.append(c['sestime'])
        IFI.append(c['timestamp'][1:] - c['timestamp'][:-1])
    t0 = np.array(t0)
    metadata = (t0, IFI)

    startCheck = t0.max() - t0.min() > startTol
    if startCheck:
        print('Failed Check: Start times are not synchronous')
        return False, metadata

    discard_length = np.zeros(len(IFI))
    for i, ifi in enumerate(IFI):
        if 1/ifi.mean() > targetRate+rateTol or 1/ifi.mean() < targetRate-rateTol:
            print('Failed Check Cam %d: Empirical rate does not match target' %i)
            return False, metadata
        if ifi[discard_frames:].max() > 1/targetRate+ifiTol or ifi[discard_frames:].min() < 1/targetRate-ifiTol:
            print('Failed Check Cam %d: IFI error, missing frames likely' %i)
            print('Max: %f, Min: %f' % (ifi.max(), ifi.min()))
            return False, metadata
        if discard_frames > 0:
            discard_length[i] = ifi[:discard_frames].sum()

    if discard_length.max()-discard_length.min() > ifiTol:
        print('Failed Discard Check: Discarded frames are not equal in duration')
        return False, discard_length

    print('Metadata passed validation!')
    return True, metadata

def resize_for_crop(img, ds_size):
    imShape = img.shape
    ds_fac = imShape[0] / ds_size[1]  # ds_size is openCV format of wxh
    if imShape[1] / imShape[0] == ds_size[0] / ds_size[1]:  # it's side camera and/or no change in aspect ratio
        ds_img = cv2.resize(img, ds_size, interpolation=cv2.INTER_AREA)
    else:  # need to pad to preserve aspect ratio
        ds_col = int(np.round(imShape[1] / ds_fac))
        ds_img = cv2.resize(img, (ds_col, ds_size[1]), interpolation=cv2.INTER_AREA)
        fill_col = np.full((ds_size[1], ds_size[0] - ds_col), 0, dtype='uint8')
        ds_img = np.concatenate((ds_img, fill_col), axis=1)

    return ds_img, ds_fac

def crop_from_com(img, com, half_width, crop_size = (320,320)):
    com = np.round(com).astype(int)
    half_width = np.round(half_width).astype(int)
    xmin = np.min([np.max([com[0]-half_width, 0]), img.shape[1]-1])
    xmax = np.max([np.min([com[0]+half_width+1, img.shape[1]]), 1])
    ymin = np.min([np.max([com[1]-half_width, 0]), img.shape[0]-1])
    ymax = np.max([np.min([com[1]+half_width+1, img.shape[0]]), 1])
    crop_img = cv2.resize(img[ymin:ymax, xmin:xmax], crop_size, cv2.INTER_AREA)
    min_ind = np.array([xmin, ymin])
    max_ind = np.array([xmax, ymax])
    crop_scale = crop_size / (max_ind - min_ind)
    return crop_img, min_ind, crop_scale

def resize_and_pad_rows(img, ds_size):
    # Crops input image to a fixed size, by appending rows at bottom of image as necessary to maintain aspect ratio
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

#%%
class posture_tracker:

    def __init__(self, readers, camParams, ds_fac=4, w3d=0.125, crop_size=(320,320), nParts=18, nCOMs=3, com_body_ind=1,
                 com_model=None, posture_model=None, face_model=None, face_w3d=0.04, com_head_ind=0, face_crop_size=(128,128)):
        # input a list of video reader objects
        self.readers = readers
        self.nCams = len(readers)
        assert self.nCams == camParams.shape[0]
        self.camParams = camParams
        self.com_model = com_model
        self.posture_model = posture_model
        self.face_model = face_model
        self.ds_fac = ds_fac
        self.ds_size = (2816//ds_fac, 1696//ds_fac)
        self.w3d = w3d
        self.crop_size = crop_size
        self.nParts = nParts
        self.nCOMs = nCOMs
        self.com_body_ind = com_body_ind
        self.face_w3d = face_w3d
        self.com_head_ind = com_head_ind
        self.face_crop_size = face_crop_size

    def crop_video(self, nFrames = 1000):
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

    def track_video(self, nFrames = 1000):
        com_mdl = dpk_load(self.com_model, compile=False)
        posture_mdl = dpk_load(self.posture_model, compile=False)
        face_mdl = self.face_model

        print('Collecting Camera Parameters')
        cameraDicts = []
        cameraMats = []
        for nCam in range(self.nCams):
            theseParams = unconvertParams(self.camParams[nCam])
            cameraDicts.append(theseParams)
            cameraMats.append(camera_matrix(theseParams['K'], theseParams['R'], theseParams['t'].reshape((1,3))))
        sba = PySBA(self.camParams, np.NaN, np.NaN, np.NaN, np.NaN)

        print('Reading and Predicting')
        comPred = []
        comReproj = []
        comConf = []
        posturePred = []
        postureReproj = []
        postureConf = []
        rawPosturePreds = []
        facePreds = []
        ds_img = np.full((self.nCams, self.ds_size[1], self.ds_size[0], 1), 0, dtype='uint8')
        crop_img = np.full((self.nCams, self.crop_size[1], self.crop_size[0], 1), 0, dtype='uint8')
        face_img = np.full((1, self.face_crop_size[1], self.face_crop_size[0], self.nCams), 0, dtype='uint8') # note different shape w/ n=1 and channels = nCams
        best_com = np.full((self.nCOMs, 3), np.NaN)
        com_reproj = np.full((self.nCOMs), np.NaN)
        com_conf = np.full((self.nCOMs), np.NaN)
        best_posture = np.full((self.nParts, 3), np.NaN)
        posture_reproj = np.full((self.nParts), np.NaN)
        posture_conf = np.full((self.nParts), np.NaN)
        stopReading = False
        for nFrame in range(nFrames):
            if np.mod(nFrame,1000)==0:
                print('Reading Frame {}'.format(nFrame))
            # Read, resize, and predict COM
            full_img = []
            for nCam in range(self.nCams):
                full_img.append(self.readers[nCam].read())
                if full_img[nCam] is None:
                    stopReading = True
                    break
                ds_img[nCam,:,:,0] = resize_and_pad_rows(full_img[nCam], self.ds_size)
            # If reading for any video failed, terminate tracking
            if stopReading:
                print('Terminated Reading on Frame {}'.format(nFrame))
                break
            preds = com_mdl.predict_on_batch(ds_img)
            COM = preds[:,:,:2] * self.ds_fac
            # Undistort com for each camera
            for nCam in range(self.nCams):
                COM[nCam] = unDistortPoints(COM[nCam], cameraDicts[nCam]['K'], cameraDicts[nCam]['d'])
            # triangulate all COMS, select best triplet and its reprojection error
            for nCom in range(self.nCOMs):
                best_com[nCom], com_reproj[nCom], com_conf[nCom] = triangulate_confThresh_lowestErr(COM[:,nCom], cameraMats, preds[:,nCom,2])
            #collect bodyCOM results
            comPred.append(best_com.copy())
            comReproj.append(com_reproj.copy())
            comConf.append(com_conf.copy())

            #calculate 3d scale and crop around body
            body_COM = best_com[self.com_body_ind]
            body_reproj = sba.project(np.tile(body_COM, (self.nCams, 1)), self.camParams) # get reprojected body centroid location for each camera
            camDist = sba.rotate(np.tile(body_COM, (self.nCams, 1)), self.camParams[:,:3]) # rotate to camera coordinates
            camDist = camDist[:, 2] + self.camParams[:,5] # get z-axis distance ie along optical axis
            camScale = self.camParams[:,6] / camDist  # convert to focal length divided by distance
            half_width = camScale * self.w3d
            min_ind = np.full((self.nCams,1,2), np.NaN)
            crop_scale = np.full((self.nCams,1,2), np.NaN)
            for nCam in range(self.nCams):
                thisCom = np.maximum(body_reproj[nCam],0)
                thisCom[0] = np.minimum(thisCom[0],full_img[nCam].shape[1]) # x limit is shape[1]
                thisCom[1] = np.minimum(thisCom[1], full_img[nCam].shape[0]) # y limit is shape[0]
                thisHalfWidth = np.maximum(half_width[nCam], 25) # minimum 51px image for body
                crop_img[nCam,:,:,0], min_ind[nCam], crop_scale[nCam] = crop_from_com(
                    full_img[nCam], thisCom, thisHalfWidth, self.crop_size)
            #predict posture and convert to full image pixel coordinates
            crop_preds = posture_mdl.predict_on_batch(crop_img)
            crop_preds[:,:,:2] = crop_preds[:,:,:2]/crop_scale + min_ind
            rawPosturePreds.append(crop_preds)
            # posture_2d = (crop_preds[:,:,:2]-min_ind.reshape((6,1,2))) * crop_scale.reshape((6,1,2))
            posture_2d = crop_preds[:,:,:2]
            # Undistort posture for each camera
            for nCam in range(self.nCams):
                posture_2d[nCam] = unDistortPoints(posture_2d[nCam], cameraDicts[nCam]['K'], cameraDicts[nCam]['d'])
            # triangulate posture
            for nPart in range(self.nParts):
                best_posture[nPart], posture_reproj[nPart], posture_conf[nPart] = triangulate_confThresh_lowestErr(
                    posture_2d[:,nPart], cameraMats, crop_preds[:,nPart,2])
            # collect posture results
            posturePred.append(best_posture.copy())
            postureReproj.append(posture_reproj.copy())
            postureConf.append(posture_conf.copy())

            # calculate 3d scale and crop around head
            if face_mdl is not None:
                head_COM = best_com[self.com_head_ind]
                head_reproj = sba.project(np.tile(head_COM, (self.nCams, 1)),
                                          self.camParams)  # get reprojected body centroid location for each camera
                camDist = sba.rotate(np.tile(head_COM, (self.nCams, 1)),
                                     self.camParams[:, :3])  # rotate to camera coordinates
                camDist = camDist[:, 2] + self.camParams[:, 5]  # get z-axis distance ie along optical axis
                camScale = self.camParams[:, 6] / camDist  # convert to focal length divided by distance
                half_width = camScale * self.face_w3d
                min_ind = np.full((self.nCams, 1, 2), np.NaN)
                crop_scale = np.full((self.nCams, 1, 2), np.NaN)
                for nCam in range(self.nCams):
                    thisCom = np.maximum(head_reproj[nCam], 0)
                    thisCom[0] = np.minimum(thisCom[0], full_img[nCam].shape[1])  # x limit is shape[1]
                    thisCom[1] = np.minimum(thisCom[1], full_img[nCam].shape[0])  # y limit is shape[0]
                    thisHalfWidth = np.maximum(half_width[nCam], 15)  # minimum 31px image for head
                    face_img[0, :, :, nCam], _, _ = crop_from_com(
                        full_img[nCam], thisCom, thisHalfWidth, self.face_crop_size)
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

#%%
class com_tracker:

    def __init__(self, reader, ds_fac = 4,
                 com_model = 'C:\\Users\\Selmaan\\PycharmProjects\\supervisely_dpk_training\\comNet\\comNet_Model.h5'):
        self.reader = reader
        self.com_model = com_model
        self.ds_fac = ds_fac
        self.ds_size = (2816//ds_fac, 1696//ds_fac)

    def track_video(self, batch_size = 10, nFrames = 1000):
        start_ind = self.reader.read_count
        stop_ind = start_ind+nFrames
        com_model = load_model(self.com_model, compile=False)

        print('Reading and Predicting')
        allPreds = []
        ds_vid = np.full((batch_size, self.ds_size[1], self.ds_size[0], 1), 0, dtype='uint8')
        while self.reader.read_count < stop_ind:
            if np.mod(self.reader.read_count,1000)==0:
                print(self.reader.read_count)
            for i in range(batch_size):
                img = self.reader.read()
                if len(img)>1:
                    ds_vid[i,:,:,0] = resize_and_pad_rows(img, self.ds_size)
                else:
                    ds_vid = ds_vid[:i]
                    break
            preds = com_model.predict_on_batch(ds_vid)
            preds[:,:,:2] *= self.ds_fac
            allPreds.append(preds)

        return np.concatenate(allPreds)