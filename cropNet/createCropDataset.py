#%%
import numpy as np
import cv2
from deepposekit.io import initialize_dataset
from deepposekit import Annotator
from supervisely_utils import getImageAndAnnotations, get_pt_dict

ds_size = (704, 352)
basePath = 'D:\\Supervisely\\Chickadee-1\\'
datasets = ['IND32 200226', 'IND32 200309',
            'IND32 200315', 'AMB23 200315', 'EMR36 200327']
skeleton = 'C:\\Users\\Selmaan\\PycharmProjects\\supervisely_dpk_training\\skeleton.csv'
dataGenPath = 'C:\\Users\\Selmaan\\PycharmProjects\supervisely_dpk_training\\cropNet\\cropNet_Dataset.h5'
ptDict = get_pt_dict(basePath=basePath)

imData = []
annData = []
for dataset in datasets:
    sideData = dataset + ' SideImages'
    topData = dataset + ' TopImages'
    iSide, aSide, mSide = getImageAndAnnotations(sideData, ptDict, basePath=basePath, center_coords=False)
    iTop, aTop, mTop = getImageAndAnnotations(topData, ptDict, basePath=basePath, center_coords=False)
    i = iSide + iTop
    a = aSide + aTop
    m = mSide + mTop
    for nIm in range(len(i)):
        nAnn = a[nIm].shape[1]
        imShape = i[nIm].shape
        ds_fac = imShape[0] / ds_size[1] # ds_size is openCV format of wxh
        if imShape[1] / imShape[0] == ds_size[0] / ds_size[1]: # it's side camera and/or no change in aspect ratio
            ds_im = cv2.resize(i[nIm], ds_size, interpolation=cv2.INTER_AREA)
        else: # need to pad to preserve aspect ratio
            ds_col = int(np.round(imShape[1] / ds_fac))
            ds_im = cv2.resize(i[nIm], (ds_col, ds_size[1]), interpolation=cv2.INTER_AREA)
            fill_col = np.full((ds_size[1], ds_size[0]-ds_col), 0, dtype='uint8')
            ds_im = np.concatenate((ds_im, fill_col), axis=1)

        for n in range(nAnn):
            imData.append(ds_im)
            annData.append(a[nIm][:,n,:]/ds_fac)

imData = np.asarray(imData)
annData = np.asarray(annData)

#%%

initialize_dataset(datapath = dataGenPath, images=imData[:,:,:,np.newaxis], skeleton=skeleton, keypoints=annData, overwrite=True)

#%%

app = Annotator(datapath=dataGenPath,
                dataset='images',
                skeleton=skeleton,
                shuffle_colors=False,
                scale=3,
                text_scale=1/4)
app.run()