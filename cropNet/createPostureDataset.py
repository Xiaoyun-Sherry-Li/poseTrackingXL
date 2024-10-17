#%%
import numpy as np
import cv2
from deepposekit.io import initialize_dataset
from deepposekit import Annotator
from deepposekit.models import load_model
from supervisely_utils import getImageAndAnnotations, get_pt_dict, crop_from_predictions, resize_for_crop

ds_size = (704, 352)
crop_size = (320, 320)
crop_border_fac = 0.85
confidence_thresh = 0.1
basePath = 'D:\\Supervisely\\Chickadee-1\\'
datasets = ['IND32 200226', 'IND32 200309',
            'IND32 200315', 'AMB23 200315', 'EMR36 200327']
skeleton = 'C:\\Users\\Selmaan\\PycharmProjects\\supervisely_dpk_training\\skeleton.csv'
dataGenPath = 'C:\\Users\\Selmaan\\PycharmProjects\supervisely_dpk_training\\cropNet\\postureNet_Dataset.h5'
cropModelPath = 'C:\\Users\\Selmaan\\PycharmProjects\supervisely_dpk_training\\cropNet\\cropNet_Model.h5'
model = load_model(cropModelPath, compile=False)
ptDict = get_pt_dict(basePath=basePath)

imData = []
annData = []
scaleData = []
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
        ds_im, ds_fac = resize_for_crop(i[nIm], ds_size)
        preds = model.predict(ds_im.reshape((1, ds_im.shape[0], ds_im.shape[1], 1)))[0]
        preds = ds_fac * preds[preds[:, 2] > confidence_thresh, :2]
        crop_img, min_ind, max_ind = crop_from_predictions(i[nIm], preds, crop_size, crop_border_fac = crop_border_fac)
        crop_scale = crop_size / (max_ind-min_ind)
        crop_ann = (a[nIm]-min_ind)*crop_scale
        scaleData.append(crop_scale)
        for n in range(nAnn):
            imData.append(crop_img)
            annData.append(crop_ann[:,n,:])

imData = np.asarray(imData)
annData = np.asarray(annData)
scaleData = np.asarray(scaleData)

print(np.mean(scaleData, axis=0))
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