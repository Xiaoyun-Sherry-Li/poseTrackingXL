import json
import glob

import numpy as np
import cv2

stdPath = 'D:\Supervisely\Chickadee-1\\'
stdPts = ['topBeak', 'botBeak','topHead','backHead','centerChest','centerBack','baseTail','tipTail',
            'leftEye','leftNeck','leftWing','leftAnkle','leftFoot',
           'rightEye','rightNeck','rightWing','rightAnkle','rightFoot']
stdCams = ['lBack', 'lFront', 'lTop', 'rBack', 'rFront', 'rTop']
stdAnnotators = ['selmaan', 'stephaniemhale', 'nsobers']
#%%
def get_pt_dict(basePath = stdPath, birdPts = stdPts):
    fnMeta = basePath + 'meta.json'
    with open(fnMeta, "r") as read_file:
        meta = json.load(read_file)
        read_file.close()

    for obj in meta['classes']:
        if obj['title']=='ChickadeePts':
            ptDict = obj['geometry_config']['nodes']

    for k in ptDict:
        thisLabel = ptDict[k]['label']
        label_id = [i for i,e in enumerate(birdPts) if e==thisLabel]
        ptDict[k]['id'] = label_id

    return ptDict

def getImageAndAnnotations(dataset, ptDict, basePath = stdPath, allCameras = stdCams,
                           allAnnotators = stdAnnotators, center_coords=False):
    datapath = basePath + dataset + '\\'
    allAnn = glob.glob(datapath + 'ann\\*.json')
    allIm = glob.glob(datapath + 'img\\*.bmp')
    assert(len(allAnn) == len(allIm))

    imCameras = []
    imNumbers = []
    for ann in allAnn:
        numStart = ann.rfind('_') + 1
        numStop = ann.rfind('.bmp')
        imNumbers.append(ann[numStart:numStop])
        camStart = ann.rfind('_', 1, numStart - 1) + 1
        imCameras.append(ann[camStart:numStart - 1])
    allNumbers = np.unique(imNumbers)

    imData = []
    annData = []
    metaData = []
    for nIm, annName in enumerate(allAnn):
        imName = annName[:-5].replace('\\ann\\','\\img\\')
        img = cv2.imread(imName)[:,:,0]
        imData.append(img)

        thisCam = [i for i, e in enumerate(allCameras) if e == imCameras[nIm]][0]
        thisIm = [i for i, e in enumerate(allNumbers) if e == imNumbers[nIm]][0]
        ind3D = thisIm * len(ptDict) + np.arange(len(ptDict))

        if center_coords:
            if (imCameras[nIm] == 'lTop') | (imCameras[nIm] == 'rTop'):
                imCenter = np.array([2816 / 2, 1696 / 2])
            else:
                imCenter = np.array([2816 / 2, 1408 / 2])
        else:
            imCenter = np.array([0, 0])

        with open(annName, 'r') as read_file:
            data = json.load(read_file)
            read_file.close()
        loc = np.full((len(ptDict), len(data['objects']), 2), np.NaN)
        theseAnnotators = []
        for nObj, obj in enumerate(data['objects']):
            if obj['labelerLogin'] == 'selmaan':
                thisAnnotator = 0
            elif obj['labelerLogin'] == 'stephaniemhale':
                thisAnnotator = 1
            else:
                thisAnnotator = 2
            theseAnnotators.append(thisAnnotator)

            for k in obj['nodes']:
                ptID = ptDict[k]['id']
                ptLoc = np.array(obj['nodes'][k]['loc'])
                loc[ptID, nObj, :] = ptLoc - imCenter
        annData.append(loc)
        metaData.append((thisCam, thisIm, ind3D, theseAnnotators))

    return imData, annData, metaData