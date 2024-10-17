#%%
import numpy as np
import pySBA
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
from ffVideoReader import ffVideoReader
from dpk_utils import posture_tracker

ds_fac = 4
path = "Z:\Selmaan\Birds\IND88\IND88_201218_112623"
camParams = loadmat("./optCamArray_laserCalib_201125.mat")['optCamArray']
comNet = "Z:\Selmaan\DPK-transfer\comNet_Model_03.h5"
stdCams = ['lBack', 'lFront', 'lTop', 'rBack', 'rFront', 'rTop']
startTime = 0 # in seconds

allReaders = []
for i in range(len(stdCams)):
    cam = stdCams[i]
    print(cam)
    if cam == 'lTop' or cam == 'rTop':
        h=1696
    else:
        h=1408
    #camPath = path+"\\videos\\"+cam+'\\0.avi'
    camPath = path+"\\"+cam+".avi"
    allReaders.append(ffVideoReader(camPath, w=2816, h=h, s=startTime))

# obj = posture_tracker(allReaders, camParams, com_model=comNet, com_body_ind=1, w3d=0.125, crop_size=(320,320))
obj = posture_tracker(allReaders, camParams, com_model=comNet, com_body_ind=0, w3d=0.025, crop_size=(64,64))

#%%
import subprocess as sp
def initialize_ffmpeg(filename,framesize, codec_name='libx264', fps=60):
    filename = filename + '.avi'
    # codec_gpu = '0'
    codec_preset = 'fast'
    codec_qp = '16'
    size_string = '%dx%d' %framesize
    fps = str(fps)
    command = [ 'ffmpeg',
        '-y', # (optional) overwrite output file if it exists
        '-vsync', '0',
        '-f', 'rawvideo',
        '-vcodec','rawvideo',
        '-s', size_string, # size of one frame
        '-pix_fmt', 'gray8',
        '-r', fps, # frames per second
        '-i', '-', # The input comes from a pipe
        '-an', # Tells FFMPEG not to expect any audio
        '-vcodec', codec_name,
        '-preset', codec_preset,
        '-crf', codec_qp,
        filename]
    pipe = sp.Popen(command, stdin=sp.PIPE, stderr=sp.STDOUT, bufsize=-1)
    return (pipe)

def write_frame_ffmpeg(pipe, frame):
    # out = cv2.cvtColor(np.hstack((left,right)), cv2.COLOR_GRAY2RGB)
    # t0 = time.perf_counter()
    try:
        pipe.stdin.write(frame.tobytes())
    except BaseException as err:
        _, ffmpeg_error = pipe.communicate()
        error = (str(err) + ("\n\nerror: FFMPEG encountered "
                             "the following error while writing file:"
                             "\n\n %s" % (str(ffmpeg_error))))

#%%

# fOut = [path + "\\"+n+"_bodyCrop" for n in stdCams]
# allWriters = [initialize_ffmpeg(f, (320,320)) for f in fOut]
fOut = [path + "\\"+n+"_headCrop" for n in stdCams]
allWriters = [initialize_ffmpeg(f, (64,64)) for f in fOut]


all_min_ind = []
all_crop_scale = []
while obj.readers[0].read_count < 576000:
    print(obj.readers[0].read_count),
    v, i, s = obj.crop_video(1000)
    all_min_ind.append(i)
    all_crop_scale.append(s)
    for c in range(6):
        for f in range(v.shape[-1]):
            write_frame_ffmpeg(allWriters[c], v[c,:,:,f])

# savemat(path+"\\bodyCrop_metadata.mat",{
#     "min_ind": np.concatenate(all_min_ind, axis=2), "crop_scale": np.concatenate(all_crop_scale,axis=2)})
savemat(path+"\\headCrop_metadata.mat",{
    "min_ind": np.concatenate(all_min_ind, axis=2), "crop_scale": np.concatenate(all_crop_scale,axis=2)})