#%%
import subprocess as sp
import numpy as np
import matplotlib.pyplot as plt

from threading import Thread
from queue import Queue

#%%
class ffVideoReader:

    def __init__(self, path, w=2816, h=1408, s=0):
        command = [ 'ffmpeg',
                    '-vsync', '0',
                    '-vcodec', 'h264',
                    '-ss', '{0}'.format(s),
                    '-i', path,
                    '-f', 'rawvideo',
                    '-pix_fmt', 'gray8', '-']
        self.pipe = sp.Popen(command, stdout = sp.PIPE, stderr=sp.DEVNULL, bufsize=-1) #remove stderr to devnull to print output
        self.read_count = 0
        self.w = w
        self.h = h
        self.frameBytes = w*h
        # below was needed before I specified pix_fmt as gray 8
        #self.frameBytes = (w * h * 6) // 4
        #self.kBytes = w*h

    def __del__(self):
        self.pipe.stdout.flush()
        self.pipe.terminate()


    def read(self):
        raw_image = self.pipe.stdout.read(self.frameBytes)
        try:
            frame = np.frombuffer(raw_image, dtype='uint8').reshape((self.h, self.w))
            self.read_count += 1
            return frame
            # frame = np.frombuffer(raw_image[:self.kBytes], dtype='uint8').reshape((self.h, self.w))
        except:
            print('Stopped Reading Movie')
            return None

#%%

class ffVideoReader_queued:

    def __init__(self, path, w=2816, h=1408, s=0, queueSize=128):
        command = [ 'ffmpeg',
                    '-vsync', '0',
                    '-vcodec', 'h264',
                    '-ss', '{0}'.format(s),
                    '-i', path,
                    '-f', 'rawvideo',
                    '-pix_fmt', 'gray8', '-']
        self.pipe = sp.Popen(command, stdout = sp.PIPE, stderr=sp.DEVNULL, bufsize=-1) #remove stderr to devnull to print output
        self.stopped = True
        self.buff_count = 0
        self.read_count = 0
        self.w = w
        self.h = h
        self.frameBytes = w*h
        # below was needed before I specified pix_fmt as gray 8
        #self.frameBytes = (w * h * 6) // 4
        #self.kBytes = w*h

        # initialize the queue used to store frames read from
        # the video file
        self.Q = Queue(maxsize=queueSize)

    def __del__(self):
        self.pipe.stdout.flush()
        self.pipe.terminate()

    def start(self):
        # start a thread to read frames from the file video stream
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True
        self.stopped = False
        self.t.start()

    def update(self):
        # keep looping infinitely
        while not self.stopped:
            try:
                raw_image = self.pipe.stdout.read(self.frameBytes)
                frame = np.frombuffer(raw_image, dtype='uint8').reshape((self.h, self.w))
                #frame = np.frombuffer(raw_image[:self.kBytes], dtype='uint8').reshape((self.h, self.w))
                self.Q.put(frame)
                self.buff_count += 1
            except:
                print('Stopped Reading Movie')
                self.stop()

    def read(self):
        if self.stopped and (self.read_count >= self.buff_count):
            print('Stopped and Queue Depleted')
            return []
        frame = self.Q.get()
        self.Q.task_done()
        self.read_count += 1
        return frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        # self.t.join()

#%%
if __name__ == '__main__':
    vidPath = "Z:\\Selmaan\\Calibration Data\\vsync methods\\vsync_auto_200511_104847\\lFront.avi"
    w=2816
    h=1408
    s = 1
    f = ffVideoReader(vidPath, w=w, h=h, s=s, queueSize=1000)