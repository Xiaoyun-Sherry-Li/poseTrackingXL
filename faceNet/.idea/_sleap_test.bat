@echo off
call C:\Users\User\miniforge3\condabin\conda.bat activate sleap > C:\Users\User\Documents\GitHub\poseTrackingXL\faceNet\.idea\_sleap_out.txt 2>&1
python -c "import numpy as np; print('numpy', np.__version__)" >> C:\Users\User\Documents\GitHub\poseTrackingXL\faceNet\.idea\_sleap_out.txt 2>&1
python -c "import zmq, sleap; print('zmq', zmq.__version__, 'sleap OK')" >> C:\Users\User\Documents\GitHub\poseTrackingXL\faceNet\.idea\_sleap_out.txt 2>&1
