%% Load in Diego Aldorondo's labeling GUI and label the key points:
% lightly modified from SC's script:
% RigControl/Camera Alignments/arena_coordinate_transform_2.m
clear all
close all;

%% Set file paths
% update the file paths and cam_array_file as needed
%codePath = 'C:\Users\ilow1\Documents\code\Label3D';
%addpath(genpath(codePath))

%% Define chickadee training video files
folderName  = 'Z:\Sherry\acquisition\SLV123_100424\';
calib.trainingVideoFiles = { ...
 [folderName 'blue_cam.avi'], ...
 [folderName 'green_cam.avi'], ...
 [folderName 'red_cam.avi'], ...
 [folderName 'yellow_cam.avi'] };
% Import chickadee body skeletons
calib.trainingPointSkeleton = 'D:\GitHub\spike-analysis\bird_pose_tracking\postureNet\posture_skeleton_IL.csv';

%tracking_root = 'C:\Users\ilow1\Documents\code\bird_pose_tracking\';
%alignPath = [tracking_root, 'calibration_files\all_opt_arrays'];
cam_array_file = 'Z:\Sherry\camera_calibration\240911_aligned_opt_cam_array_XL.mat'; % this is a correct file that I've double checked

skeleton_path = ['C:\Users\xl313\OneDrive\Documents\GitHub\bird_pose_tracking\postureNet\'];
skeleton_file = 'posture_skeleton_IL.csv';

% define the video file
% bird_id = 'SLV123';
% session_date = '100424';
% vidRoot = ['Z:\Sherry\acquisition\', bird_id, '\'];

bird_id = 'AMB155';
session_date = '100424_01';
vidRoot = ['Z:\Sherry\acquisition\', bird_id, '\'];
vidFolder = [bird_id, '_', session_date];
vidPath = fullfile(vidRoot, vidFolder);

%% Read in the video for each camera
% data params
camNames = {'red_cam', 'yellow_cam', 'green_cam', 'blue_cam'};
nFrames = 5; % 100 for AMB104
%maxFrames = 50*60*100; % 50 Hz * 60 seconds * 100 minutes
nCams = length(camNames);
%frame_idx = round(linspace(5*60*50, maxFrames, nFrames)); % for uniform spacing

%%
% read in and reformat camera array
load(cam_array_file) % output: camParams
%%
frame_idx = [1 2 3 4 5];

videos = cell(nCams,1);
for cam_idx = 1:nCams
    v = VideoReader(calib.trainingVideoFiles{cam_idx});
    vid = zeros(v.Height, v.Width, 3, nFrames, 'uint8');
    for f = frame_idx
        v.CurrentTime = f * 500;
        %img = readFrame(v);
        img = read(v,f);
        vid(:,:,:,f) = img(:, :, :);
    end
    videos{cam_idx} = vid;
end


