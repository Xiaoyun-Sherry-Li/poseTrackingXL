% this script is used to just convert videos from all cameras to individual
% consecutive frame to feed in LP predict. 
clear all
close all;

%% Set file paths
% update the file paths and cam_array_file as needed
codePath = 'C:\Users\xl313\OneDrive\Documents\GitHub\Label3D\'; % location of all code scripts
addpath(genpath(codePath));
addpath('Z:\Sherry\poseTrackingXL\training_files\raw_acquisition_copy\SLV123_110824_noephys\'); % location of bird raw videos in mp4 format
addpath('C:\Users\xl313\OneDrive\Documents\GitHub\Label3D\');
addpath('Z:\Sherry\poseTrackingXL\training_files\');

%% Define chickadee training video files
folderName  = 'Z:\Sherry\poseTrackingXL\training_files\raw_acquisition_copy\SLV123_110824_noephys\' 
trainingVideoFiles = { ...
 [folderName 'blue_cam.avi'], ...
 [folderName 'green_cam.avi'], ...
 [folderName 'red_cam.avi'], ...
 [folderName 'yellow_cam.avi'] 
 };

cam_array_file = 'Z:\Sherry\camera_calibration\240911_aligned_opt_cam_array_XL.mat'; % this is a correct file that I've double checked
load(cam_array_file) % output: camParams

skeleton_path = ['Z:\Sherry\poseTrackingXL\postureNet\'];
skeleton_file = 'posture_skeleton_IL.csv';

% define the video file
bird_id = 'SLV123';
session_date = '110824_noephys';

%% Define frame extraction parameters 
nFrames = 24;
startFrames = 650 * 50; % Dec 19, 2024: 
endFrames = 670 * 50; % (input) minutes * 50 Hz * 60 seconds 
nCams = 4;
frame_idx = startFrames: endFrames; % for uniform spacing

%% to load both unlabelled and label frames 
videos = cell(nCams,1);
for cam_idx = 1:nCams
    v = VideoReader(trainingVideoFiles{cam_idx});
    vid = zeros(v.Height, v.Width, 3, nFrames, 'uint8');
    frame_counter = 1;
    for f = frame_idx/50
        v.CurrentTime = f;
        img = readFrame(v);
        %img = read(v,f);
        vid(:,:,:,frame_counter) = img(:, :, :);
        frame_counter = frame_counter + 1;
    end
    videos{cam_idx} = vid;
end

%% Check output
figure;
% Plot different camera views
subplot(1, 2, 1);
imshow(videos{1,1}(:,:,:,500));
subplot(1, 2, 2);
imshow(videos{2,1}(:,:,:,500));

%% Extract label frames for Label3D input
label_frames = cell(nCams, 1);

for cam_idx = 1:nCams
    for i = 1:numel(frame_idx)
        label_frames{cam_idx}(:,:,:,i) = videos{cam_idx,1}(:,:,:,frame_idx);
    end
end 

%% 
% Corrresponding video frames + frame indices + labelled & unlabelled frames attached for
% Lightning Pose 

%dataFolder = ['/Users/cutiecorgi/PycharmProjects/bird_pose_tracking/training_files/Label3D/']; % tmp store locally
dataFolder = ['Z:\Sherry\poseTrackingXL\training_files\LP_data\']
dataPath = fullfile(dataFolder,[bird_id,'_',session_date,'_',num2str(nFrames),'_start',num2str(startFrames),'_end',num2str(endFrames),'_LP_ctx_unseen_inference_frames.mat']);
save(dataPath,'videos','frame_idx','camParams','-v7.3')

%%
img = ;
% Check the data type of the image
imgType = class(video{1,1});