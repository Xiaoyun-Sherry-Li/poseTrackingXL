%% Load in label3D and label the key points:
clear all
close all;
codePath = 'C:\Users\xl313\OneDrive\Documents\GitHub\Label3D';
addpath(genpath(codePath))

%% Define training video files
folderName  = 'Z:\Sherry\acquisition\SLV123_110824_wEphys\';
calib.trainingVideoFiles = { ...
 [folderName 'blue_cam.avi'], ...
 [folderName 'green_cam.avi'], ...
 [folderName 'red_cam.avi'], ...
 [folderName 'yellow_cam.avi'] };

% Import body skeletons
calib.trainingPointSkeleton = 'D:\GitHub\spike-analysis\bird_pose_tracking\postureNet\posture_skeleton_IL.csv';
skeleton_path = ['C:\Users\xl313\OneDrive\Documents\GitHub\bird_pose_tracking\postureNet'];
skeleton_file = 'posture_skeleton_IL.csv';

cam_array_file = 'Z:\Sherry\camera_calibration\092124_camOptArrayDA_XL.mat'; % this is a correct file that I've double checked

% define the video file
vidPath = 'Z:\Sherry\acquisition\SLV123_110824_wEphys';

% ensure that the Label3D file autosaves in the right place
save_dir = 'Z:\Sherry\poseTrackingXL\training_files\Label3D\';

%% Read in video for each camera
% data params
camNames = {'blue_cam', 'green_cam', 'red_cam', 'yellow_cam'};
nCams = length(camNames);
nFrames = 15; % 100 for AMB104
maxFrames = 50*60*105; % Hz * seconds * minutes
frame_idx = round(linspace(5*60*50, maxFrames, nFrames)); % for uniform spacing
videos = cell(nCams,1);

for cam_idx = 1:nCams
    disp(camNames(cam_idx))
    fn = fullfile(vidPath, [camNames{cam_idx}, '.avi']);
    reader = VideoReader(fn);
    frame_rate = reader.FrameRate;
    vid = zeros(reader.Height, reader.Width, 3, nFrames, 'uint8');
    for f = 1:nFrames
        reader.CurrentTime = frame_idx(f)/frame_rate;
        frameRGB = readFrame(reader);
        vid(:,:,:,f) = frameRGB;
    end
    videos{cam_idx} = vid;
end

%% Read in and reformat camera array
load(fullfile(cam_array_file))
allParams = cell(nCams, 1);
for cam_idx = 1:nCams
    f = optCamArrayXL(cam_idx, 7);
    tmp = struct;
    prinpoint = optCamArrayXL(cam_idx,10:11);
    tmp.K = cat(1,[f, 0, 0], [0, f, 0], [prinpoint, 1]);
    tmp.RDistort = optCamArrayXL(cam_idx,8:9);
    tmp.TDistort = [0, 0];
    tmp.r = rotvec2mat3d(optCamArrayXL(cam_idx,1:3));
    tmp.t = optCamArrayXL(cam_idx,4:6);
    allParams{cam_idx} = tmp;
end

%% Define the skeleton
% load the csv file
opts = detectImportOptions(fullfile(skeleton_path, skeleton_file));
opts.SelectedVariableNames = [1:3];
skeleton_info = readmatrix(fullfile(skeleton_path, skeleton_file), opts);

% define the nodes and edges
skeleton.joint_names = transpose(skeleton_info(:,1));
n_keypoints = length(skeleton.joint_names);
all_parents = [];
joint_idx = [];
for node_idx = 1:n_keypoints
    parent = skeleton_info{node_idx, 2};
    if ~isempty(parent)
        match_mask = strcmp(skeleton_info(:, 1), parent);
        parent_idx = find(match_mask);
        if isempty(joint_idx)
        joint_idx = [node_idx, parent_idx];
        else
            joint_idx = cat(1, joint_idx, [node_idx, parent_idx]);
        end
    end
end
skeleton.joints_idx = repmat(joint_idx, 2, 1);
skeleton.color = lines(length(skeleton.joints_idx));

%% Start the GUI
labelGui = Label3D(allParams, videos, skeleton, 'defScale', 35); % original defscale .12
colormap(labelGui.h{1}.Parent, 'gray'),

%% To save the training file
% the auto save doesn't include the video files...
labelGui.saveAll()
% MANUAL: change the name of the saved Label3D GUI to bird name_sleap
% format 

%% To load an existing training file
close all
clear all
clc
load('Z:\Sherry\poseTrackingXL\training_files\Label3D\20250312_162059_Label3D_videos.mat');
labelGui = Label3D(camParams, videos, skeleton, 'defScale', 35);
labelGui.loadFrom3D(data_3D);
colormap(labelGui.h{1}.Parent, 'gray'),
