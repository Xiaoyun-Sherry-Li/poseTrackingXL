%% Check and relabel inferred frames for a semi-manual training set
clear all
close all;
clc
codePath = 'C:\Users\User\Documents\GitHub\Label3D';
addpath(genpath(codePath))
cd 'C:\Users\User\Documents\GitHub\bird_pose_tracking\training_files\Label3D'

%% Set file paths and load results from SLEAP output 
vidPath = 'Z:\Sherry\acquisition\AMB151_071025'; % behavioral session
load(fullfile(vidPath, 'new_081525_posture.mat'));

%% Read in and reformat camera parameter array
load('Z:\Sherry\camera_calibration\092124_camOptArrayDA_XL.mat')
camNames = {'blue_cam', 'green_cam', 'red_cam', 'yellow_cam'};
nCams = length(camNames);

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
skeleton_file = 'C:\Users\xl313\OneDrive\Documents\GitHub\bird_pose_tracking\postureNet\posture_skeleton_IL.csv';
opts = detectImportOptions(skeleton_file);
opts.SelectedVariableNames = 1:3;
skeleton_info = readmatrix(skeleton_file, opts);

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

%% com or posture skeletons 
% posture skeleton
skeleton.joints_idx = repmat(joint_idx, 2, 1);
skeleton.color = lines(length(skeleton.joints_idx)); % 15 body parts -> 15 distinct colors

%% load selected frames for making a sanity check video
% inputs
cd 'Z:\Sherry\acquisition\LVN4_040725' % path
FPS = 50;
predStart = 2000; % in seconds % successful retrieval detection 
predFrames = 3 * FPS; % duration, in frames 
predIdx = predStart * FPS : predStart * FPS + predFrames - 1;

% read in frames
predVid = cell(nCams,1);
for cam_idx = 1:nCams
    disp(camNames(cam_idx))
    fn = fullfile(vidPath, [camNames{cam_idx}, '.avi']);
    reader = VideoReader(fn);
    frame_rate = reader.FrameRate;
    vid = zeros(reader.Height, reader.Width, 3, predFrames, 'uint8');
    for f = 1:predFrames
        reader.CurrentTime = predIdx(f)/frame_rate;
        frameRGB = readFrame(reader);
        vid(:,:,:,f) = frameRGB;
    end
    predVid{cam_idx} = vid;
end

%% view3d 
close all
viewGui = View3D(allParams, predVid, skeleton);
viewGui.defScale= 40;
colormap(viewGui.h{1}.Parent, 'gray');

% Load Posture Data
pts3d_posture = permute(results.posture_preds, [1, 3, 2]);
viewGui.loadFrom3D(pts3d_posture(predIdx, :, :));

%% Create a video 
v = VideoWriter(fullfile(vidPath,'interaction'),'MPEG-4');
v.Quality=95;
v.FrameRate = 10;
v.open,
for i = 1:predFrames
    viewGui.setFrame(i);
    viewGui.triangulateView();
    viewGui.resetAspectRatio();
    F = getframe(viewGui.Parent); % grab a frame from viewGUI.Parent - a handle to an axis
    v.writeVideo(F.cdata);
end
v.close