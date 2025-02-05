%% Check and relabel inferred frames for a semi-manual training set
clear all
close all;

%% Set file paths and load results from SLEAP inference
bird_id = 'SLV123';
session_date = '110824_wEphys';
session_root = ['Z:\Sherry\poseTrackingXL\training_files\raw_acquisition_copy\'];
session_dir = [bird_id, '_', session_date];
results_file = '011125_posture_2stage.mat';
load(fullfile(session_root, session_dir, results_file))
% define the video file
vidPath = fullfile(session_root, session_dir);
% load('Z:\Sherry\acquisition\ROS103_090324\091124_posture_2stage.mat');

%%
codePath = 'C:\Users\xl313\OneDrive\Documents\GitHub\Label3D';
addpath(genpath(codePath))

tracking_root = 'C:\Users\xl313\OneDrive\Documents\GitHub\bird_pose_tracking\';
cd 'C:\Users\xl313\OneDrive\Documents\GitHub\bird_pose_tracking\training_files\Label3D'
% alignPath = [tracking_root, 'calibration_files\all_opt_arrays'];
% cam_array_file = '240903_aligned_opt_cam_array.mat'; % calib from 7/16/24

alignPath = 'Z:\Sherry\camera_calibration\';
cam_array_file = '092124_camOptArrayDA_XL.mat';

skeleton_path = [tracking_root, 'postureNet\'];
skeleton_file = 'posture_skeleton_IL.csv';

%% Read in and reformat camera array
camNames = {'blue_cam', 'green_cam', 'red_cam', 'yellow_cam'};
nCams = length(camNames);

load(fullfile(alignPath, cam_array_file))
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

%% Get indices for high/low confidence frames
avg_rep_err = median(posture_reproj, 2);
avg_conf = median(posture_conf, 2);

% high confidence frames
low_rep_err = avg_rep_err < prctile(avg_rep_err, 95); % originally 0.5
high_conf = avg_conf > prctile(avg_conf, 95); % originally 99.5
good_frame_idx = low_rep_err & high_conf;


% low confidence frames
high_rep_err = avg_rep_err > prctile(avg_rep_err, 90); % used to be 99.9
low_conf = avg_conf < prctile(avg_conf, 5); % used to be 0.1
bad_frame_idx = high_rep_err & low_conf;


%% Load the good video frames
% data params
nFrames = sum(good_frame_idx);
maxFrames = length(good_frame_idx);
all_frames = 35500:(35500 + maxFrames);
frame_idx = all_frames(good_frame_idx);

% %%% TMP
% nFrames = 200;
% maxFrames = 200;
% all_frames = 3500:200;
% frame_idx = all_frames;
start_frame = 35500;

% read in images
good_videos = cell(nCams,1);
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
    good_videos{cam_idx} = vid;
end

    % reader = VideoReaderFFMPEG(fn, 'FFMPEGPATH', ffmpegPath);
%     frame_rate = reader.FrameRate;
%     vid = zeros(reader.Height, reader.Width, 3, nFrames, 'uint8');
%     for f = 1:nFrames
%         f_idx = frame_idx(f);
%         frameRGB = reader.read(round(f_idx));
%         vid(:,:,:,f) = frameRGB;
%     end
%     good_videos{cam_idx} = vid;
% end

%% Makes a Label3D object and start the GUI (high conf)
% To confirm that high conf frames look good
pts3d = permute(posture_preds, [1, 3, 2]);
labelGui = Label3D(allParams, good_videos, skeleton, 'defScale', .12);
labelGui.loadFrom3D(pts3d(good_frame_idx, :, :));
%labelGui.loadFrom3D(pts3d)
colormap(labelGui.h{1}.Parent, 'gray'),

%% Save as a training file
% Done through 85, but need to re-check early frames esp face
save_dir = 'Z:\Isabel\data\training_data\';
save_file = [bird_id, '_', session_date, '_inferred']; % good frames
labelGui.savePath = fullfile(save_dir, save_file);
labelGui.saveAll()


%% Load the bad video frames
% data params
start_frame = 35500;
nFrames = sum(bad_frame_idx);
maxFrames = length(bad_frame_idx);
all_frames = start_frame:(start_frame + maxFrames);
frame_idx = all_frames(bad_frame_idx);

% read in images
bad_videos = cell(nCams,1);
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
    bad_videos{cam_idx} = vid;
end
%% Makes a Label3D object and start the GUI (low conf)
% To identify and correct failure modes
%pts3d = permute(posture_preds, [1, 3, 2]);
labelGui = Label3D(allParams, bad_videos, skeleton, 'defScale', .12);
labelGui.loadFrom3D(pts3d(bad_frame_idx, :, :));
colormap(labelGui.h{1}.Parent, 'gray'),

%% Save as a training file
save_dir = 'Z:\Isabel\data\training_data\';
save_file = [bird_id, '_', session_date, '_mistakes_manual']; % bad frames
labelGui.savePath = fullfile(save_dir, save_file);
labelGui.saveAll()

%% To load an existing training file
training_file = [save_file, '.mat'];
labelGui = Label3D(fullfile(save_dir, training_file), 'defScale', .12);
colormap(labelGui.h{1}.Parent, 'gray'),











%%%% 
%% Makes a Label3D object and start the GUI (high conf)
% To confirm that high conf frames look good
labelGui = Label3D(camParams, label_frames, skeleton, 'defScale', .12);
% labelGui.loadFrom3D(data_3D)
colormap(labelGui.h{1}.Parent, 'gray'),

%%
points_3d = labelGui.points3D;
pts3d = points_3d;
pts3d = reshape(pts3d, size(pts3d, 1), 3, []);
pts3d = permute(pts3d, [3, 2, 1]);


