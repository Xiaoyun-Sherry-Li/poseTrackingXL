%% Check and relabel inferred frames for a semi-manual training set
clear all
close all;
clc
codePath = 'C:\Users\xl313\OneDrive\Documents\GitHub\Label3D';
addpath(genpath(codePath))
cd 'C:\Users\xl313\OneDrive\Documents\GitHub\bird_pose_tracking\training_files\Label3D'

%% [change this] Set file paths and load results from SLEAP output 
vidPath = 'Z:\Sherry\acquisition\RBY52_2ndPart_012425'; % behavioral session
load(fullfile(vidPath,'032325_posture_face.mat')); % results

%% Read in and reformat camera parameter array
load('Z:\Sherry\camera_calibration\092124_camOptArrayDA_XL.mat') % "optCamArrayXL"
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

% posture skeleton
skeleton.joints_idx = repmat(joint_idx, 2, 1);
skeleton.color = lines(length(skeleton.joints_idx)); % 15 body parts -> 15 distinct colors

%  the following sessions identify good and bad individual frames
%% visualise comNet performances
figure;
lw_hist = 1.5; % Line width for histograms

% First subplot: Model confidence
ax(1) = subplot(1, 3, 1);
hold(ax(1), 'on');
histogram(ax(1), results.com_conf(:, 1), 100, 'EdgeColor', [0, 0.5, 1], 'LineWidth', lw_hist, 'DisplayStyle', 'stairs', 'DisplayName', 'head');
histogram(ax(1), results.com_conf(:, 2), 100, 'EdgeColor', [1, 0, 0], 'LineWidth', lw_hist, 'DisplayStyle', 'stairs', 'DisplayName', 'body');
histogram(ax(1), results.com_conf(:, 3), 100, 'EdgeColor', [1, 1, 0], 'LineWidth', lw_hist, 'DisplayStyle', 'stairs', 'DisplayName', 'tail');
title(ax(1), 'Model Confidence');
xlabel(ax(1), 'Confidence');
hold(ax(1), 'off');

% Second subplot: Reprojection error (98th percentile)
pct98 = prctile(results.com_rep_err, 98);
ax(2) = subplot(1, 3, 2);
hold(ax(2), 'on');
histogram(ax(2), results.com_rep_err(results.com_rep_err(:, 1) < pct98(1), 1), 100, 'EdgeColor', [0, 0.5, 1], 'LineWidth', lw_hist, 'DisplayStyle', 'stairs', 'DisplayName', 'head');
histogram(ax(2), results.com_rep_err(results.com_rep_err(:, 2) < pct98(2), 2), 100, 'EdgeColor', [1, 0, 0], 'LineWidth', lw_hist, 'DisplayStyle', 'stairs', 'DisplayName', 'body');
histogram(ax(2), results.com_rep_err(results.com_rep_err(:, 3) < pct98(3), 3), 100, 'EdgeColor', [1, 1, 0], 'LineWidth', lw_hist, 'DisplayStyle', 'stairs', 'DisplayName', 'tail');
title(ax(2), 'Reprojection Error (98%)');
xlabel(ax(2), 'Error (pixels)');
hold(ax(2), 'off');

% Third subplot: Reprojection error (100%)
ax(3) = subplot(1, 3, 3);
hold(ax(3), 'on');
histogram(ax(3), results.com_rep_err(:, 1), 100, 'EdgeColor', [0, 0.5, 1], 'LineWidth', lw_hist, 'DisplayStyle', 'stairs', 'DisplayName', 'head');
histogram(ax(3), results.com_rep_err(:, 2), 100, 'EdgeColor', [1, 0, 0], 'LineWidth', lw_hist, 'DisplayStyle', 'stairs', 'DisplayName', 'body');
histogram(ax(3), results.com_rep_err(:, 3), 100, 'EdgeColor', [1, 1, 0], 'LineWidth', lw_hist, 'DisplayStyle', 'stairs', 'DisplayName', 'tail');
legend(ax(3), 'Location', 'northeastoutside');
title(ax(3), 'Reprojection Error (100%)');
xlabel(ax(3), 'Error (pixels)');
hold(ax(3), 'off');

%% visualise postureNet performances
figure;
lw_hist = 1.5; % Line width for histograms

% First subplot: Model confidence
ax(1) = subplot(1, 3, 1);
hold(ax(1), 'on');
histogram(ax(1), results.posture_conf(:, 1), 100, 'EdgeColor', [0, 0.5, 1], 'LineWidth', lw_hist, 'DisplayStyle', 'stairs', 'DisplayName', 'beak');
histogram(ax(1), results.posture_conf(:, 11), 100, 'EdgeColor', [1, 0, 0], 'LineWidth', lw_hist, 'DisplayStyle', 'stairs', 'DisplayName', 'left foot');
histogram(ax(1), results.posture_conf(:, 3), 100, 'EdgeColor', [1, 1, 0], 'LineWidth', lw_hist, 'DisplayStyle', 'stairs', 'DisplayName', 'tail');
title(ax(1), 'Model Confidence');
xlabel(ax(1), 'Confidence');
hold(ax(1), 'off');

% Second subplot: Reprojection error (95th percentile)
pct95 = prctile(results.posture_rep_err, 95);
ax(2) = subplot(1, 3, 2);
hold(ax(2), 'on');
histogram(ax(2), results.posture_rep_err(results.posture_rep_err(:, 1) < pct95(1), 1), 100, 'EdgeColor', [0, 0.5, 1], 'LineWidth', lw_hist, 'DisplayStyle', 'stairs', 'DisplayName', 'beak');
histogram(ax(2), results.posture_rep_err(results.posture_rep_err(:, 11) < pct95(2), 2), 100, 'EdgeColor', [1, 0, 0], 'LineWidth', lw_hist, 'DisplayStyle', 'stairs', 'DisplayName', 'left foot');
histogram(ax(2), results.posture_rep_err(results.posture_rep_err(:, 3) < pct95(3), 3), 100, 'EdgeColor', [1, 1, 0], 'LineWidth', lw_hist, 'DisplayStyle', 'stairs', 'DisplayName', 'tail');
title(ax(2), 'Reprojection Error (95%)');
xlabel(ax(2), 'Error (pixels)');
hold(ax(2), 'off');

% Third subplot: Reprojection error (100%)
ax(3) = subplot(1, 3, 3);
hold(ax(3), 'on');
histogram(ax(3), results.posture_rep_err(:, 1), 100, 'EdgeColor', [0, 0.5, 1], 'LineWidth', lw_hist, 'DisplayStyle', 'stairs', 'DisplayName', 'beak');
histogram(ax(3), results.posture_rep_err(:, 11), 100, 'EdgeColor', [1, 0, 0], 'LineWidth', lw_hist, 'DisplayStyle', 'stairs', 'DisplayName', 'left foot');
histogram(ax(3), results.posture_rep_err(:, 3), 100, 'EdgeColor', [1, 1, 0], 'LineWidth', lw_hist, 'DisplayStyle', 'stairs', 'DisplayName', 'tail');
legend(ax(3), 'Location', 'northeastoutside');
title(ax(3), 'Reprojection Error (100%)');
xlabel(ax(3), 'Error (pixels)');
hold(ax(3), 'off');

%% high confidence & low reprojection error frames
avg_rep_err = median(results.posture_rep_err, 2);
avg_conf = median(results.posture_conf, 2);
low_rep_err = avg_rep_err < prctile(avg_rep_err, 5) & avg_rep_err > prctile(avg_rep_err, 3); % originally 0.5
high_conf = avg_conf > prctile(avg_conf, 98); % originally 99.5
good_frame_idx = low_rep_err & high_conf;

% low confidence & low reprojection error frames
high_rep_err = avg_rep_err > prctile(avg_rep_err, 99); % used to be 99.9
low_conf = avg_conf < prctile(avg_conf, 0.5); % used to be 0.1
bad_frame_idx = high_rep_err & low_conf;

%% Load the bad video frames
% data params
start_frame = 0;
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
labelGui = Label3D(allParams, bad_videos, skeleton, 'defScale', 35);
pts3d = permute(results.posture_preds, [1, 3, 2]);
% bad_pts = pts3d(bad_frame_idx, :, :);
% labelGui.loadFrom3D(bad_pts);
colormap(labelGui.h{1}.Parent, 'gray'),

%% Save as a training file
save_dir = 'Z:\Sherry\poseTrackingXL\training_files\Label3D\';
save_file = 'RBY52_2ndPart_012425_inferred_mistake'; 
labelGui.savePath = fullfile(save_dir, save_file);
labelGui.saveAll()


%% Load the good video frames
% data params
nFrames = sum(good_frame_idx);
maxFrames = length(good_frame_idx);
all_frames = 0:(0 + maxFrames);
frame_idx = all_frames(good_frame_idx);
start_frame = 0;

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

%% Makes a Label3D object and start the GUI (low conf)
% To identify and correct failure modes
labelGui = Label3D(allParams, good_videos, skeleton, 'defScale', 35);
pts3d = permute(results.posture_preds, [1, 3, 2]);
good_pts = pts3d(good_frame_idx, :, :);
labelGui.loadFrom3D(good_pts);
colormap(labelGui.h{1}.Parent, 'gray'),
%% the modified points will them be automatically saved as data. To merged
% with postureNet output and visualise them: open the automatically saved
% files 
good_pts = reshape(good_pts, size(data_3D,1), []);
good_pts(~isnan(data_3D)) = data_3D(~isnan(data_3D));
labelGui.loadFrom3D(good_pts);

%% Save as a training file
save("Z:\Sherry\poseTrackingXL\training_files\Label3D\RBY52_2ndPart_012425_109goodFrames_videos_sleap.mat", "camParams", "videos", "data_3D", "skeleton")


%% [optional]: view the saved output 
load("Z:\Sherry\poseTrackingXL\training_files\Label3D\RBY52_2ndPart_012425_109goodFramesV1_videos.mat")
labelGui = Label3D(camParams, videos, skeleton, 'defScale', 35);
labelGui.loadFrom3D(data_3D);