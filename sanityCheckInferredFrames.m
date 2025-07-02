%% Check and relabel inferred frames for a semi-manual training set
clear all
close all;
clc
codePath = 'C:\Users\xl313\OneDrive\Documents\GitHub\Label3D';
addpath(genpath(codePath))
cd 'C:\Users\xl313\OneDrive\Documents\GitHub\bird_pose_tracking\training_files\Label3D'

%% [change this] Set file paths and load results from SLEAP output 
vidPath = 'Z:\Sherry\acquisition\LVN4_040725'; % behavioral session
% load(fullfile(vidPath, '032325_posture_face.mat')); % results
% load(fullfile(vidPath,'behavioral_data\posture_pos_smooth_032325.mat')); % load the smoothed version
load(fullfile(vidPath, '050125_posture.mat'));

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
pct98 = prctile(results.com_reproj, 98);
ax(2) = subplot(1, 3, 2);
hold(ax(2), 'on');
histogram(ax(2), results.com_reproj(results.com_reproj(:, 1) < pct98(1), 1), 100, 'EdgeColor', [0, 0.5, 1], 'LineWidth', lw_hist, 'DisplayStyle', 'stairs', 'DisplayName', 'head');
histogram(ax(2), results.com_reproj(results.com_reproj(:, 2) < pct98(2), 2), 100, 'EdgeColor', [1, 0, 0], 'LineWidth', lw_hist, 'DisplayStyle', 'stairs', 'DisplayName', 'body');
histogram(ax(2), results.com_reproj(results.com_reproj(:, 3) < pct98(3), 3), 100, 'EdgeColor', [1, 1, 0], 'LineWidth', lw_hist, 'DisplayStyle', 'stairs', 'DisplayName', 'tail');
title(ax(2), 'Reprojection Error (98%)');
xlabel(ax(2), 'Error (pixels)');
hold(ax(2), 'off');

% Third subplot: Reprojection error (100%)
ax(3) = subplot(1, 3, 3);
hold(ax(3), 'on');
histogram(ax(3), results.com_reproj(:, 1), 100, 'EdgeColor', [0, 0.5, 1], 'LineWidth', lw_hist, 'DisplayStyle', 'stairs', 'DisplayName', 'head');
histogram(ax(3), results.com_reproj(:, 2), 100, 'EdgeColor', [1, 0, 0], 'LineWidth', lw_hist, 'DisplayStyle', 'stairs', 'DisplayName', 'body');
histogram(ax(3), results.com_reproj(:, 3), 100, 'EdgeColor', [1, 1, 0], 'LineWidth', lw_hist, 'DisplayStyle', 'stairs', 'DisplayName', 'tail');
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
pct95 = prctile(results.posture_reproj, 95);
ax(2) = subplot(1, 3, 2);
hold(ax(2), 'on');
histogram(ax(2), results.posture_reproj(results.posture_reproj(:, 1) < pct95(1), 1), 100, 'EdgeColor', [0, 0.5, 1], 'LineWidth', lw_hist, 'DisplayStyle', 'stairs', 'DisplayName', 'beak');
histogram(ax(2), results.posture_reproj(results.posture_reproj(:, 11) < pct95(2), 2), 100, 'EdgeColor', [1, 0, 0], 'LineWidth', lw_hist, 'DisplayStyle', 'stairs', 'DisplayName', 'left foot');
histogram(ax(2), results.posture_reproj(results.posture_reproj(:, 3) < pct95(3), 3), 100, 'EdgeColor', [1, 1, 0], 'LineWidth', lw_hist, 'DisplayStyle', 'stairs', 'DisplayName', 'tail');
title(ax(2), 'Reprojection Error (95%)');
xlabel(ax(2), 'Error (pixels)');
hold(ax(2), 'off');

% Third subplot: Reprojection error (100%)
ax(3) = subplot(1, 3, 3);
hold(ax(3), 'on');
histogram(ax(3), results.posture_reproj(:, 1), 100, 'EdgeColor', [0, 0.5, 1], 'LineWidth', lw_hist, 'DisplayStyle', 'stairs', 'DisplayName', 'beak');
histogram(ax(3), results.posture_reproj(:, 11), 100, 'EdgeColor', [1, 0, 0], 'LineWidth', lw_hist, 'DisplayStyle', 'stairs', 'DisplayName', 'left foot');
histogram(ax(3), results.posture_reproj(:, 3), 100, 'EdgeColor', [1, 1, 0], 'LineWidth', lw_hist, 'DisplayStyle', 'stairs', 'DisplayName', 'tail');
legend(ax(3), 'Location', 'northeastoutside');
title(ax(3), 'Reprojection Error (100%)');
xlabel(ax(3), 'Error (pixels)');
hold(ax(3), 'off');


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

% com skeleton (uncomment to produce a com keypoint video)
% skeleton.joint_names = {'Head','Body','Tail'};
% skeleton.joints_idx = [[1,1];[2,2];[3,3]];
% skeleton.color = lines(length(skeleton.joints_idx));

%% [optional] load selected frames for making a sanity check video
% 28-35 sec 
FPS = 50;
predStart = 661402/50; % in seconds % successful retrieval detection 
predFrames = 3 * FPS; % duration, in frames 
predIdx = predStart * FPS : predStart * FPS + predFrames - 1;

% load(fullfile(vidPath,"annotatedSeeds.mat"));
% [cacheInteractions, cacheSiteID] = find(annotatedSeeds.seedChanges == 1);
% % find the start & end frame idx of caches and retrievals
% cacheOnsetFrame = annotatedSeeds.countData.newSite(cacheInteractions); % frame idx for start of site interactions 
% cacheOffsetFrame = annotatedSeeds.countData.endSite(cacheInteractions); % frame idx for end of site interactions 
% predFrames = cacheOffsetFrame(2) - cacheOnsetFrame(2) + 1;
% predIdx = cacheOnsetFrame(2):cacheOffsetFrame(2); % tmp, to visualise frames during cache interaction 

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

%% SC's view3d 
close all
viewGui = View3D(allParams, predVid, skeleton);
viewGui.defScale= 40;
colormap(viewGui.h{1}.Parent, 'gray');

% Load COM Data (uncomment next line, comment the next session to produce comNet keypoint predictions video)
% pts3d_com = permute(results.com_preds, [1, 3, 2]);
% viewGui.loadFrom3D(pts3d_com(predIdx, :, :));

% Load Posture Data
pts3d_posture = permute(results.posture_preds, [1, 3, 2]);
% pts3d_posture = permute(pos_pts_smooth, [1, 3, 2]);
viewGui.loadFrom3D(pts3d_posture(predIdx, :, :));


%% Create a video 
% cd 'Z:\Sherry\acquisition\RBY52_2ndPart_012425'  
cd 'Z:\Sherry\acquisition\LVN4_040725'
v = VideoWriter(fullfile(vidPath,'missed_interaction'),'MPEG-4');
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

%%  the following sessions identify good and bad individual frames
% Get an idea of the distribution of confidence levels
% Get the median values for all body parts
avg_rep_err = median(results.posture_rep_err, 2);
avg_conf = median(results.posture_conf, 2);

hist(avg_rep_err(avg_rep_err < prctile(avg_rep_err, 95)))
% hist(avg_rep_err)

%%
% high confidence & low reprojection error frames
low_rep_err = avg_rep_err < prctile(avg_rep_err, 0.5); % originally 0.5
high_conf = avg_conf > prctile(avg_conf, 99.5); % originally 99.5
good_frame_idx = low_rep_err & high_conf;

% low confidence & low reprojection error frames
high_rep_err = avg_rep_err > prctile(avg_rep_err, 99.9); % used to be 99.9
low_conf = avg_conf < prctile(avg_conf, 0.1); % used to be 0.1
bad_frame_idx = high_rep_err & low_conf;

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

%% Makes a Label3D object and start the GUI (high conf)
close all
% To confirm that high conf frames look good
pts3d = permute(results.posture_preds, [1, 3, 2]);
labelGui = Label3D(allParams, good_videos, skeleton, 'defScale', 35);
labelGui.loadFrom3D(pts3d(good_frame_idx, :, :));
colormap(labelGui.h{1}.Parent, 'gray'),

%% Before closing the GUI, save as a training file
save_dir = 'Z:\Sherry\poseTrackingXL\training_files\Label3D\';
save_file = 'AMB155_031025_inferred'; % good frames

labelGui.savePath = fullfile(save_dir, save_file);
labelGui.saveAll()

%% To integrate adjusted points with the unmodified points and save again
close all
load(fullfile(save_dir, save_file));
labelGui = Label3D(camParams, videos, skeleton, 'defScale', 35);
pts3d = permute(results.posture_preds, [1, 3, 2]);
good_pts = pts3d(good_frame_idx, :, :);
good_pts = reshape(good_pts, 72, []);
good_pts(~isnan(data_3D)) = data_3D(~isnan(data_3D));
labelGui.loadFrom3D(good_pts);
colormap(labelGui.h{1}.Parent, 'gray'),

%% Final save
save_dir = 'Z:\Sherry\poseTrackingXL\training_files\Label3D\';
save_file = 'AMB155_031025_inferred_sleap'; % good frames
labelGui.savePath = fullfile(save_dir, save_file);
labelGui.saveAll()

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
labelGui.loadFrom3D(pts3d(bad_frame_idx, :, :));
colormap(labelGui.h{1}.Parent, 'gray'),

%% Save as a training file
save_dir = 'Z:\Sherry\poseTrackingXL\training_files\Label3D\';
save_file = 'AMB155_031025_inferred_mistake'; 
labelGui.savePath = fullfile(save_dir, save_file);
labelGui.saveAll()

%% To integrate adjusted points with the unmodified points and save again
close all
load(fullfile(save_dir, save_file));
labelGui = Label3D(camParams, videos, skeleton, 'defScale', 35);
pts3d = permute(results.posture_preds, [1, 3, 2]);
bad_pts = pts3d(bad_frame_idx, :, :);
bad_pts = reshape(bad_pts, 72, []);
bad_pts(~isnan(data_3D)) = data_3D(~isnan(data_3D));
labelGui.loadFrom3D(bad_pts);
colormap(labelGui.h{1}.Parent, 'gray'),

%% Final save
save_dir = 'Z:\Sherry\poseTrackingXL\training_files\Label3D\';
save_file = 'AMB155_031025_inferred_corrected_sleap'; % corrected frames
labelGui.savePath = fullfile(save_dir, save_file);
labelGui.saveAll()

