%% Check and relabel inferred frames for a semi-manual training set
% this function visualises the predicted keypoints overlaying on the video
% frames 
clear all
close all;
clc
codePath = 'C:\Users\User\Documents\GitHub\Label3D';
addpath(genpath(codePath))
cd 'C:\Users\User\Documents\GitHub\Label3D'

% [change this] Set file paths and load results from SLEAP output 
vidPath = 'Z:\Sherry\acquisition\LIM130_031626_E1'; % behavioral session
load(fullfile(vidPath, 'raw_pred_031626.mat'));

%% visualise comNet performances
frames = 1: size(results.com_conf,1);% 58*60*50: 59*60*50; %size(results.com_conf,1);
figure;
lw_hist = 1.5;                        
nbins = 100;                      
colors = [0 0.5 1;    % head
          1 0   0];   % body
labels = {'head','body'};

ax(1) = subplot(1,3,1); hold on;
for k = 1:2
    histogram(results.com_conf(frames,k), nbins, ...
        'DisplayStyle','stairs', ...
        'EdgeColor',colors(k,:), ...
        'LineWidth',lw_hist, ...
        'DisplayName',labels{k});
end
title('Model Confidence');
xlabel('Confidence');

ax(2) = subplot(1,3,2); hold on;
pct98 = prctile(results.com_reproj,98);
for k = 1:2
    vals = results.com_reproj(frames,k);
    vals = vals(vals < pct98(k));

    histogram(vals, nbins, ...
        'DisplayStyle','stairs', ...
        'EdgeColor',colors(k,:), ...
        'LineWidth',lw_hist, ...
        'DisplayName',labels{k});
end
title('Reprojection Error (98%)');
xlabel('Error (pixels)');

ax(3) = subplot(1,3,3); hold on;
for k = 1:2
    histogram(results.com_reproj(frames,k), nbins, ...
        'DisplayStyle','stairs', ...
        'EdgeColor',colors(k,:), ...
        'LineWidth',lw_hist, ...
        'DisplayName',labels{k});
end
legend('Location','northeastoutside');
title('Reprojection Error (100%)');
xlabel('Error (pixels)');
hold off;

%% PostureNet evaluation
idx      = [1 ]%11 15 12 14 10 7];
labels   = {'topBeak'}%,'left foot','right foot','rightEye','right ankle','left ankle','tail tip'};
colors   = [ ...
    0   0.5 1;]   % topBeak
    % 1   0   0;   % left foot
    % 0   0   1;   % right foot
    % 1   1   0;   % rightEye
    % 1   0.5 0;   % right ankle
    % 0.5 1   0;
    % 0   0   0];  % left ankle

% 1) Model confidence
figure
ax(1) = subplot(1,3,1);
hold(ax(1),'on');

for i = 1:numel(idx)
    histogram(ax(1), results.posture_conf(frames,idx(i)), ...
        100, ...
        'EdgeColor', colors(i,:), ...
        'LineWidth', lw_hist, ...
        'DisplayStyle', 'stairs', ...
        'DisplayName', labels{i});
end

title(ax(1),'Model Confidence');
xlabel(ax(1),'Confidence');
hold(ax(1),'off');

% 2) Reprojection error (95th percentile)
ax(2) = subplot(1,3,2);
hold(ax(2),'on');

for i = 1:numel(idx)
    histogram(ax(2), results.posture_reproj(frames,idx(i)), ...
    'BinWidth', 1, ...
    'EdgeColor', colors(i,:), ...
    'LineWidth', lw_hist, ...
    'DisplayStyle', 'stairs', ...
    'DisplayName', labels{i});
end

xlim(ax(2),[0 10]); title(ax(2),'Reprojection Error (95%)'); xlabel(ax(2),'Error (pixels)');
hold(ax(2),'off');

% 3) Reprojection error (100%)
ax(3) = subplot(1,3,3);
hold(ax(3),'on');
for i = 1:numel(idx)
    histogram(ax(3), results.posture_reproj(frames,idx(i)), ...
        100, ...
        'EdgeColor', colors(i,:), ...
        'LineWidth', lw_hist, ...
        'DisplayStyle', 'stairs', ...
        'DisplayName', labels{i});
end
xlim(ax(3),[0 40]); legend(ax(3),'Location','northeastoutside'); title(ax(3),'Reprojection Error (100%)');
xlabel(ax(3),'Error (pixels)');
hold(ax(3),'off');


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

% Define the skeleton
% load the csv file
skeleton_file = 'C:\Users\User\Documents\GitHub\bird_pose_tracking\postureNet\posture_skeleton_IL.csv';
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

% com or posture skeletons 
% posture skeleton
skeleton.joints_idx = repmat(joint_idx, 2, 1);
skeleton.color = lines(length(skeleton.joints_idx)); % 15 body parts -> 15 distinct colors

% com skeleton (uncomment to produce a com keypoint video)
% skeleton.joint_names = {'Head','Body','Tail'};
% skeleton.joints_idx = [[1,1];[2,2];[3,3]];
% skeleton.color = lines(length(skeleton.joints_idx));

%% [optional] load selected frames for making a sanity check video
% predStart = 1; % in seconds % successful retrieval detection 
% predFrames = 50 % 1* reader.FrameRate; % duration, in frames 
% predIdx = predStart * reader.FrameRate : predStart * reader.FrameRate + predFrames - 1;
predIdx = 85600: 85900;%(14*60 + 30)*50:(14*60+45)*50;
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
    vid = zeros(reader.Height, reader.Width, 3, length(predIdx), 'uint8');
    for f = 1:length(predIdx) % predFrames
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
viewGui.loadFrom3D(pts3d_posture(predIdx, :, :));

%% Create a video 
% cd 'Z:\Sherry\acquisition\RBY52_2ndPart_012425'  
v = VideoWriter(fullfile(vidPath,'raw100000_200frames_021826'),'MPEG-4');
v.Quality=95;
v.FrameRate = 10;
v.open,
for i = 1:length(predIdx)
    viewGui.setFrame(i);
    viewGui.triangulateView();
    viewGui.resetAspectRatio();
    F = getframe(viewGui.Parent); % grab a frame from viewGUI.Parent - a handle to an axis
    v.writeVideo(F.cdata);
end
v.close

%% bad individual frames
noTail = [1:6, 8:15];
avg_rep_err = median(results.posture_reproj(frames,noTail), 2); % using median, so not capturing outliers e.g. tails
avg_conf = median(results.posture_conf(frames,noTail), 2);

figure; subplot(1,2,1); hist(avg_rep_err); hold on; subplot(1,2,2); hist(avg_conf);

high_rep_err = avg_rep_err > 5; % used to be 99.9
low_conf = avg_conf < 0.7; % used to be 0.1
bad_frame_idx = high_rep_err & low_conf;

%% Load the bad video frames
% data params
nSampledFrames = 30;
frame_idx = frames(bad_frame_idx);
sampled_frame_idx = frame_idx(round(linspace(1,size(frame_idx,2),nSampledFrames)));

% read in images
bad_videos = cell(nCams,1);
for cam_idx = 1:nCams
    disp(camNames(cam_idx))
    fn = fullfile(vidPath, [camNames{cam_idx}, '.avi']);
    reader = VideoReader(fn);
    frame_rate = reader.FrameRate;
    vid = zeros(reader.Height, reader.Width, 3, nSampledFrames, 'uint8');
    for f = 1:nSampledFrames
        reader.CurrentTime = sampled_frame_idx(f)/frame_rate;
        frameRGB = readFrame(reader);
        vid(:,:,:,f) = frameRGB;
    end
    bad_videos{cam_idx} = vid;
end

%% Makes a Label3D object and start the GUI (low conf)
close all
% To confirm that high conf frames look good
pts3d = permute(results.posture_preds, [1, 3, 2]);
% To identify and correct failure modes
labelGui = Label3D(allParams, bad_videos, skeleton, 'defScale', 35);

labelGui.loadFrom3D(pts3d(sampled_frame_idx, :, :));

colormap(labelGui.h{1}.Parent, 'gray'),

%% Save as a training file
save_file = 'AMB156_040726_nF30'; 
draft = [save_file '_draft'];
save_dir = 'Z:\Sherry\poseTrackingXL\training_files\Label3D\';
labelGui.savePath = fullfile(save_dir, draft);
labelGui.saveAll()

%% To integrate adjusted points with the unmodified points and save again
close all
load(fullfile(save_dir, [draft '_videos.mat']));
labelGui = Label3D(camParams, videos, skeleton, 'defScale', 35);
pts3d = permute(results.posture_preds, [1, 3, 2]);
bad_pts = pts3d(sampled_frame_idx, :, :);
bad_pts = reshape(bad_pts, size(bad_pts,1), []); % shape from nFrames x 3 x 15 to nFrames x 45
bad_pts(~isnan(data_3D)) = data_3D(~isnan(data_3D));

labelGui.loadFrom3D(bad_pts);
colormap(labelGui.h{1}.Parent, 'gray'),
data_3D = bad_pts;
% don't close before saving! 

%% Final save
% save(fullfile(save_dir, [save_file 'nFrame' num2str(size(data_3D,1)) '_videos']), "camParams", "videos", "data_3D", "skeleton", '-v7.3')
save(fullfile(save_dir, 'AMB156_040726nFrame26_v73_videos'), "camParams", "videos", "data_3D", "skeleton", '-v7.3')

%% test again (optional) 
close all
labelGui = Label3D(camParams, videos, skeleton, 'defScale', 35);
labelGui.loadFrom3D(data_3D);
colormap(labelGui.h{1}.Parent, 'gray'),