%% import files
Label3DPath = 'C:\Users\xl313\OneDrive\Documents\GitHub\Label3D';
addpath(genpath(Label3DPath))

%% Construct variables for Label3D GUI
% camParams: Cell array of structures denoting camera parameters for each camera.
        % Structure has five fields:
        %     K - Intrinsic Matrix
        %     RDistort - Radial distortion
        %     TDistort - Tangential distortion
        %     r - Rotation matrix
        %     t - Translation vector
alignPath = 'C:\Users\ilow1\Documents\code\il_rig_control\camera_alignment\';
cam_array_file = '240802_opt_cam_array.mat';
load(fullfile(alignPath, cam_array_file))
% read in and reformat camera array
allParams = cell(nCams,1);
for cam_idx = 1:nCams
    f = optCamArray(cam_idx, 7);
    tmp = struct;
    prinpoint = optCamArray(cam_idx,10:11);
    tmp.K = cat(1,[f, 0, 0], [0, f, 0], [prinpoint, 1]);
    tmp.RDistort = optCamArray(cam_idx,8:9);
    tmp.TDistort = [0, 0];
    tmp.r = rotationVectorToMatrix(optCamArray(cam_idx,1:3));
    tmp.t = optCamArray(cam_idx,4:6);
    allParams{cam_idx} = tmp;
end

%videos: Cell array of h x w x c x nFrames videos.


%skeleton: Structure with three fields:
    % skeleton.color: nSegments x 3 matrix of RGB values
    % skeleton.joints_idx: nSegments x 2 matrix of integers
    %     denoting directed edges between markers.
    % skeleton.joint_names: cell array of names of each joint


%% generate optimise camera array (allParams)


skeleton.joint_names = {'center', 'red_feeder', 'yellow_feeder', 'green_feeder', 'blue_feeder',...
    'water_ry', 'water_yg', 'water_gb', 'water_br'};
labelGui = Label3D(allParams, videos, skeleton, 'defScale', .06);
