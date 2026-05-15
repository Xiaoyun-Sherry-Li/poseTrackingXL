function [allParams, skeleton] = loadSessionConfig(calFile, nCams, skelFile)
% loadSessionConfig  Load camera parameters and skeleton definition.
%
%   [allParams, skeleton] = loadSessionConfig(calFile, nCams, skelFile)
%
%   Inputs:
%     calFile  – path to .mat file containing optCamArrayXL
%     nCams    – number of cameras
%     skelFile – path to skeleton CSV file
%
%   Outputs:
%     allParams – nCams×1 cell array of structs with fields:
%                   K, RDistort, TDistort, r, t
%     skeleton  – struct with fields:
%                   joint_names, joints_idx, color

    allParams = loadCameraParams(calFile, nCams);
    skeleton  = loadSkeleton(skelFile);
end

% -------------------------------------------------------------------------
function allParams = loadCameraParams(calFile, nCams)
    load(calFile, 'optCamArrayXL');
    allParams = cell(nCams, 1);
    for cam_idx = 1:nCams
        f           = optCamArrayXL(cam_idx, 7);
        principalPt = optCamArrayXL(cam_idx, 10:11);
        tmp.K        = [f, 0, 0; 0, f, 0; principalPt, 1];
        tmp.RDistort = optCamArrayXL(cam_idx, 8:9);
        tmp.TDistort = [0, 0];
        tmp.r        = rotvec2mat3d(optCamArrayXL(cam_idx, 1:3));
        tmp.t        = optCamArrayXL(cam_idx, 4:6);
        allParams{cam_idx} = tmp;
    end
end

% -------------------------------------------------------------------------
function skeleton = loadSkeleton(skelFile)
    opts = detectImportOptions(skelFile);
    opts.SelectedVariableNames = 1:3;
    skeleton_info        = readmatrix(skelFile, opts);
    n_keypoints          = size(skeleton_info, 1);
    skeleton.joint_names = transpose(skeleton_info(:, 1));

    joint_idx = [];
    for node_idx = 1:n_keypoints
        parent = skeleton_info{node_idx, 2};
        if ~isempty(parent)
            parent_idx = find(strcmp(skeleton_info(:, 1), parent));
            joint_idx  = cat(1, joint_idx, [node_idx, parent_idx]);
        end
    end

    skeleton.joints_idx = repmat(joint_idx, 2, 1);
    skeleton.color      = lines(size(skeleton.joints_idx, 1));
end