function viewGui = viewInferredFrames(vidPath, results, predIdx, varargin)
%VIEWINFERREDFRAMES  Overlay postureNet keypoints on video frames in View3D.
%   Loads the requested frames from the four camera videos and displays the
%   predicted posture keypoints on top of them using SC's View3D GUI. No
%   video is written - this is purely for interactive sanity checking.
%
%   viewGui = viewInferredFrames(vidPath, results, predIdx)
%   viewGui = viewInferredFrames(vidPath, results, predIdx, Name, Value, ...)
%
%   Required inputs:
%     vidPath  - behavioral session folder containing the *_cam.avi files
%     results  - struct loaded from the raw_*.mat file (needs .posture_preds)
%     predIdx  - vector of frame indices to load (1-based), e.g.
%                  (4*60+39)*50 : (4*60+41)*50
%
%   Name/Value options:
%     'CamParamFile' - .mat with optCamArrayXL
%                      (default 'Z:\Sherry\camera_calibration\092124_camOptArrayDA_XL.mat')
%     'SkeletonFile' - posture skeleton csv
%                      (default 'C:\Users\User\Documents\GitHub\bird_pose_tracking\postureNet\posture_skeleton_IL.csv')
%     'CamNames'     - cell array of camera basenames
%                      (default {'blue_cam','green_cam','red_cam','yellow_cam'})
%     'DefScale'     - View3D default scale (default 45)
%     'PredField'    - field of results to display: 'posture' or 'com'
%                      (default 'posture')
%
%   Example:
%     vidPath = 'Z:\Sherry\acquisition\ROS108\ROS108_062526';
%     load(fullfile(vidPath,'raw_062626.mat'));   % loads 'results'
%     predIdx = (4*60+39)*50 : (4*60+41)*50;
%     vg = viewInferredFrames(vidPath, results, predIdx);

% ---- parse options -----------------------------------------------------
p = inputParser;
p.addParameter('CamParamFile', 'Z:\Sherry\camera_calibration\092124_camOptArrayDA_XL.mat');
p.addParameter('SkeletonFile', 'C:\Users\User\Documents\GitHub\bird_pose_tracking\postureNet\posture_skeleton_IL.csv');
p.addParameter('CamNames', {'blue_cam','green_cam','red_cam','yellow_cam'});
p.addParameter('DefScale', 45);
p.addParameter('PredField', 'posture');
p.parse(varargin{:});
opt = p.Results;

camNames = opt.CamNames;
nCams = numel(camNames);

% ---- read in and reformat camera parameter array -----------------------
S = load(opt.CamParamFile);
optCamArrayXL = S.optCamArrayXL;

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

% ---- define the skeleton ----------------------------------------------
opts = detectImportOptions(opt.SkeletonFile);
opts.SelectedVariableNames = 1:3;
skeleton_info = readmatrix(opt.SkeletonFile, opts);

skeleton = struct;
skeleton.joint_names = transpose(skeleton_info(:,1));
n_keypoints = length(skeleton.joint_names);
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
skeleton.color = lines(length(skeleton.joints_idx));

% ---- read in the requested frames -------------------------------------
predVid = cell(nCams,1);
for cam_idx = 1:nCams
    disp(camNames(cam_idx))
    fn = fullfile(vidPath, [camNames{cam_idx}, '.avi']);
    reader = VideoReader(fn);
    frame_rate = reader.FrameRate;
    vid = zeros(reader.Height, reader.Width, 3, length(predIdx), 'uint8');
    for f = 1:length(predIdx)
        reader.CurrentTime = predIdx(f) / frame_rate;
        frameRGB = readFrame(reader);
        vid(:,:,:,f) = frameRGB;
    end
    predVid{cam_idx} = vid;
end

% ---- launch View3D and overlay the keypoints --------------------------
viewGui = View3D(allParams, predVid, skeleton);
viewGui.defScale = opt.DefScale;
colormap(viewGui.h{1}.Parent, 'gray');

switch lower(opt.PredField)
    case 'com'
        pts3d = permute(results.com_preds, [1, 3, 2]);
    otherwise
        pts3d = permute(results.posture_preds, [1, 3, 2]);
end
viewGui.loadFrom3D(pts3d(predIdx, :, :));

end
