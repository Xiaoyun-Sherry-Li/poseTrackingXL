%% sanityCheckInferredFrames.m
% Visualises predicted keypoints overlaid on video frames.
% Supports sanity-check videos, bad-frame identification, and Label3D
% training file generation.
%
% Workflow:
%   1. Load results and session metadata
%   2. (Optional) Plot model confidence / reprojection diagnostics
%   3. Read video frames for a selected index range
%   4. Overlay posture predictions in View3D
%   5. (Optional) Export a video
%   6. Identify bad frames, correct in Label3D, and save training file

clear; close all; clc;

% ── Paths ────────────────────────────────────────────────────────────────
CODE_PATH   = 'C:\Users\User\Documents\GitHub';
VID_PATH    = 'Z:\Sherry\acquisition\LIM124_082125_E3';
CAL_FILE    = 'Z:\Sherry\camera_calibration\092124_camOptArrayDA_XL.mat';
SKEL_FILE   = 'C:\Users\User\Documents\GitHub\bird_pose_tracking\postureNet\posture_skeleton_IL.csv';
% SAVE_DIR    = 'Z:\Sherry\poseTrackingXL\training_files\Label3D\';
% SAVE_FILE   = 'AMB151_071025_E4_nf';

% ── Load session data ────────────────────────────────────────────────────
load(fullfile(VID_PATH, '082125_rig_posture.mat'));
load(fullfile(VID_PATH, 'Int_LIM124_082125_v1.mat'));    % loads annotatedSeeds if needed

addpath(genpath(CODE_PATH));
cd(CODE_PATH);

CAM_NAMES = {'blue_cam', 'green_cam', 'red_cam', 'yellow_cam'};
N_CAMS    = numel(CAM_NAMES);
frames    = 1:size(results.com_conf, 1);

[allParams, skeleton] = loadSessionConfig(CAL_FILE, N_CAMS, SKEL_FILE);

% =========================================================================
%% 1.  Diagnostic plots (comment out if not needed)
% =========================================================================
% postureNet keypoint index map (see plotDiagnostics docstring for full list):
%   1=topBeak  2=bottomBeak  3=head      4=neck    5=leftEye   6=rightEye
%   7=leftWing 8=rightWing   9=leftFoot  10=rightFoot
%   11=leftAnkle  12=rightAnkle  13=chest  14=back  15=tail
POSTURE_KP_IDX = [1];           % <-- change to whichever keypoints you want
plotDiagnostics(results, frames, POSTURE_KP_IDX);

% =========================================================================
%% 2.  Sanity-check video: pick frame range and visualise
% =========================================================================
predIdx = 43979:43989;

predVid = readVideoFrames(VID_PATH, CAM_NAMES, predIdx);

% View in 3-D GUI
close all;
viewGui        = View3D(allParams, predVid, skeleton);
viewGui.defScale = 40;
colormap(viewGui.h{1}.Parent, 'gray');

pts3d_posture = permute(results.posture_preds, [1, 3, 2]);
viewGui.loadFrom3D(pts3d_posture(predIdx, :, :));

% =========================================================================
%% BELOW: haven't checked yet 3.  (Optional) Export video
% =========================================================================
% exportSanityVideo(viewGui, predIdx, fullfile(VID_PATH, 'raw100000_200frames_021826'));

% =========================================================================
%% 4.  Identify bad frames, correct in Label3D, save training file
% =========================================================================
N_SAMPLED     = 30;
NO_TAIL_IDX   = [1:6, 8:15];           % keypoint indices to include in QC metric
REP_ERR_THRESH = 5;                    % pixels
CONF_THRESH    = 0.7;

[sampled_frame_idx] = identifyBadFrames( ...
    results, frames, NO_TAIL_IDX, REP_ERR_THRESH, CONF_THRESH, N_SAMPLED);

bad_videos = readVideoFrames(VID_PATH, CAM_NAMES, sampled_frame_idx);

% Label3D correction GUI
close all;
pts3d      = permute(results.posture_preds, [1, 3, 2]);
labelGui   = Label3D(allParams, bad_videos, skeleton, 'defScale', 35);
labelGui.loadFrom3D(pts3d(sampled_frame_idx, :, :));
colormap(labelGui.h{1}.Parent, 'gray');

% ── After manual correction, run section below ───────────────────────────
%% 5.  Save Label3D training file
% saveLabelGuiDraft(labelGui, SAVE_DIR, SAVE_FILE);

% =========================================================================
%% 6.  Merge manual corrections with model predictions and save final file
% =========================================================================
% mergeAndSaveTrainingFile(SAVE_DIR, SAVE_FILE, results, sampled_frame_idx, skeleton);