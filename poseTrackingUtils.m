%% poseTrackingUtils.m
% Helper functions for sanityCheckInferredFrames.m
%
% Functions:
%   loadCameraParams       – parse camera calibration array into structs
%   loadSkeleton           – read skeleton CSV and build joint graph
%   readVideoFrames        – read specific frame indices from .avi files
%   getFrameIdxFromAnnotations – extract frame range from annotatedSeeds
%   plotDiagnostics        – confidence and reprojection histograms
%   identifyBadFrames      – flag and sample high-error / low-confidence frames
%   exportSanityVideo      – write View3D output to an MPEG-4 file
%   saveLabelGuiDraft      – save the Label3D draft before manual edits
%   mergeAndSaveTrainingFile – merge corrected points with model preds and save

% =========================================================================
function allParams = loadCameraParams(calFile, nCams)
% loadCameraParams  Load and reformat camera calibration array.
%
%   allParams = loadCameraParams(calFile, nCams)
%
%   Returns a nCams×1 cell array of structs with fields:
%     K, RDistort, TDistort, r, t

    load(calFile, 'optCamArrayXL');
    allParams = cell(nCams, 1);
    for cam_idx = 1:nCams
        f          = optCamArrayXL(cam_idx, 7);
        principalPt = optCamArrayXL(cam_idx, 10:11);
        tmp.K       = [f, 0, 0; 0, f, 0; principalPt, 1];
        tmp.RDistort = optCamArrayXL(cam_idx, 8:9);
        tmp.TDistort = [0, 0];
        tmp.r        = rotvec2mat3d(optCamArrayXL(cam_idx, 1:3));
        tmp.t        = optCamArrayXL(cam_idx, 4:6);
        allParams{cam_idx} = tmp;
    end
end

% =========================================================================
function skeleton = loadSkeleton(skelFile)
% loadSkeleton  Read skeleton CSV and construct joint graph for Label3D.
%
%   skeleton = loadSkeleton(skelFile)
%
%   Returns a struct with fields:
%     joint_names, joints_idx, color

    opts = detectImportOptions(skelFile);
    opts.SelectedVariableNames = 1:3;
    skeleton_info  = readmatrix(skelFile, opts);
    n_keypoints    = size(skeleton_info, 1);
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

% =========================================================================
function predVid = readVideoFrames(vidPath, camNames, frameIdx)
% readVideoFrames  Read specific frames from each camera's .avi file.
%
%   predVid = readVideoFrames(vidPath, camNames, frameIdx)
%
%   frameIdx  – 1-based frame indices to extract
%   predVid   – nCams×1 cell array of [H×W×3×nFrames] uint8 arrays
%
%   Note on CurrentTime:
%     VideoReader.CurrentTime is 0-based (frame 1 → t = 0).
%     readFrame returns the frame AT OR AFTER CurrentTime, so:
%       CurrentTime = (frameIdx - 1) / frameRate   → exact 1-based frame
%     The -1 offset is intentional and required for correct alignment.

    nCams   = numel(camNames);
    predVid = cell(nCams, 1);

    for cam_idx = 1:nCams
        fprintf('Reading camera: %s\n', camNames{cam_idx});
        fn     = fullfile(vidPath, [camNames{cam_idx}, '.avi']);
        reader = VideoReader(fn);
        fr     = reader.FrameRate;
        nFrames = numel(frameIdx);
        vid    = zeros(reader.Height, reader.Width, 3, nFrames, 'uint8');

        for f = 1:nFrames
            reader.CurrentTime = (frameIdx(f) - 1) / fr;
            vid(:, :, :, f)    = readFrame(reader);
        end
        predVid{cam_idx} = vid;
    end
end

% =========================================================================
function predIdx = getFrameIdxFromAnnotations(annotatedSeeds, eventIndex)
% getFrameIdxFromAnnotations  Extract frame range for a cache/retrieval event.
%
%   predIdx = getFrameIdxFromAnnotations(annotatedSeeds, eventIndex)
%
%   eventIndex – which cache interaction to extract (1-based)

    [cacheInteractions, ~] = find(annotatedSeeds.seedChanges == 1);
    onsetFrames  = annotatedSeeds.countData.newSite(cacheInteractions);
    offsetFrames = annotatedSeeds.countData.endSite(cacheInteractions);
    predIdx = onsetFrames(eventIndex):offsetFrames(eventIndex);
end

% =========================================================================
function plotDiagnostics(results, frames)
% plotDiagnostics  Plot comNet and postureNet confidence / reprojection error.
%
%   plotDiagnostics(results, frames)

    lw_hist = 1.5;
    nbins   = 100;
    colors  = [0 0.5 1; 1 0 0];
    labels  = {'head', 'body'};

    % comNet
    figure('Name', 'comNet Diagnostics');
    pct98 = prctile(results.com_reproj, 98);

    subplot(1,3,1); hold on;
    for k = 1:2
        histogram(results.com_conf(frames, k), nbins, ...
            'DisplayStyle', 'stairs', 'EdgeColor', colors(k,:), ...
            'LineWidth', lw_hist, 'DisplayName', labels{k});
    end
    title('comNet – Confidence'); xlabel('Confidence');

    subplot(1,3,2); hold on;
    for k = 1:2
        vals = results.com_reproj(frames, k);
        histogram(vals(vals < pct98(k)), nbins, ...
            'DisplayStyle', 'stairs', 'EdgeColor', colors(k,:), ...
            'LineWidth', lw_hist, 'DisplayName', labels{k});
    end
    title('comNet – Reproj Error (98%)'); xlabel('Error (pixels)');

    subplot(1,3,3); hold on;
    for k = 1:2
        histogram(results.com_reproj(frames, k), nbins, ...
            'DisplayStyle', 'stairs', 'EdgeColor', colors(k,:), ...
            'LineWidth', lw_hist, 'DisplayName', labels{k});
    end
    legend('Location', 'northeastoutside');
    title('comNet – Reproj Error (100%)'); xlabel('Error (pixels)');

    % postureNet (example: topBeak only — extend idx/labels/colors as needed)
    kp_idx    = [1];
    kp_labels = {'topBeak'};
    kp_colors = [0 0.5 1];

    figure('Name', 'postureNet Diagnostics');
    subplot(1,3,1); hold on;
    for i = 1:numel(kp_idx)
        histogram(results.posture_conf(frames, kp_idx(i)), nbins, ...
            'DisplayStyle', 'stairs', 'EdgeColor', kp_colors(i,:), ...
            'LineWidth', lw_hist, 'DisplayName', kp_labels{i});
    end
    title('postureNet – Confidence'); xlabel('Confidence');

    subplot(1,3,2); hold on;
    for i = 1:numel(kp_idx)
        histogram(results.posture_reproj(frames, kp_idx(i)), ...
            'BinWidth', 1, 'DisplayStyle', 'stairs', ...
            'EdgeColor', kp_colors(i,:), 'LineWidth', lw_hist, ...
            'DisplayName', kp_labels{i});
    end
    xlim([0 10]); title('postureNet – Reproj Error (95%)'); xlabel('Error (pixels)');

    subplot(1,3,3); hold on;
    for i = 1:numel(kp_idx)
        histogram(results.posture_reproj(frames, kp_idx(i)), nbins, ...
            'DisplayStyle', 'stairs', 'EdgeColor', kp_colors(i,:), ...
            'LineWidth', lw_hist, 'DisplayName', kp_labels{i});
    end
    xlim([0 40]); legend('Location', 'northeastoutside');
    title('postureNet – Reproj Error (100%)'); xlabel('Error (pixels)');
end

% =========================================================================
function sampled_frame_idx = identifyBadFrames( ...
        results, frames, noTailIdx, repErrThresh, confThresh, nSampled)
% identifyBadFrames  Flag frames with high reprojection error or low confidence.
%
%   sampled_frame_idx = identifyBadFrames(results, frames, noTailIdx, ...
%                           repErrThresh, confThresh, nSampled)
%
%   noTailIdx    – keypoint column indices to include (exclude tail etc.)
%   repErrThresh – median reprojection error threshold (pixels)
%   confThresh   – median confidence threshold (0–1)
%   nSampled     – number of bad frames to sample for Label3D

    avg_rep_err = median(results.posture_reproj(frames, noTailIdx), 2);
    avg_conf    = median(results.posture_conf(frames, noTailIdx), 2);

    figure('Name', 'Bad Frame QC Metrics');
    subplot(1,2,1); histogram(avg_rep_err); xlabel('Median Reproj Error');
    subplot(1,2,2); histogram(avg_conf);    xlabel('Median Confidence');

    bad_mask      = (avg_rep_err > repErrThresh) & (avg_conf < confThresh);
    bad_frame_idx = frames(bad_mask);

    fprintf('Bad frames: %d / %d (%.1f%%)\n', ...
        sum(bad_mask), numel(frames), 100*mean(bad_mask));

    sampled_frame_idx = bad_frame_idx( ...
        round(linspace(1, numel(bad_frame_idx), nSampled)));
end

% =========================================================================
function exportSanityVideo(viewGui, predIdx, outPath)
% exportSanityVideo  Write View3D frames to an MPEG-4 file.
%
%   exportSanityVideo(viewGui, predIdx, outPath)
%
%   outPath – full path without extension

    v           = VideoWriter(outPath, 'MPEG-4');
    v.Quality   = 95;
    v.FrameRate = 10;
    v.open();
    for i = 1:numel(predIdx)
        viewGui.setFrame(i);
        viewGui.triangulateView();
        viewGui.resetAspectRatio();
        F = getframe(viewGui.Parent);
        v.writeVideo(F.cdata);
    end
    v.close();
    fprintf('Video saved to: %s.mp4\n', outPath);
end

% =========================================================================
function saveLabelGuiDraft(labelGui, saveDir, saveFile)
% saveLabelGuiDraft  Save the Label3D GUI state before manual correction.
%
%   saveLabelGuiDraft(labelGui, saveDir, saveFile)

    draft = [saveFile '_draft'];
    labelGui.savePath = fullfile(saveDir, draft);
    labelGui.saveAll();
    fprintf('Draft saved to: %s\n', labelGui.savePath);
end

% =========================================================================
function mergeAndSaveTrainingFile(saveDir, saveFile, results, sampledFrameIdx, skeleton)
% mergeAndSaveTrainingFile  Integrate manual corrections with model predictions.
%
%   mergeAndSaveTrainingFile(saveDir, saveFile, results, sampledFrameIdx, skeleton)
%
%   Loads the saved draft, overlays manually-corrected points (non-NaN)
%   onto the model predictions, then saves the final training file.

    draft = [saveFile '_draft'];
    load(fullfile(saveDir, [draft '_videos.mat']), 'camParams', 'videos', 'data_3D');

    pts3d    = permute(results.posture_preds, [1, 3, 2]);
    bad_pts  = pts3d(sampledFrameIdx, :, :);
    bad_pts  = reshape(bad_pts, size(bad_pts,1), []);       % nFrames × (3×nKP)
    bad_pts(~isnan(data_3D)) = data_3D(~isnan(data_3D));   % apply manual edits

    % Review merged result
    close all;
    labelGui = Label3D(camParams, videos, skeleton, 'defScale', 35);
    labelGui.loadFrom3D(bad_pts);
    colormap(labelGui.h{1}.Parent, 'gray');

    data_3D  = bad_pts;
    nFrames  = size(data_3D, 1);
    outName  = sprintf('%snFrame%d_v73_videos', saveFile, nFrames);
    save(fullfile(saveDir, outName), 'camParams', 'videos', 'data_3D', 'skeleton', '-v7.3');
    fprintf('Training file saved: %s\n', fullfile(saveDir, outName));
end
