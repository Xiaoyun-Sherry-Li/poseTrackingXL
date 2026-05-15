function plotDiagnostics(results, frames, postureKeypointIdx)
% plotDiagnostics  Plot comNet and postureNet confidence / reprojection error.
%
%   plotDiagnostics(results, frames, postureKeypointIdx)
%
%   Inputs:
%     results            – results struct from posture .mat file
%     frames             – frame indices to include (e.g. 1:nFrames)
%     postureKeypointIdx – vector of postureNet keypoint indices to plot,
%                          chosen from the index map below
%
%   postureNet keypoint index map:
%     1  – topBeak
%     2  – bottomBeak
%     3  – head
%     4  – neck
%     5  – leftEye
%     6  – rightEye
%     7  – leftWing
%     8  – rightWing
%     9  – leftFoot
%     10 – rightFoot
%     11 – leftAnkle
%     12 – rightAnkle
%     13 – chest
%     14 – back
%     15 – tail
%
%   Example – plot head and both feet:
%     plotDiagnostics(results, frames, [1, 9, 10])
 
    % Full keypoint name lookup (used to auto-label plots)
    allKeypointNames = { ...
        'topBeak',    ...  % 1
        'bottomBeak', ...  % 2
        'head',       ...  % 3
        'neck',       ...  % 4
        'leftEye',    ...  % 5
        'rightEye',   ...  % 6
        'leftWing',   ...  % 7
        'rightWing',  ...  % 8
        'leftFoot',   ...  % 9
        'rightFoot',  ...  % 10
        'leftAnkle',  ...  % 11
        'rightAnkle', ...  % 12
        'chest',      ...  % 13
        'back',       ...  % 14
        'tail'        ...  % 15
    };
 
    lw_hist   = 1.5;
    nbins     = 100;
    kp_labels = allKeypointNames(postureKeypointIdx);
    kp_colors = lines(numel(postureKeypointIdx));
 
    % ── comNet (fixed: col 1 = head, col 2 = body) ───────────────────────
    com_colors = [0 0.5 1; 1 0 0];
    com_labels = {'head', 'body'};
    pct98      = prctile(results.com_reproj, 98);
 
    figure('Name', 'comNet Diagnostics');
    subplot(1,3,1); hold on;
    for k = 1:2
        histogram(results.com_conf(frames, k), nbins, ...
            'DisplayStyle', 'stairs', 'EdgeColor', com_colors(k,:), ...
            'LineWidth', lw_hist, 'DisplayName', com_labels{k});
    end
    legend('Location', 'northeastoutside');
    title('comNet – Confidence'); xlabel('Confidence');
 
    subplot(1,3,2); hold on;
    for k = 1:2
        vals = results.com_reproj(frames, k);
        histogram(vals(vals < pct98(k)), nbins, ...
            'DisplayStyle', 'stairs', 'EdgeColor', com_colors(k,:), ...
            'LineWidth', lw_hist, 'DisplayName', com_labels{k});
    end
    legend('Location', 'northeastoutside');
    title('comNet – Reproj Error (98%)'); xlabel('Error (pixels)');
 
    subplot(1,3,3); hold on;
    for k = 1:2
        histogram(results.com_reproj(frames, k), nbins, ...
            'DisplayStyle', 'stairs', 'EdgeColor', com_colors(k,:), ...
            'LineWidth', lw_hist, 'DisplayName', com_labels{k});
    end
    legend('Location', 'northeastoutside');
    title('comNet – Reproj Error (100%)'); xlabel('Error (pixels)');
 
    % ── postureNet ────────────────────────────────────────────────────────
    figure('Name', sprintf('postureNet Diagnostics – %s', ...
        strjoin(kp_labels, ', ')));
 
    subplot(1,3,1); hold on;
    for i = 1:numel(postureKeypointIdx)
        histogram(results.posture_conf(frames, postureKeypointIdx(i)), nbins, ...
            'DisplayStyle', 'stairs', 'EdgeColor', kp_colors(i,:), ...
            'LineWidth', lw_hist, 'DisplayName', kp_labels{i});
    end
    legend('Location', 'northeastoutside');
    title('postureNet – Confidence'); xlabel('Confidence');
 
    subplot(1,3,2); hold on;
    for i = 1:numel(postureKeypointIdx)
        histogram(results.posture_reproj(frames, postureKeypointIdx(i)), ...
            'BinWidth', 1, 'DisplayStyle', 'stairs', ...
            'EdgeColor', kp_colors(i,:), 'LineWidth', lw_hist, ...
            'DisplayName', kp_labels{i});
    end
    xlim([0 10]); legend('Location', 'northeastoutside');
    title('postureNet – Reproj Error (95%)'); xlabel('Error (pixels)');
 
    subplot(1,3,3); hold on;
    for i = 1:numel(postureKeypointIdx)
        histogram(results.posture_reproj(frames, postureKeypointIdx(i)), nbins, ...
            'DisplayStyle', 'stairs', 'EdgeColor', kp_colors(i,:), ...
            'LineWidth', lw_hist, 'DisplayName', kp_labels{i});
    end
    xlim([0 40]); legend('Location', 'northeastoutside');
    title('postureNet – Reproj Error (100%)'); xlabel('Error (pixels)');
end