function diagnoseFrameSeek(vidPath, camName, nProbe)
% diagnoseFrameSeek  Determine how VideoReader maps CurrentTime to frames.
%
%   diagnoseFrameSeek(vidPath, camName)
%   diagnoseFrameSeek(vidPath, camName, nProbe)
%
%   Background: these AVIs are CFR 50/1 containers, but the presentation
%   timestamps are NOT equal to the decode index. In AMB151_071025_E4,
%   blue_cam.avi has a burst of 6 dropped frames around decode index 251,
%   so pts = n + 2 before the burst and pts = n + 8 after it, constant for
%   the remaining ~533k frames.
%
%   That leaves two possibilities for readVideoFrames:
%     (a) VideoReader seeks by PRESENTATION TIME  -> CurrentTime = n/fr
%         lands on the frame whose pts is n, i.e. decode index n-8.
%         Every frame comes back 8 early. Seeking must use pts, not index.
%     (b) VideoReader seeks by DECODED FRAME COUNT -> the pts gaps are
%         absorbed and the existing mid-frame formula is already correct.
%
%   This function decides between them. It reads the first nProbe frames
%   sequentially (ground truth, no seeking), then seeks to several target
%   indices and reports which sequential frame each seek actually returned.
%   Probe targets straddle the drop burst, so (a) and (b) differ by 6.

    if nargin < 3, nProbe = 340; end

    fn     = fullfile(vidPath, [camName, '.avi']);
    reader = VideoReader(fn);
    fr     = reader.FrameRate;

    fprintf('\n=== %s ===\n', fn);
    fprintf('FrameRate = %.10g   Duration = %.4f s\n', fr, reader.Duration);
    try
        fprintf('NumFrames = %d\n', reader.NumFrames);
    catch
        fprintf('NumFrames = <unavailable>\n');
    end

    % --- ground truth: sequential decode, no seeking ----------------------
    fprintf('Reading %d frames sequentially for reference...\n', nProbe);
    sigs = zeros(nProbe, 64);
    for i = 1:nProbe
        sigs(i, :) = frameSignature(readFrame(reader));
    end

    % --- probe: seek to each target and see what comes back ---------------
    targets = [50, 150, 245, 260, 300, 330];
    targets = targets(targets <= nProbe);

    formulas = { ...
        'mid-frame  (k-1+0.5)/fr', @(k) (k - 1 + 0.5) / fr; ...
        'current    k/fr',         @(k) k / fr };

    for fi = 1:size(formulas, 1)
        fprintf('\n-- formula: %s --\n', formulas{fi, 1});
        fprintf('  %-8s %-12s %-12s %s\n', 'want k', 'CurrentTime', 'got seq k', 'offset');
        for k = targets
            r2 = VideoReader(fn);
            r2.CurrentTime = formulas{fi, 2}(k);
            sig  = frameSignature(readFrame(r2));
            d    = sum(abs(sigs - sig), 2);
            [~, got] = min(d);
            fprintf('  %-8d %-12.6f %-12d %+d\n', k, formulas{fi, 2}(k), got, got - k);
        end
    end

    fprintf(['\nIf offset is 0 everywhere -> VideoReader counts decoded frames;\n' ...
             'the mid-frame formula is correct and no pts table is needed.\n' ...
             'If offset jumps (e.g. -2 before k=251, -8 after) -> VideoReader\n' ...
             'seeks by pts and readVideoFrames must seek by pts instead.\n']);
end

% -------------------------------------------------------------------------
function s = frameSignature(f)
% 8x8 grid sample of the first channel - cheap, toolbox-free frame fingerprint
    [h, w, ~] = size(f);
    rows = round(linspace(1, h, 8));
    cols = round(linspace(1, w, 8));
    patch = double(f(rows, cols, 1));
    s = patch(:).';
end
