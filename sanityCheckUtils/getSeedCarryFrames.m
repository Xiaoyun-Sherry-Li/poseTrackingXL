function carryFrames = getSeedCarryFrames(seedStruct, nFrames, secBefore, secAfter, fps)
% getSeedCarryFrames  Frame indices where the bird is carrying a seed.
%
%   carryFrames = getSeedCarryFrames(seedStruct, nFrames, ...
%                       secBefore, secAfter, fps)
%
%   Detects carry bouts from the smSeed carry-probability signal (same
%   thresholding as trackSeedTrajectories: binarise at gainThresh, then bridge
%   losses shorter than minLoseDur), and pads each bout by a user-specified
%   number of seconds before and after. Intended as a candidate-frame pool for
%   identifyBadFrames, so bad-frame QC also covers seed-transport periods
%   (where the beak keypoint is often occluded / unreliable).
%
%   Inputs
%     seedStruct – annotatedSeeds, with fields .smSeed, .gainThresh, .minLoseDur
%     nFrames    – total number of frames (windows are clamped to [1, nFrames])
%     secBefore  – seconds before each carry bout onset to include
%     secAfter   – seconds after  each carry bout offset to include
%     fps        – video frame rate (Hz), to convert seconds -> frames
%
%   Output
%     carryFrames – sorted, unique column vector of 1-based frame indices

    smSeed     = seedStruct.smSeed(:);
    gainThresh = seedStruct.gainThresh;
    minLoseDur = seedStruct.minLoseDur;

    % Forward-fill NaN gaps so brief occlusions don't split a carry bout
    smFilled = smSeed;
    lastVal  = 0;
    for f = 1:numel(smFilled)
        if isnan(smFilled(f)); smFilled(f) = lastVal;
        else;                  lastVal      = smFilled(f);
        end
    end

    % Binary carry, then bridge short losses (< minLoseDur frames)
    carrying = smFilled >= gainThresh;
    lossOn   = find(diff([1; double(carrying)]) == -1);
    lossOff  = find(diff([double(carrying); 1]) ==  1);
    for k = 1:numel(lossOn)
        if (lossOff(k) - lossOn(k)) < minLoseDur
            carrying(lossOn(k):lossOff(k)) = true;
        end
    end

    % Carry bout boundaries
    boutOn  = find(diff([0; double(carrying)]) ==  1);
    boutOff = find(diff([double(carrying); 0]) == -1);

    % Pad each bout by the requested seconds and union into a frame mask
    padBefore = round(secBefore * fps);
    padAfter  = round(secAfter  * fps);
    mask = false(nFrames, 1);
    for k = 1:numel(boutOn)
        f1 = max(1,       boutOn(k)  - padBefore);
        f2 = min(nFrames, boutOff(k) + padAfter);
        if f2 >= f1
            mask(f1:f2) = true;
        end
    end

    carryFrames = find(mask);
end
