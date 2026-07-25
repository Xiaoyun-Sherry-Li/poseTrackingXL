function feederFrames = getFeederInteractionFrames(countData, nFrames, secBefore, secAfter, fps)
% getFeederInteractionFrames  Frame indices around and during feeder visits.
%
%   feederFrames = getFeederInteractionFrames(countData, nFrames, ...
%                       secBefore, secAfter, fps)
%
%   Builds the set of frames covering every feeder interaction, padded by a
%   user-specified number of seconds before and after each one. Intended as
%   the candidate-frame pool for identifyBadFrames, so bad-frame QC focuses on
%   feeder-interaction periods (where the beak dips into the feeder and
%   tracking is most likely to fail).
%
%   Inputs
%     countData  – annotatedSeeds.countData, with fields .newFeeder / .endFeeder
%                  (1-based onset/offset frame of each feeder interaction)
%     nFrames    – total number of frames (windows are clamped to [1, nFrames])
%     secBefore  – seconds before each interaction onset to include
%     secAfter   – seconds after  each interaction offset to include
%     fps        – video frame rate (Hz), to convert seconds -> frames
%
%   Output
%     feederFrames – sorted, unique column vector of 1-based frame indices

    padBefore = round(secBefore * fps);
    padAfter  = round(secAfter  * fps);

    mask = false(nFrames, 1);
    for k = 1:numel(countData.newFeeder)
        f1 = max(1,       countData.newFeeder(k) - padBefore);
        f2 = min(nFrames, countData.endFeeder(k) + padAfter);
        if f2 >= f1
            mask(f1:f2) = true;
        end
    end

    feederFrames = find(mask);
end
