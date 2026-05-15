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
