function predVid = readVideoFrames(vidPath, camNames, frameIdx)
% readVideoFrames  Read specific frames from each camera's .avi file.
%
%   predVid = readVideoFrames(vidPath, camNames, frameIdx)
%
%   frameIdx  – 1-based frame indices to extract (decode order, i.e. the
%               same numbering OpenCV/SLEAP produce when reading the file
%               sequentially, which is what the predictions use)
%   predVid   – nCams×1 cell array of [H×W×3×nFrames] uint8 arrays
%
%   Seeking: VideoReader seeks by PRESENTATION time, and in these files the
%   presentation timestamps are not equal to the decode index -- each AVI
%   drops a burst of 6 frames early on, leaving a constant offset of about
%   8 frames for the rest of the session (see buildPtsIndex for details).
%   Seeking with CurrentTime = frameIdx/fr therefore returned frames ~8
%   early. We look the true pts up in a per-camera table instead.
%
%   The +0.5 lands mid-frame: seeking to the frame START rounds below the
%   boundary for most indices (the time is rarely exactly representable in
%   floating point) and returns the PREVIOUS frame.
%
%   Index-based read(reader, k) is also wrong for these AVIs (returns frames
%   from a different part of the file entirely).

    nCams   = numel(camNames);
    predVid = cell(nCams, 1);

    for cam_idx = 1:nCams
        fprintf('Reading camera: %s\n', camNames{cam_idx});
        fn     = fullfile(vidPath, [camNames{cam_idx}, '.avi']);
        ptsTbl = buildPtsIndex(fn);
        reader = VideoReader(fn);
        fr     = reader.FrameRate;

        if max(frameIdx) > numel(ptsTbl)
            error('readVideoFrames:indexOutOfRange', ...
                  '%s has %d decoded frames but frame %d was requested.', ...
                  camNames{cam_idx}, numel(ptsTbl), max(frameIdx));
        end

        nFrames = numel(frameIdx);
        vid     = zeros(reader.Height, reader.Width, 3, nFrames, 'uint8');

        for f = 1:nFrames
            reader.CurrentTime = (ptsTbl(frameIdx(f)) + 0.5) / fr;
            vid(:, :, :, f)    = readFrame(reader);
        end
        predVid{cam_idx} = vid;
    end
end
