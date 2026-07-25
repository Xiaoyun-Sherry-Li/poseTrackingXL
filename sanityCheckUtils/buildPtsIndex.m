function pts = buildPtsIndex(vidFile, forceRebuild)
% buildPtsIndex  Map decode index -> presentation timestamp for an AVI.
%
%   pts = buildPtsIndex(vidFile)
%   pts = buildPtsIndex(vidFile, true)     % ignore cache, rescan
%
%   pts(k) is the presentation timestamp (in units of the stream time_base,
%   here 1/50 s) of the k-th decoded frame, 1-based.
%
%   Why this is needed: these AVIs are constant-rate 50/1 containers, but
%   the pts sequence has GAPS where the camera dropped frames. Each file in
%   AMB151_071025_E4 drops a burst of 6 frames around decode index 251, so
%   pts = k + 1 before the burst and pts = k + 7 after it (1-based), constant
%   for the remaining ~533k frames. Seeking with CurrentTime = k/fr therefore
%   lands on the frame whose pts is k -- about 8 frames earlier than intended.
%   The offset also differs slightly between cameras (green is one further
%   along than blue/red/yellow), so using the raw index misaligns the cameras
%   against each other as well.
%
%   The scan requires ffprobe on the system PATH. It reads the whole file
%   (~1 min per 2.8 GB over the network) and caches the result next to the
%   video as <cam>_ptsIndex.mat, so it is paid once per camera per session.

    if nargin < 2, forceRebuild = false; end

    [vidDir, vidName] = fileparts(vidFile);
    cacheFile = fullfile(vidDir, [vidName, '_ptsIndex.mat']);

    if ~forceRebuild && exist(cacheFile, 'file')
        S = load(cacheFile, 'pts');
        pts = S.pts;
        return;
    end

    fprintf('  building pts index for %s (one-time scan)...\n', vidName);
    cmd = sprintf(['ffprobe -v error -select_streams v:0 ' ...
                   '-show_entries packet=pts -of csv=p=0 "%s"'], vidFile);
    [status, out] = system(cmd);
    if status ~= 0
        error('buildPtsIndex:ffprobeFailed', ...
              ['ffprobe failed on %s (status %d).\n' ...
               'Is ffprobe on the PATH? Output:\n%s'], vidFile, status, out);
    end

    vals = sscanf(strrep(out, 'N/A', 'NaN'), '%f');
    vals = vals(~isnan(vals));
    if isempty(vals)
        error('buildPtsIndex:noPackets', 'ffprobe returned no timestamps for %s', vidFile);
    end

    % n-th decoded frame is the n-th smallest pts (B-frames arrive reordered)
    pts = sort(vals);

    nMissing = (pts(end) - pts(1) + 1) - numel(pts);
    fprintf('    %d frames, pts %d..%d, %d dropped, offset %+d at end\n', ...
            numel(pts), pts(1), pts(end), nMissing, pts(end) - numel(pts));

    try
        save(cacheFile, 'pts');
    catch ME
        warning('buildPtsIndex:cacheFailed', ...
                'Could not cache pts index to %s (%s). Will rescan next time.', ...
                cacheFile, ME.message);
    end
end
