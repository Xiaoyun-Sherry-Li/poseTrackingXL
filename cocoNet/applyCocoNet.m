%% visualise the distribution of facePred values in different interactions 

% load the facePred predictions for the entire video first 
addpath C:\Users\xl313\OneDrive\Documents\GitHub\trackingIntegration
sessionPath = 'Z:\Sherry\acquisition\LVN4_040725';
load(fullfile(sessionPath, 'annotatedSeeds'));

%% loading cocoNet outputs
cocoNetCaches = load("Z:\Sherry\acquisition\LVN4_040725\cocoPredCaches.mat");
cocoNetRetrieves = load("Z:\Sherry\acquisition\LVN4_040725\cocoPredRetrieves.mat");

%% take the medium across 10 frames for every caches
finalCocoNetCaches = median(cocoNetCaches.cocoPredCaches);
finalCocoNetRetrieves = median(cocoNetRetrieves.cocoPredRetrieves);
histogram(finalCocoNetCaches,NumBins = 40)
histogram(finalCocoNetRetrieves,NumBins = 40)

%% find the indices where the numbers are >0.2 and < 0.8
mistakeCacheIdx = find(finalCocoNetCaches > 0.2 & finalCocoNetCaches < 0.8);
mistakeRetrieveIdx = find(finalCocoNetRetrieves > 0.2 & finalCocoNetRetrieves < 0.8);

%% visualise these mistake frames and manually label them
% load all the frames of cache/retrieval interactions 
load("Z:\Sherry\acquisition\LVN4_040725\foodIDframes10.mat")
%% take the later frame (8th) of each mistake interaction and visualise
mistakeCacheImgs = beforeCacheImgs(mistakeCacheIdx,8);
% load all 4 camera video fils 
nCams = 4;
FPS = 50;
vidPath = 'Z:\Sherry\acquisition\LVN4_040725'; % behavioral session
camNames = {'blue_cam', 'green_cam', 'red_cam', 'yellow_cam'};

% read in frames
predVid = cell(nCams,1);
for cam_idx = 1:nCams
    disp(camNames(cam_idx))
    fn = fullfile(vidPath, [camNames{cam_idx}, '.avi']);
    reader = VideoReader(fn);
    frame_rate = reader.FrameRate;
    vid = zeros(reader.Height, reader.Width, 3, length(mistakeCacheImgs), 'uint8');
    for f = 1:length(mistakeCacheImgs)
        reader.CurrentTime = mistakeCacheImgs(f)/frame_rate;
        frameRGB = readFrame(reader);
        vid(:,:,:,f) = frameRGB;
    end
    predVid{cam_idx} = vid;
end

%% take the earlier frame (2nd) of each mistake interaction and visualise
mistakeRetrieveImgs = AfterRetrieveImgs(mistakeRetrieveIdx,2);
% load all 4 camera video fils 
nCams = 4;
FPS = 50;
vidPath = 'Z:\Sherry\acquisition\LVN4_040725'; % behavioral session
camNames = {'blue_cam', 'green_cam', 'red_cam', 'yellow_cam'};

% read in frames
predVid = cell(nCams,1);
for cam_idx = 1:nCams
    disp(camNames(cam_idx))
    fn = fullfile(vidPath, [camNames{cam_idx}, '.avi']);
    reader = VideoReader(fn);
    frame_rate = reader.FrameRate;
    vid = zeros(reader.Height, reader.Width, 3, length(mistakeRetrieveImgs), 'uint8');
    for f = 1:length(mistakeRetrieveImgs)
        reader.CurrentTime = mistakeRetrieveImgs(f)/frame_rate;
        frameRGB = readFrame(reader);
        vid(:,:,:,f) = frameRGB;
    end
    predVid{cam_idx} = vid;
end

%% visualise the 4 camera views of every interaction 
for i = 1:16
    figure;
    hold on 
    for f = 1:4
        subplot(2, 2, f);
        imshow(predVid{f}(:,:,:,i));
    end
    title(['Mistake Int ', num2str(i)]);
end

%% mistake manual annotation 
mistakeManualFixRetrieves
%%
finalCocoNetRetrieves(mistakeRetrieveIdx) = mistakeManualFixRetrieves;
histogram(finalCocoNetRetrieves,Numbins = 20);
% binarise finalCocoNetCaches for final output 
finalCocoNetRetrieves = round(finalCocoNetRetrieves); 

%%  curated coconut/peanut labels for cache/ retrieval interactions: finalCocoNetRetrieves, finalCocoNetCaches
% make coconut be 2, peanut be 3
finalCocoNetCaches = finalCocoNetCaches + 2;
finalCocoNetRetrieves = finalCocoNetRetrieves + 2;
save("Z:\Sherry\acquisition\LVN4_040725\FinalCocoNetAnnotations.mat",'finalCocoNetRetrieves','finalCocoNetCaches')