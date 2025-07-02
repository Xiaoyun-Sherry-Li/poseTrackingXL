%% use existing network to predict on my data 
close all, 
clear all,
clc
% load("Z:\Sherry\acquisition\AMB155_031025\botAnnotations_AMB155_031025_curated.mat");
load("Z:\Sherry\acquisition\LVN4_040725\annotatedSeeds_updatedImages.mat");
cacheNet = load('C:\Users\xl313\OneDrive\Documents\GitHub\poseTrackingXL\cacheNet\cacheNet_043025_size25.mat'); % original 25 
cacheNet = cacheNet.cacheNet;

%% 
preIms = annotatedSeeds.preIms;
postIms = annotatedSeeds.postIms;
%% 
prePreds = annotatedSeeds.prePreds;
postPreds = annotatedSeeds.postPreds;


%%
rawprePreds = cacheNet.predict(reshape(preIms,51,51,1,[])); % sherry changed this, used to be 51, 51
rawprePreds = rawprePreds(:,2); % first and second columns are n,1-n complementary to each other
rawpostPreds = cacheNet.predict(reshape(postIms,51,51,1,[])); % sherry changed this, used to be 51, 51
rawpostPreds = rawpostPreds(:,2);

%% binarise the predictions
rawprePreds(rawprePreds > 0.5) = 1;
rawprePreds(rawprePreds < 0.5) = 0;

rawpostPreds(rawpostPreds > 0.5) = 1;
rawpostPreds(rawpostPreds < 0.5) = 0;

%% 
start = 600;
n = 200;
for i = start: start + n-1
    subplot(10, 20, i-start+1); 
    imshow(postIms(:,:,i));
    % title(sprintf('Id%d:%d', i, rawpostPreds(i)));
    title(sprintf('%.4f', postPreds(i)));
end

%% 
prePreds = rawprePreds; 
postPreds = rawpostPreds;

%%
save("Z:\Sherry\acquisition\LVN4_040725\botAnnotations_LVN4_040725_curated.mat", "prePreds","postPreds","preIms","postIms");
