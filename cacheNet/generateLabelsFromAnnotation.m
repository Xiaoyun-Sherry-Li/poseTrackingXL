%% create new training data from seedStruct annotations
close all,
clear all,
fn = "Z:\Sherry\acquisition\AMB155_031025";
cd(fn),
load('annotatedSeeds.mat'),

%%
% get occupancy
seedCounts = cumsum(annotatedSeeds.seedChanges) +  annotatedSeeds.initSeedCounts';  % cumulate seed change across interactions 
preIntSeedCounts = seedCounts - annotatedSeeds.seedChanges; 
prePreds = false(size(seedCounts,1),1);
postPreds = false(size(seedCounts,1),1);
for i = 1:length(prePreds)
    n = annotatedSeeds.cacheNum(i); % cache site #
    prePreds(i) = preIntSeedCounts(i, n)>1/2; % select data from the relevant cache site, note seed situation
    postPreds(i) = seedCounts(i,n)>1/2; %
end

% construct image pairs
preIms = annotatedSeeds.preIms;
postIms = annotatedSeeds.postIms;

% manually save pre/post Ims and pre/post Preds to a mat file
[~,sn] = fileparts(fn);
save(strcat('botAnnotations_',sn,'.mat'),'preIms','prePreds','postIms','postPreds'),


%%
% TODO: save pre, post Ims as color
figure;
for i = 1:100
    subplot(10, 10, i);
    imshow(preIms(:,:,i));
end
%%
figure;
for i = 1:100
    subplot(10, 10, i);
    imshow(postIms(:,:,i));
end

