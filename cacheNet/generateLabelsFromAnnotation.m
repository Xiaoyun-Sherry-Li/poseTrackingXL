%% create new training data from seedStruct annotations
close all,
clear all,
fn = "Z:\Selmaan\Birds\SLV121\SLV121_20240515_140344";
cd(fn),
load('annotatedSeeds.mat'),

% get occupancy
seedCounts = cumsum(annotatedSeeds.seedChanges) +  annotatedSeeds.initSeedCounts';
preIntSeedCounts = seedCounts - annotatedSeeds.seedChanges;
prePreds = false(size(seedCounts,1),1);
postPreds = false(size(seedCounts,1),1);
for i = 1:length(prePreds)
    n = annotatedSeeds.cacheNum(i);
    prePreds(i) = preIntSeedCounts(i, n)>1/2;
    postPreds(i) = seedCounts(i,n)>1/2;
end

% construct image pairs
preIms = annotatedSeeds.preIms;
postIms = annotatedSeeds.postIms;

% manually save pre/post Ims and pre/post Preds to a mat file
[~,sn] = fileparts(fn);
save(strcat('botAnnotations_',sn,'.mat'),'preIms','prePreds','postIms','postPreds'),
