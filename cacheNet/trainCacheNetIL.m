%% load and preprocess data
close all,
clear all,

% TODO possibly won't need parts of this...see what is needed after using 
% up to date annotation GUI

allData = cat(1,load('C:\Users\xl313\OneDrive\Documents\GitHub\bird_pose_tracking\training_files\bottomCamera\botAnnotations_TRQ177_210318.mat'),...
    load('C:\Users\xl313\OneDrive\Documents\GitHub\bird_pose_tracking\training_files\bottomCamera\botAnnotations_IND88_210401.mat'),...
    load('C:\Users\xl313\OneDrive\Documents\GitHub\bird_pose_tracking\training_files\bottomCamera\botAnnotations_LIM99_210503.mat'),...
    load('C:\Users\xl313\OneDrive\Documents\GitHub\bird_pose_tracking\training_files\bottomCamera\botAnnotations_IND106_220701.mat'),...
    load('C:\Users\xl313\OneDrive\Documents\GitHub\bird_pose_tracking\training_files\bottomCamera\botAnnotations_IND106_220705.mat'),...
    load('C:\Users\xl313\OneDrive\Documents\GitHub\bird_pose_tracking\training_files\bottomCamera\botAnnotations_TRQ66_220728.mat'),...
    load('C:\Users\xl313\OneDrive\Documents\GitHub\bird_pose_tracking\training_files\bottomCamera\botAnnotations_AMB43_221103.mat'),...
    load('C:\Users\xl313\OneDrive\Documents\GitHub\bird_pose_tracking\training_files\bottomCamera\botAnnotations_LMN169_221104.mat'),...
    load('C:\Users\xl313\OneDrive\Documents\GitHub\bird_pose_tracking\training_files\bottomCamera\botAnnotations_SLV121_20240513_104112.mat'),...
    load('C:\Users\xl313\OneDrive\Documents\GitHub\bird_pose_tracking\training_files\bottomCamera\botAnnotations_SLV121_20240515_140344.mat'));

%% format input data
for i=1:length(allData)
    if size(allData(i).preIms,3)~=2
        allData(i).preIms = permute(repmat(allData(i).preIms,1,1,1,2),[1,2,4,3]);
        allData(i).postIms = permute(repmat(allData(i).postIms,1,1,1,2),[1,2,4,3]);
    end
end

preIms = cat(4, allData.preIms);
postIms = cat(4, allData.postIms);
prePreds = cat(1, allData.prePreds)>1/2;
postPreds = cat(1, allData.postPreds)>1/2;
nPairs = size(postIms,4);
assert(nPairs == size(preIms,4)),
% take only 2nd channel (present image, ignore baseline image)
preIms = preIms(:,:,2,:);
postIms = postIms(:,:,2,:);

%% set paths
root_dir = 'C:\Users\xl313\OneDrive\Documents\GitHub\bird_pose_tracking\';
data_dir = strcat(root_dir, 'training_files\cache_annotations\');
data_file = 'botAnnotations_paddedsize25_042625.mat';
save_dir = strcat(root_dir, 'cacheNet\');
save_file = strrep(save_dir + "cacheNet_" + string(datenum(datetime)),'.','_');

% load the data
load(fullfile(data_dir, data_file))

% add an extra channel to the images (redundant?)
if size(preIms, 3)~=2
    preIms = permute(repmat(preIms,1,1,1,2),[1,2,4,3]);
    postIms = permute(repmat(postIms,1,1,1,2),[1,2,4,3]);
end
preIms = preIms(:,:,2,:);
postIms = postIms(:,:,2,:);

% reformat and get file info
prePreds = prePreds > 1/2;
postPreds = postPreds > 1/2;
nPairs = size(postIms, 4);
assert(nPairs == size(preIms, 4)),

%% format the data for the cacheNet
% split into train/validation 
trainInd = randperm(nPairs,round(.9*nPairs));
validInd = setdiff(1:nPairs,trainInd);
trainIms = cat(4, preIms(:,:,:,trainInd), postIms(:,:,:,trainInd));
trainPreds = cat(1, prePreds(trainInd), postPreds(trainInd));
validIms = cat(4, preIms(:,:,:,validInd), postIms(:,:,:,validInd));
validPreds = cat(1, prePreds(validInd), postPreds(validInd));

% remove images/predictions with nan values
nanIms = find(isnan(squeeze(sum(trainIms,1:3))));
trainIms(:,:,:,nanIms) = [];
trainPreds(nanIms) = [];
nanIms = find(isnan(squeeze(sum(validIms,1:3))));
validIms(:,:,:,nanIms) = [];
validPreds(nanIms) = [];

% define augmentation
imageAugmenter = imageDataAugmenter(...
    'RandXReflection',true,'RandYReflection',true,...
    'FillValue',1/2,...
    'RandScale',[0.9, 1.1],...
    'RandXTranslation',[-5, 5],...
    'RandYTranslation',[-5, 5]);

% create augmented image datastore
augImgs = augmentedImageDatastore(size(trainIms,1:2), trainIms,...
    categorical(trainPreds), 'DataAugmentation', imageAugmenter);

%% Construct network
% Note 1/29/25: Was getting an Input size mismatch error
% chatGPT suggested that this was due to downsampling too much (it was
% originally designed for a larger image and was basically downsampling to
% nothing). To fix this issue, I adjusted the first maxPooling2dLayer(2,'Stride',2)
% to have a stride of 1. The model still seems to train and test well...

% (Sherry)deepseek no donwsampling: 1. Set the Stride of convolutional layers to 1 (no downsampling). 2. Remove max-pooling layers
nConvFilt = [10, 25, 50]; % Number of filters for each convolutional layer
pDrop = 0.25; % Dropout probability
nDenseHidden = 10; % Number of neurons in the fully connected layer

layers = [ ...
    % Input layer
    imageInputLayer(size(trainIms,1:3), 'Normalization', 'none')
    % First convolutional layer (no downsampling)
    convolution2dLayer(7, nConvFilt(1), 'Stride', 1) % Stride set to 1
    batchNormalizationLayer
    reluLayer
    % Second convolutional layer (no downsampling)
    convolution2dLayer(3, nConvFilt(2), 'Stride', 1) % Stride set to 1
    batchNormalizationLayer
    reluLayer
    % Third convolutional layer (no downsampling)
    convolution2dLayer(3, nConvFilt(3), 'Stride', 1) % Stride set to 1
    batchNormalizationLayer
    reluLayer
    % Global average pooling (no downsampling, reduces spatial dimensions to 1x1)
    globalAveragePooling2dLayer
    % Dropout layer
    dropoutLayer(pDrop)
    % Fully connected layer
    fullyConnectedLayer(nDenseHidden)
    reluLayer
    % Output layer for 2 classes
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer
];
%%

options = trainingOptions('adam', ...
    'MiniBatchSize',250, ...
    'MaxEpochs', 40, ...
    'ValidationData', {validIms, categorical(validPreds)}, ...
    'ValidationFrequency', 20, ...
    'Verbose',false,...
    'Plots','training-progress');
cacheNet = trainNetwork(augImgs, layers, options);
% cacheNet = trainNetwork(allIms(:,:,:,trainInd), allPreds(trainInd), layers, options);
y_hat = classify(cacheNet, validIms);
plotconfusion(y_hat, categorical(validPreds)),

%%
save(save_file, 'cacheNet'),

