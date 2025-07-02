%% load and preprocess data
close all,
clear all,

%% TODO possibly won't need parts of this...see what is needed after using 
% set paths
data_dir = 'Z:\Sherry\poseTrackingXL\cacheNet\bottomCamera\';

%% 11242 labels from Selmaan
allData = cat(1,load("Z:\Sherry\poseTrackingXL\cacheNet\bottomCamera\botAnnotations_AMB43_221103.mat"),... % 3619
load("Z:\Sherry\poseTrackingXL\cacheNet\bottomCamera\botAnnotations_LMN169_221104.mat"),... % 1298
load("Z:\Sherry\poseTrackingXL\cacheNet\bottomCamera\botAnnotations_TRQ66_220728.mat"),... % 3407 (2D)
load("Z:\Sherry\poseTrackingXL\cacheNet\bottomCamera\botAnnotations_IND106_220705.mat"),... % 1728 (2D)
load("Z:\Sherry\poseTrackingXL\cacheNet\bottomCamera\botAnnotations_IND106_220701.mat"),... % 1190 (3rd label is wrong))
load("Z:\Sherry\acquisition\AMB155_031025\botAnnotations_AMB155_031025_curated.mat"),...
load("Z:\Sherry\acquisition\LVN4_040725\botAnnotations_LVN4_040725_curated.mat"));

%% checking data
% close all, 
% clear all
% load("Z:\Sherry\acquisition\AMB155_031025\botAnnotations_AMB155_031025_curated.mat");
% % load("Z:\Sherry\acquisition\LVN4_040725\botAnnotations_LVN4_040725_curated.mat")
% start = 1;
% n = 25;
% for i = start: start + n-1
%     subplot(5, 5, i-start+1); 
%     imshow(preIms(:,:,i));
%     title(sprintf('%.4f', prePreds(i)));
% end
%%
for i=1:length(allData)
    if size(allData(i).preIms,3) == 2
        allData(i).preIms = squeeze(allData(i).preIms(:,:,2,:));
        allData(i).postIms = squeeze(allData(i).postIms(:,:,2,:));
    end
end

%% 
prePreds = cat(1, allData.prePreds)>1/2;
postPreds = cat(1, allData.postPreds)>1/2;
preIms = cat(3, allData.preIms);
postIms = cat(3, allData.postIms);
preIms = reshape(preIms,[size(preIms,1), size(preIms,2),1,size(preIms,3)]);
postIms = reshape(postIms,[size(postIms,1), size(postIms,2),1,size(postIms,3)]);
nPairs = size(postIms,4);
assert(nPairs == size(preIms,4)),
sum(prePreds)
sum(postPreds)
%%
data_file = 'padded25.mat';
root_dir = 'C:\Users\xl313\OneDrive\Documents\GitHub\poseTrackingXL\';
save_dir = strcat(root_dir, 'cacheNet\');
save_file = strrep(save_dir + "cacheNet_" + string(datenum(datetime)),'.','_');

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

