%% Make starter training set
% Take SC annotated bottom images and resample them to be the correct pixel
% dimensions for IL's data
close all,
clear all,
clc

% define data dir for training images
data_dir = 'C:\Users\xl313\OneDrive\Documents\GitHub\bird_pose_tracking\training_files\cache_annotations\';
addpath(genpath(data_dir))

% save file
save_file = 'botAnnotations_resized8_030325.mat';

% define image size
site_half_w = 8; % XL is 8, IL is 15 
site_w = site_half_w*2 + 1;

%% load the data
allData = cat(1,load(fullfile(data_dir, 'botAnnotations_AMB43_221103.mat')),...
    load(fullfile(data_dir, 'botAnnotations_LMN169_221104.mat')));

% grab the data
preIms_hd = cat(3, allData.preIms);
prePreds = cat(1, allData.prePreds);
postIms_hd = cat(3, allData.postIms);
postPreds = cat(1, allData.postPreds);

%% resample the images to be the correct size
preIms = imresize(preIms_hd, [site_w, site_w]);
postIms = imresize(postIms_hd, [site_w, site_w]);

% Display an example original and resized images
figure;
subplot(1,2,1), imshow(preIms_hd(:, :, 9), []), title('Original Image');
subplot(1,2,2), imshow(preIms(:, :, 9), []), title('Resampled Image');

%% save the updated training data
save(strcat(data_dir, save_file),'preIms','prePreds','postIms','postPreds'),


