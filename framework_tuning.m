%% load test data
[imagesTest, imagesLabelsTest] = load_subcuboid_images(32, 20, 20, 20, 20, 'test');
categories_labels{1}='clearwood';
categories_labels{2}='softrot';
categories_labels{3}='brownstain';
categories_labels{4}='bluestain';
imagesLabelsTest=categorical(imagesLabelsTest,0:3,categories_labels);
imagesTest = reshape(imagesTest, [32 32 337 1 80]);

%% load train data
[imagesTrain, imagesLabelsTrain] = load_subcuboid_images(32, 80, 80, 80, 80, 'train');
imagesLabelsTrain=categorical(imagesLabelsTrain,0:3,categories_labels);
imagesTrain = reshape(imagesTrain, [32 32 337 1 320]);
idx_train=randperm( size(imagesTrain,5));
imagesTrain=imagesTrain(:,:,:,:,idx_train);
imagesLabelsTrain=imagesLabelsTrain(idx_train);

%% load two branches
% load spatial classifier and convert it in its 3d version
load('pretrained_models/spatial_multiblock_32x32_nopadding.mat');
spatial_net_3d = spatial_3d_branch_definition(multiblock_classifier_nopadding_32);

% load spectral classifier and modify for downsampling
model_path = '/home/matlab/Downloads/SpectralClassifier/3d_conv_fungi/pretrained_models/cifar10_norm2_0.001_conv_32_fc_64_pool_3_kernel_5.mat';
[weights, bias,avg]=load_weights(model_path);
spectral_net_3d = spectral_3d_branch_definition([32, 32, 337],weights,bias,avg);

% combine classifiers
combined_graph = graph_net_definition_average(spatial_net_3d, spectral_net_3d, false);
combined_net = assembleNetwork(combined_graph);

%% TRAIN THE COMBINED NET, tuning final layers
opts = trainingOptions('sgdm', ...
    'Momentum', 0.9, ...
    'InitialLearnRate', 0.00001, ...%%%0.0001,0.001
    'MaxEpochs',100, ...
    'MiniBatchSize', 20, ...
    'Verbose', true);

combined_net = trainNetwork(imagesTrain,imagesLabelsTrain, combined_graph, opts);
save('tuned_models/lr_00001_tunelast2_100ep','combined_net');

%% EVALUATE THE COMPLETE NET
%divided in two blocks to avoid GPU memory error
for i = 1:2 
    [classifications((i-1)*40+1:i*40),score((i-1)*40+1:i*40, :)] = classify(combined_net,imagesTest(:,:,:,:,(i-1)*40+1:i*40));
end

accuracy3d = sum(classifications(:) == imagesLabelsTest(:))/numel(imagesLabelsTest(:));
confmat3d = confusionmat(imagesLabelsTest(:),classifications(:), 'Order', {'clearwood', 'softrot', 'brownstain', 'bluestain'});
% get which test image was missclassified
wrong_classified = find((classifications == imagesLabelsTest)==0);







