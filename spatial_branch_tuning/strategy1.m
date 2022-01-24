%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% set up the experiments parameters
block_size = 32;
learningRate = 0.0001;
epochs = 1500;

% define number of training samples
num_clear_wood_images_train = 140;
num_soft_rot_images_train = get_num_images('soft_rot', 'train');
num_brown_stain_images_train = get_num_images('brown_stain', 'train');
num_blue_stain_images_train = get_num_images('blue_stain', 'train');

% define number of test samples
num_clear_wood_images_test = 20;
num_soft_rot_images_test = 20;
num_brown_stain_images_test = 20;
num_blue_stain_images_test = 20;


% load train images
[imagesTrain, imagesLabelsTrain] = load_subcuboid_images(block_size, num_clear_wood_images_train, num_soft_rot_images_train, num_brown_stain_images_train, num_blue_stain_images_train, 'train');
% load train images
[imagesTest, imagesLabelsTest] = load_subcuboid_images(block_size, num_clear_wood_images_test, num_soft_rot_images_test, num_brown_stain_images_test, num_blue_stain_images_test, 'test');





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load the network for the experiment
load('rcnnStopSigns.mat','cifar10Net');
cifar10net_layers = restructure_network_removepadding(cifar10Net);
cifar10net_layers = restructure_network_handle_hyperspectral(cifar10net_layers, false);


% select how much to train
% retrain entire network
cifar10net_layers(2).WeightLearnRateFactor=1;
cifar10net_layers(5).WeightLearnRateFactor=0;
cifar10net_layers(8).WeightLearnRateFactor=0;
cifar10net_layers(11).WeightLearnRateFactor=1;
cifar10net_layers(13).WeightLearnRateFactor=1;
cifar10net_layers(2).BiasLearnRateFactor=1;
cifar10net_layers(5).BiasLearnRateFactor=0;
cifar10net_layers(8).BiasLearnRateFactor=0;
cifar10net_layers(11).BiasLearnRateFactor=1;
cifar10net_layers(13).BiasLearnRateFactor=1;


% select options
miniBatchSize = 50;
opts = trainingOptions('sgdm', ...
    'Momentum', 0.9, ...
    'InitialLearnRate', learningRate, ...%%%0.0001,0.001
    'MaxEpochs',epochs , ...
    'MiniBatchSize', miniBatchSize, ...
    'Verbose', true);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% perform classification
% define categories labels
categories_labels{1}='clearwood';
categories_labels{2}='softrot';
categories_labels{3}='brownstain';
categories_labels{4}='bluestain';

imagesLabelsTrain=categorical(imagesLabelsTrain,0:3,categories_labels);
imagesLabelsTest=categorical(imagesLabelsTest,0:3,categories_labels);

% permutate training and testing samples
% training
idx_train=randperm( size(imagesTrain,4));
imagesTrain=imagesTrain(:,:,:,idx_train);
imagesLabelsTrain=imagesLabelsTrain(idx_train);

% testing
idx_test=randperm( size(imagesTest,4));
imagesTest=imagesTest(:,:,:,idx_test);
imagesLabelsTest=imagesLabelsTest(idx_test);


trained_net = trainNetwork(imagesTrain,imagesLabelsTrain,cifar10net_layers, opts);
[YTest,score] = classify(trained_net,imagesTest);
accuracy = sum(YTest(:) ==imagesLabelsTest(:))/numel(imagesLabelsTest(:));
confmat = confusionmat(imagesLabelsTest(:),YTest(:), 'Order', {'clearwood', 'softrot', 'brownstain', 'bluestain'});


