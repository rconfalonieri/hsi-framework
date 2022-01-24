%%%%% Script for  3bands Images for two fully connected layer net, padding is removed

% set up the experiments parameters
block_size = 32;
learningRate = 0.0001;
epochs = 1500;
num_experiments = 5;

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


% collect training samples
imagesTrain = zeros(32,32,3,50*500);
imagesLabelsTrain = zeros(1, 50*500);

for idx=1:50

   % disp(idx);

    total_num_of_images= num_clear_wood_images_train + num_soft_rot_images_train + num_brown_stain_images_train + num_blue_stain_images_train;
    firstB  = floor(48 + (384-48) * rand(1));
    secondB = floor(48 + (384-48) * rand(1));
    thirdB  = floor(48 + (384-48) * rand(1));

    % get training data
    [threeBandsImagesTrain, threeBandsImagesLabelsTrain] = load_three_bands_images(firstB, secondB, thirdB, block_size, num_clear_wood_images_train, num_soft_rot_images_train, num_brown_stain_images_train, num_blue_stain_images_train, 'train');




    idxesOfBlock=[1: total_num_of_images]+ total_num_of_images*(idx-1);
    imagesTrain(:,:,:,idxesOfBlock)=threeBandsImagesTrain;
    imagesLabelsTrain(idxesOfBlock)=threeBandsImagesLabelsTrain;

end


 % get testing data
[imagesTest, imagesLabelsTest] = load_three_bands_images(firstB, secondB, thirdB, block_size, num_clear_wood_images_test, num_soft_rot_images_test, num_brown_stain_images_test, num_blue_stain_images_test, 'test');

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


% build the network for the experiment
% load cifar10net model
load('rcnnStopSigns.mat','cifar10Net');

cifar10net_layers = restructure_network_removepadding(cifar10Net);

% select how much to train
% retrain entire network
cifar10net_layers(2).WeightLearnRateFactor=0;
cifar10net_layers(5).WeightLearnRateFactor=0;
cifar10net_layers(8).WeightLearnRateFactor=0;
cifar10net_layers(11).WeightLearnRateFactor=1;
cifar10net_layers(2).BiasLearnRateFactor=0;
cifar10net_layers(5).BiasLearnRateFactor=0;
cifar10net_layers(8).BiasLearnRateFactor=0;
cifar10net_layers(11).BiasLearnRateFactor=1;

% select options
miniBatchSize = 50;
opts = trainingOptions('sgdm', ...
    'Momentum', 0.9, ...
    'InitialLearnRate', learningRate, ...%%%0.0001,0.001
    'MaxEpochs',epochs , ...
    'MiniBatchSize', miniBatchSize, ...
    'Verbose', true);


% train 
trained_net = trainNetwork(imagesTrain,imagesLabelsTrain,cifar10net_layers, opts);

% calculate accuracy on test
[YTest,score] = classify(trained_net,imagesTest );

accuracy = sum(YTest(:) ==imagesLabelsTest(:))/numel(imagesLabelsTest(:));
fprintf('The accuracy of trial is %.4f', accuracy);

% save
save('experiments_results/trained_nets/strategy3_trials/phase1_trial4', 'trained_net');



