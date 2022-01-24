%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% set up the experiments parameters
block_size = 32;
learningRate = 0.001;
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
% load the network after phase2 of the same strategy
load('experiments_results/trained_nets/strategy3_trials/phase2_trial4.mat'); % this will create variable 'trained_net'
phase2_trained_layers = phase2_trained_net.Layers;



% select how much to train
% fine tune first conv layer and last 2 FC layers
phase2_trained_layers(2).WeightLearnRateFactor=1;
phase2_trained_layers(5).WeightLearnRateFactor=0;
phase2_trained_layers(8).WeightLearnRateFactor=0;
phase2_trained_layers(11).WeightLearnRateFactor=1;
phase2_trained_layers(13).WeightLearnRateFactor=1;
phase2_trained_layers(2).BiasLearnRateFactor=1;
phase2_trained_layers(5).BiasLearnRateFactor=0;
phase2_trained_layers(8).BiasLearnRateFactor=0;
phase2_trained_layers(11).BiasLearnRateFactor=1;
phase2_trained_layers(13).BiasLearnRateFactor=1;


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

phase3_trained_net = trainNetwork(imagesTrain,imagesLabelsTrain,phase2_trained_layers, opts);



[YTest,score] = classify(phase3_trained_net,imagesTest);
accuracy = sum(YTest(:) ==imagesLabelsTest(:))/numel(imagesLabelsTest(:));
confmat = confusionmat(imagesLabelsTest(:),YTest(:), 'Order', {'clearwood', 'softrot', 'brownstain', 'bluestain'});

save('experiments_results/trained_nets/strategy3_trials/phase3_trial4', 'phase3_trained_net');


