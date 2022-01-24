%model_path = 'D:\\Unibz\\implementation\\classification\\spectral_classifer\\Matlab\\model\\cifar10_train_70_80_0.0001_0.001_0.001.mat';
%model_path = '/home/matlab/h2i/implementation/classification/spectral_classifer/Matlab/model/cifar10_train_70_80_0.0001_0.001_0.001.mat';
model_path = 'pretrained_models/cifar10_norm2_0.001_conv_32_fc_64_pool_3_kernel_5.mat';
[weights, bias,avg]=load_weights(model_path);
size_input = [32, 32, 337];
net = spectral_3d_branch_definition(size_input,weights,bias,avg);

% load test data
[imagesTest, imagesLabelsTest] = load_subcuboid_images(32, 20, 20, 20, 20, 'test');
categories_labels{1}='clearwood';
categories_labels{2}='softrot';
categories_labels{3}='brownstain';
categories_labels{4}='bluestain';
imagesLabelsTest=categorical(imagesLabelsTest,0:3,categories_labels);
imagesTest = reshape(imagesTest, [32 32 337 1 80]);

% block classification
% divided in two blocks to avoid GPU memory error
for i = 1:2 
    [classifications((i-1)*40+1:i*40),score((i-1)*40+1:i*40, :)] = classify(net,imagesTest(:,:,:,:,(i-1)*40+1:i*40));
end


% performance evaluation
accuracy_final = sum(classifications == imagesLabelsTest)/numel(imagesLabelsTest);
confmat_final = confusionmat(imagesLabelsTest,classifications,'Order', {'clearwood', 'softrot', 'brownstain', 'bluestain'});
% get which test image was missclassified
wrong_classified = find((classifications == imagesLabelsTest)==0);


