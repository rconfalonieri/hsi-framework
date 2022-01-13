function [weights,bias,avg]=load_weights(model_path)
%model_path = '/home/matlab/h2i/implementation/classification/spectral_classifer/Matlab/model/no_norm/cifar10_90_0.0001_conv_32_fc_64_pool_3_kernel_5.mat';
%model_path = 'D:\\Unibz\\implementation\\classification\\spectral_classifer\\Matlab\\model\\cifar10_train_70_80_0.0001_0.001_0.001.mat';
load(model_path)
avg = cifar10_spectral.Layers(1,1).Mean;
weights = cell(5,1);
bias = cell(5,1);


weights{1} = zeros(1,1,5,1,32);
for i = 1:32
    weights{1}(:,:,:,:,i) = reshape(cifar10_spectral.Layers(2,1).Weights(:,1,1,i),[1,1,5,1]);
end

weights{2} = zeros(1,1,5,32,32);
for i = 1:32
    for j = 1:32
        weights{2}(:,:,:,j,i) = reshape (cifar10_spectral.Layers(5,1).Weights(:,:,j,i),[1,1,5,1]);
    end
end

weights{3} = zeros(1,1,5,32,64);

for i = 1:64
    for j = 1:32
        weights{3}(:,:,:,j,i) = reshape (cifar10_spectral.Layers(8,1).Weights(:,:,j,i),[1,1,5,1]);
    end
end

weights{4} = zeros(1,1,41,64,64);
for i = 1:64
    weights{4}(1,1,:,:,i) = reshape(cifar10_spectral.Layers(11,1).Weights(i,:),[41,64]);
end

weights{5} = zeros(1,1,1,64,4);
for i = 1:4
    weights{5}(1,1,:,:,i) = reshape(cifar10_spectral.Layers(13,1).Weights(i,:),[1,64]);
end
%%
% temp_bias1 = reshape(cifar10_spectral.Layers(2,1).Bias,[1,1,1,32]);
% temp_bias2= reshape(cifar10_spectral.Layers(5,1).Bias,[1,1,1,32]);
% temp_bias3 = reshape(cifar10_spectral.Layers(8,1).Bias,[1,1,1,64]);
% temp_bias4 = reshape(cifar10_spectral.Layers(11,1).Bias,[1,1,1,64]);
% temp_bias5 = reshape(cifar10_spectral.Layers(13,1).Bias,[1,1,1,2]);
%%
bias{1} = reshape(cifar10_spectral.Layers(2,1).Bias,[1,1,1,32]);
bias{2}= reshape(cifar10_spectral.Layers(5,1).Bias,[1,1,1,32]);
bias{3} = reshape(cifar10_spectral.Layers(8,1).Bias,[1,1,1,64]);
bias{4} = reshape(cifar10_spectral.Layers(11,1).Bias,[1,1,1,64]);
bias{5} = reshape(cifar10_spectral.Layers(13,1).Bias,[1,1,1,4]);