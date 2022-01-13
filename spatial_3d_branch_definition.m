function net = spatial_3d_branch_definition(spatial_net)

% layers and input size
input_size = spatial_net.Layers(1,1).InputSize;
% play with Mean values: reshape and convert into negative
mean_reshaped = -reshape(spatial_net.Layers(1,1).Mean, [1, 1, 1, 337]);
% 3D conv layer that simulates the reshaping operation 
simulation_weights = zeros([1, 1, 337, 1, 337]); 
for i = 1:337
    simulation_weights(:,:,i,:,i) = 1;
end
% bias rearrangement to simulate the mean normalization


layers = [
    % input layer 
    image3dInputLayer([input_size 1],"Name","image3dinput","Normalization", "none");

    
    % "fake" 3D conv that simulates reshaping
    convolution3dLayer([1 1 337], 337, "Name", "reshaping_conv", "Weights", simulation_weights, "Bias", mean_reshaped)
   
    convolution3dLayer([5 5 1],32,"Name","conv3d_1","Padding",[0 0 0;0 0 0],"Weights",reshape(spatial_net.Layers(2,1).Weights, [5, 5, 1, 337, 32]),"Bias",reshape(spatial_net.Layers(2,1).Bias, [1, 1, 1, 32]))
    reluLayer("Name","relu_1")
    maxPooling3dLayer([2 2 1],"Name","maxpool3d_1","Stride",[2 2 1])
    
    convolution3dLayer([5 5 1],32,"Name","conv3d_2","Padding",[0 0 0;0 0 0],"Weights",reshape(spatial_net.Layers(5,1).Weights, [5, 5, 1, 32, 32]),"Bias",reshape(spatial_net.Layers(5,1).Bias, [1, 1, 1, 32]))
    reluLayer("Name","relu_2")
    
    convolution3dLayer([5 5 1],64,"Name","conv3d_3","Padding",[0 0 0;0 0 0],"Weights",reshape(spatial_net.Layers(8,1).Weights, [5, 5, 1, 32, 64]),"Bias",reshape(spatial_net.Layers(8,1).Bias, [1, 1, 1, 64]))
    reluLayer("Name","relu_3")
    maxPooling3dLayer([2 2 1],"Name","maxpool3d_2","Stride",[2 2 1])
    
    convolution3dLayer([3 3 1],64,"Name","conv3d_4","Stride",[8 8 1],"Weights",reshape(spatial_net.Layers(11,1).Weights, [3, 3, 1, 64, 64]),"Bias",reshape(spatial_net.Layers(11,1).Bias, [1, 1, 1, 64]))
    reluLayer("Name","relu_4")
    
    convolution3dLayer([1 1 1],4,"Name","conv3d_5","Weights",reshape(spatial_net.Layers(13,1).Weights, [1, 1, 1, 64, 4]),"Bias",reshape(spatial_net.Layers(13,1).Bias, [1, 1, 1, 4]))
    
    softmaxLayer("Name","softmax")
    pixelClassificationLayer("Name","pixel-class",'Classes',["clearwood" "softrot", "brownstain", "bluestain"])
    ];

net = assembleNetwork(layers);
