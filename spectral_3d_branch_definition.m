function net = spectral_3d_branch_definition(in_size,weights, bias,avg)
% modification of Boyuan's 3d network definition, to (1) perform downsample
% and produce block-based predictions and (2) add a convolution layer that
% simulates the mean normalization

layers = [
    image3dInputLayer(in_size,"Name","image3dinput","Normalization", "none");
    % add the normalization_conv
    convolution3dLayer([1 1 1],1,"Name","norm_conv","Weights",ones([1,1,1,1,1]),"Bias",reshape(-avg, [1,1,1,1]))

    convolution3dLayer([1 1 5],32,"Name","conv3d_1","Padding",[0 0 2;0 0 2],"Weights",weights{1},"Bias",bias{1})
    reluLayer("Name","relu_1")
    maxPooling3dLayer([1 1 3],"Name","maxpool3d_1","Stride",[1 1 2])
    convolution3dLayer([1 1 5],32,"Name","conv3d_2","Padding",[0 0 2;0 0 2],"Weights",weights{2},"Bias",bias{2})
    reluLayer("Name","relu_2")
    maxPooling3dLayer([1 1 3],"Name","maxpool3d_2","Stride",[1 1 2])
    convolution3dLayer([1 1 5],64,"Name","conv3d_3","Padding",[0 0 2;0 0 2],"Weights",weights{3},"Bias",bias{3})
    reluLayer("Name","relu_3")
    maxPooling3dLayer([1 1 3],"Name","maxpool3d_3","Stride",[1 1 2])
    %convolution3dLayer([32 32 41],64,"Name","conv3d_4","Weights", repmat(weights{4}, 32, 32), "Bias",bias{4})
    convolution3dLayer([32 32 41],64,"Name","conv3d_4","Stride", [32 32 1], "Weights", repmat(weights{4}, 32, 32)/1024, "Bias",bias{4})
    reluLayer("Name","relu_4")
    convolution3dLayer([1 1 1],4,"Name","conv3d_5","Weights",weights{5},"Bias",bias{5})
    softmaxLayer("Name","softmax")
    pixelClassificationLayer("Name","classification_layers",'Classes',["clearwood" "softrot", "brownstain", "bluestain"])
    ];
net = assembleNetwork(layers);
