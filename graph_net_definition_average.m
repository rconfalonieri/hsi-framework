function lgraph = graph_net_definition_average(spatial_net_3d,spectral_net_3d, freeze_weights)
%LAYERGRAPH CREATION combine the two branches: spatial and spectral

    lgraph = layerGraph;
    spatial_layers = spatial_net_3d.Layers(1:12);
    spectral_layers = spectral_net_3d.Layers(2:13);
    % freeze layers (but tune some parameters, comment previous part if this approach is chosen)
    if freeze_weights
       %spatial_layers = freezeLayers(spatial_layers);
       %spectral_layers = freezeLayers(spectral_layers);
       %spatial_layers(1:10) = freezeLayers(spatial_layers(1:10));
       spatial_layers(1:7) = freezeLayers(spatial_layers(1:7));
       %spectral_layers(1:10) = freezeLayers(spectral_layers(1:10));
       spectral_layers(1:7) = freezeLayers(spectral_layers(1:7));
    end 
    
    % rename layers
    for i = 1:length(spatial_layers)
        spatial_layers(i).Name = strcat("spat_", spatial_layers(i).Name);
    end 
    % add layers
    lgraph = lgraph.addLayers(spatial_layers);
    lgraph = lgraph.addLayers(spectral_layers);

    concat = concatenationLayer(4, 2, 'Name', 'concat_layer');
    
    % define the weights of the first convolution layer after the
    % concatenation
    conv_weights = zeros([1,1,1,128,8]);
    conv_weights(:,:,:,1:64, 1) = spatial_net_3d.Layers(13,1).Weights(:,:,:,:,1);
    conv_weights(:,:,:,1:64, 2) = spatial_net_3d.Layers(13,1).Weights(:,:,:,:,2);
    conv_weights(:,:,:,1:64, 3) = spatial_net_3d.Layers(13,1).Weights(:,:,:,:,3);
    conv_weights(:,:,:,1:64, 4) = spatial_net_3d.Layers(13,1).Weights(:,:,:,:,4);
    
    conv_weights(:,:,:,65:128, 5) = spectral_net_3d.Layers(14,1).Weights(:,:,:,:,1);
    conv_weights(:,:,:,65:128, 6) = spectral_net_3d.Layers(14,1).Weights(:,:,:,:,2);
    conv_weights(:,:,:,65:128, 7) = spectral_net_3d.Layers(14,1).Weights(:,:,:,:,3);
    conv_weights(:,:,:,65:128, 8) = spectral_net_3d.Layers(14,1).Weights(:,:,:,:,4);
    
    % define the weights of the last layer that performs average
    avg_weights = zeros([1 1 1 8 4]);
    for i =1:4
        avg_weights(:,:,:,i,i) = 0.5;
        avg_weights(:,:,:,4+i,i) = 0.5;
    end
    
  
    layers = [
        concat
        convolution3dLayer([1 1 1],8,"Name","combine_conv","Weights", conv_weights,"Bias",cat(4, spatial_net_3d.Layers(13,1).Bias, spectral_net_3d.Layers(14,1).Bias))
        convolution3dLayer([1 1 1],4,"Name","final_conv","Weights", avg_weights,"Bias",zeros([1,1,1,4]))
        softmaxLayer("Name","softmax")
        pixelClassificationLayer("Name","pixel-class",'Classes',["clearwood" "softrot", "brownstain", "bluestain"])
        ];

    lgraph = lgraph.addLayers(layers);
    lgraph = connectLayers(lgraph, 'spat_image3dinput', 'norm_conv');
    lgraph = connectLayers(lgraph, 'spat_relu_4', 'concat_layer/in1');
    lgraph = connectLayers(lgraph, 'relu_4', 'concat_layer/in2');
end

