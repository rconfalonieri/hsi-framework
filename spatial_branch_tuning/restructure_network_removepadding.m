function cifar10net_layers = restructure_network_removepadding(cifar10Net)
    cifar10net_layers = cifar10Net.Layers;

    % initialize weights
    last_FC_weights_of_interest = repmat(cifar10net_layers(13).Weights(4,:), [4,1]);
    last_FC_bias_of_interest = repmat(cifar10net_layers(13).Bias(4,:), [4,1]);

    % change last layers of classifier, to handle 4 classes instead of 10
    newFClayer = fullyConnectedLayer(4, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',1, ...
        'BiasLearnRateFactor',1, ...
        'WeightL2Factor', 1,...
        'Weights', last_FC_weights_of_interest, ...
        'Bias', last_FC_bias_of_interest);

    newClassLayer = classificationLayer('Name','new_classoutput');

    cifar10net_layers(13) = newFClayer; 
    cifar10net_layers(15) = newClassLayer; 
    
    % remove padding from Conv layers
    cifar10net_layers(2).PaddingSize = [0,0,0,0];
    cifar10net_layers(5).PaddingSize = [0,0,0,0];
    cifar10net_layers(8).PaddingSize = [0,0,0,0];

    % change MaxPooling layers to have 2x2 instead of 3x3 filter size
    maxpool1 = maxPooling2dLayer(2, "Name", "maxpool1", "Stride", 2);
    maxpool2 = maxPooling2dLayer(1, "Name", "maxpool2", "Stride", 1);
    maxpool3 = maxPooling2dLayer(2, "Name", "maxpool3", "Stride", 2);
    cifar10net_layers(4) = maxpool1;
    cifar10net_layers(7) = maxpool2;
    cifar10net_layers(10) = maxpool3;

   
end
