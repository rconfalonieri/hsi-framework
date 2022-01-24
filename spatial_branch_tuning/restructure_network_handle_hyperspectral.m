function hyperspectral_layers = restructure_network_handle_hyperspectral(network, is_network)

    if is_network
        hyperspectral_layers = network.Layers;
    else
        hyperspectral_layers = network; 
    end 
    % adapt to handle hyperspectral images
    % image input layer
    new_imagelayer = imageInputLayer([32, 32, 337], 'Name', 'new_imageinputlayer');

    % conv layer
    w = hyperspectral_layers(2).Weights;

    for ii=1:32
        win=w(:,:,:,ii);
        wout=repmat(win,1,1,113);
        wout=wout(:,:,1:337);
        wout_total(:,:,:,ii)=wout;
    end


    new_convlayer = convolution2dLayer([5,5], 32, ...
        'NumChannels', 337,...
        'Stride', 1,...
        'Padding', 0, ...
        'DilationFactor', [1,1],...
        'Name', 'new_conv', 'WeightsInitializer', 'narrow-normal', ...
        'Weights', wout_total, ...
        'Bias', hyperspectral_layers(2).Bias);

    hyperspectral_layers(1) = new_imagelayer;
    hyperspectral_layers(2) = new_convlayer; 
end