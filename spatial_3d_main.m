% load test data 
[imagesTest, imagesLabelsTest] = load_subcuboid_images(32, 20, 20, 20, 20, 'test');
categories_labels{1}='clearwood';
categories_labels{2}='softrot';
categories_labels{3}='brownstain';
categories_labels{4}='bluestain';
imagesLabelsTest=categorical(imagesLabelsTest,0:3,categories_labels);


% try final version of reshaping + move_normalization
load('pretrained_models/spatial_multiblock_32x32_nopadding.mat');
spatial_net_3d_final = spatial_3d_branch_definition(multiblock_classifier_nopadding_32);

% evaluate
imagesTest3d = reshape(imagesTest, [32 32 337 1 80]);
[YTest_3d_final, score_3d] = classify(spatial_net_3d_final, imagesTest3d);
accuracy_final = sum(YTest_3d_final(:) == imagesLabelsTest(:))/numel(imagesLabelsTest(:));
confmat_final = confusionmat(imagesLabelsTest(:),YTest_3d_final(:), 'Order', {'clearwood', 'softrot', 'brownstain', 'bluestain'});
wrong_classified = find((YTest_3d_final(:) == imagesLabelsTest(:))==0);


