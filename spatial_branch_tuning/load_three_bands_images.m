
% function to load three-bands images after specifying 3 random bands to
% select. This function can be used for both training or test samples by
% specifying the parameter folder either to 'train' or 'test'
function [threeBandsImages, threeBandsImagesLabels] =load_three_bands_images(firstB, secondB, thirdB, block_size,num_clear_wood_images, num_soft_rot_images, num_brown_stain_images, num_blue_stain_images, folder)

    threeBandsImages=zeros(block_size,block_size,3,num_clear_wood_images + num_soft_rot_images + num_brown_stain_images + num_blue_stain_images);
    threeBandsImagesLabels = zeros(1, num_clear_wood_images + num_soft_rot_images + num_brown_stain_images + num_blue_stain_images);
    
    % clear wood
    clear_wood_images = getThreeBandImages(firstB, secondB, thirdB, num_clear_wood_images, block_size, 'clear_wood', folder);
    % soft rot
    soft_rot_images = getThreeBandImages(firstB, secondB, thirdB, num_soft_rot_images, block_size, 'soft_rot', folder);
    % brown stain
    brown_stain_images = getThreeBandImages(firstB, secondB, thirdB, num_brown_stain_images, block_size, 'brown_stain', folder);
    % blue stain
    blue_stain_images = getThreeBandImages(firstB, secondB, thirdB, num_blue_stain_images, block_size, 'blue_stain', folder);
    
    % process and store three-bands images for each class
    threeBandsImages(:,:,:,1:num_clear_wood_images) = clear_wood_images;
    threeBandsImages(:,:,:,num_clear_wood_images+1 : num_clear_wood_images + num_soft_rot_images) = soft_rot_images;
    threeBandsImages(:,:,:,num_clear_wood_images + num_soft_rot_images + 1 : num_clear_wood_images + num_soft_rot_images + num_brown_stain_images) = brown_stain_images;
    threeBandsImages(:,:,:,num_clear_wood_images + num_soft_rot_images + num_brown_stain_images + 1 : num_clear_wood_images + num_soft_rot_images + num_brown_stain_images + num_blue_stain_images) = blue_stain_images;

    % store labels
    threeBandsImagesLabels(1:num_clear_wood_images) = 0;
    threeBandsImagesLabels(num_clear_wood_images+1 : num_clear_wood_images + num_soft_rot_images) = 1;
    threeBandsImagesLabels(num_clear_wood_images + num_soft_rot_images + 1 : num_clear_wood_images + num_soft_rot_images + num_brown_stain_images) = 2;
    threeBandsImagesLabels(num_clear_wood_images + num_soft_rot_images + num_brown_stain_images + 1 : num_clear_wood_images + num_soft_rot_images + num_brown_stain_images + num_blue_stain_images) = 3;



    return
end


% function to get all images of specified 3 bands from training, for a
% specific class
function classImages = getThreeBandImages(firstB, secondB, thirdB, num_images, block_size, class_name, folder)
    class_folder = dir(sprintf('/home/matlab/Downloads/sub-cuboids_32x32/%s/%s', folder, class_name));
    class_data_indexes = find([class_folder.isdir] == 0);
    classImages = zeros(block_size, block_size, 3, num_images);
    
    for j = 1:num_images
        filepath = sprintf('%s/%s', class_folder(class_data_indexes(j)).folder, class_folder(class_data_indexes(j)).name);
        load(filepath, 'sub_cuboid');
        classImage=zeros(block_size,block_size,3);
        classImage(:,:,1) = sub_cuboid(:,:,firstB);
        classImage(:,:,2) = sub_cuboid(:,:,secondB);
        classImage(:,:,3)= sub_cuboid(:,:,thirdB);
        
        classImages(:,:,:,j) = classImage;
    end
return 
end 