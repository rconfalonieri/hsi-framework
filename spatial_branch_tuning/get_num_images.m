% function to get number of images for specific class
function class_num_images = get_num_images(class_name, folder)
    %class_folder = dir(sprintf('/Users/DavideCremonini/DocumentiDavide/UniBZ_Master/FESR_Project_Thesis/hyperspectral-fungi/sub-cuboids_32x32/%s/%s',folder, class_name));
    class_folder = dir(sprintf('/home/matlab/Downloads/sub-cuboids_32x32/%s/%s',folder, class_name));
    class_data_indexes = find([class_folder.isdir] == 0);
    class_num_images = length(class_data_indexes);
return