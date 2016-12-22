function [raw_data, labels] = align_cohn_ims(top_level)
% input: dir top_level
%   extract all subfiles as image
%   align + resize
% output: [H x W x D x N] dataset

% image file
im_paths = get_filenames(top_level);
n = length(im_paths);

% text files w labels
emotion_paths = get_filenames('../data/Emotion/');
emotionIndexMap = getEmotionIndexMap();

% select landmarks used for alignment
lmark_paths = get_filenames('../data/Landmarks/');
% each landmark file has 68 entries
eyeL = 14; % top left eyebrow
eyeR = 8; % top right eyebrow
lipL = 41; % bottom of left lip
lipR = 47; % bottom of right lip
chin = 4;

% align all images to this one
ref_im = imread('../data/training/neutral/S037_006_00000001.png');
% extract landmarks from ref image
ref_lmark_file = fopen('../data/Landmarks/S037/006/S037_006_00000001_landmarks.txt', 'r');
ref_lmark = fscanf(ref_lmark_file, '%f', [2, 68])'; % [y coord, x coord]
fclose(ref_lmark_file);

% adjust ref_size to crop under the chin, and to the right of the ear
crop_top = 50;
crop_bottom = 400;
crop_left = 200;
crop_right = 550;
ref_crop = ref_im(crop_top:crop_bottom, crop_left:crop_right);
imshowpair(ref_im, ref_crop, 'montage')

% build cropped dataset, simulataneously reading labels
raw_data = zeros(crop_bottom - crop_top, crop_right - crop_left, length(emotion_paths));
labels = cell(length(emotion_paths));
label_count = 0;

im_sizes = zeros(n, 3); % print out all the diff image sizes
for i = 1:n
    %%% read, align, crop image
    im_path = im_paths{i};
    im = imread(im_path);
    im_size = size(im);
    
    % if color image, remap to b & w
    if length(im_size) == 3
        im_sizes(i, :) = size(im);
        im = rgb2gray(im);
    else
        im_sizes(i, :) = [size(im) 0];
    end

    % use this to look up Landmarks or Emotion labels
    l = length(im_path);
    im_name = im_path(l-4-16:l-4);
    
    %%% SEARCH FOR LABEL
    % if no label found - class 'NA'
    if any(contains(emotion_paths, im_name))
        label_count = label_count + 1;
        % get Landmarks files
        lmark_path = lmark_paths{(contains(lmark_paths, im_name))};
        lmark_file = fopen(lmark_path, 'r');
        lmark = fscanf(lmark_file, '%f', [2, 68])';
        fclose(lmark_file);
        
        % use landmarks to estimate transformation
        % you can use any subset of the landmarks-better alignment than full set
        cpts = [eyeL eyeR chin];
        ref_cpts = ref_lmark(cpts, :);
        im_cpts = lmark(cpts, :);
        
        est_trans = fitgeotrans(im_cpts, ref_cpts, 'similarity');
        im_align = imwarp(im, est_trans, 'OutputView', imref2d(size(im)));

        % get label
        emo_path = emotion_paths{(contains(emotion_paths, im_name))};
        emo_file = fopen(emo_path, 'r');
        emotion_code = fscanf(emo_file, '%d');
        fprintf('(%d/%d) name: %s, emotion: %d\n', i, n, im_name, emotion_code);
        fclose(emo_file);
        
        emotion = emotionIndexMap(emotion_code + 1);
        labels{label_count} = emotion;
        
        % crop & save
        im_crop = im_align(crop_top+1:crop_bottom, crop_left+1:crop_right);
        imshowpair(im, im_crop);
        imwrite(im_crop, strcat('../paper/figs/ck-samples/', im_name, '_', emotion, '.jpg'));        
        raw_data(:, :, label_count) = im_crop;
        
    end
end

unique(im_sizes, 'rows')

end