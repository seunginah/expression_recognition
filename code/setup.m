% from the raw data, organize images into folders by emotion

% specify dataset folder: 'cohn-kanade' original, 'cohn-kanade-images' extended,
ck_dataset = 'cohn-kanade-combined'; 

% 1. create training data
% cohn-kanade/S005/001/ is a series of expressions from 1 person:
%   1st image in cohn-kanade/S005/001/ is neutral expression
%   peak image in cohn-kanade/S005/001/ is the one corresponding to the
%   emotion label, in this case: Emotion/S005/001/S005_001_00000011_emotion.txt
% each txt file in 'Emotion/' has 1 number, corresponding to an emotion
emotions = {'neutral', 'anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise', 'NA'};

% organize all the images into folders based on emotion label
rmdir('../data/training', 's') % delete any previous training ddata
mkdir('../data/training')
for i = 1:length(emotions)
    mkdir(strcat('../data/training/', emotions{i}));
end

top_level_dir = '../data/Emotion/'; % text files w labels
emotion_labels = get_filenames(top_level_dir);
labeled_images = {};
for i = 1:length(emotion_labels)
    % open file, extract emotion code
    filepath = emotion_labels{i};
    file = fopen(filepath,'r');
    emotion_code = fscanf(file, '%d');
    fprintf('emotioncode: %d\n', emotion_code)
    
    % look up the emotion
    emotion = emotions{emotion_code + 1};
    fclose(file);
    
    % organize images corresponding to the file
    start_idx = length(top_level_dir)+1;
    file_dir1 = filepath(start_idx:start_idx+3); % 17:20
    file_dir2 = filepath(start_idx+5:start_idx+7); % 22:24
    file_name = strcat(filepath(start_idx+9:start_idx+25), '.png'); % 26:42
    im_dir = fullfile('..', 'data', ck_dataset, file_dir1, file_dir2);
    labeled_images{i} = file_name;
    
    % peak image has same name as text file
    peak_im = fullfile(im_dir, file_name);
    
    % neutral image is first in the series
    im_dir_files = get_filenames(im_dir);
    neut_im = im_dir_files{1};
    
    % move the corresponding samples into the folder for that emotion
    copyfile(peak_im, strcat('../data/training/', emotion, '/'))
    copyfile(neut_im, '../data/training/neutral/')
end

% all of the images that didn't get labeled, move them to NA
top_level_dir = strcat('../data/', ck_dataset, '/');
all_images = get_filenames(top_level_dir);
all_images = all_images(~contains(all_images, '.DS_Store')); % annoying
for i = 1:length(all_images)
    filepath = all_images{i};
    % if we haven't already labeled this image
    if sum(contains(labeled_images, filepath)) == 0
        % get the path of its series, move the whole dir to NA
        start_idx = length(top_level_dir)+1;
        file_dir1 = filepath(start_idx:start_idx+3); 
        file_dir2 = filepath(start_idx+5:start_idx+7); 
        
        im_dir = fullfile('..', 'data', ck_dataset, file_dir1, file_dir2);
        copyfile(im_dir, '../data/training/NA/')
    end
end

fprintf('its done %s', '')

    