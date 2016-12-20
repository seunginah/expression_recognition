% extract + align raw data
% normalize brightness
demo_mode = true; 

% for cohn:
% align images using annotated landmarks to normalize against head translation/rotation/scale

% extract all images, align by eyes, put into 1 matrix
% top_level = '../data/training/';
% raw_data = align_cohn_ims(top_level); % [H x W x N]
% save '../data/raw_data_cohn.mat' raw_data -v7.3

% load previously extracted data
load('../data/raw_data_cohn.mat')
n = size(raw_data, 3);

% normalize brightness
norm_light = zeros(size(raw_data));
for i = 1:n
    im = raw_data(:, :, i);
    max_val = max(im);
    if max_val > 1
         im = double(im./max_val); 
    end
    norm_light(:, :, i) = im;
end

% make sure no pixel < 0
norm_light(norm_light < 0) = 0;

if roi==1,  
            featureType='gabor_norm1';         
            fprintf('+++Extracting ROI features...%s\n ',featureType);
            tic;
            images=extractFeaturesROI(croppedImages,featureType); 
            fprintf('Time to extract ROI features %.2f\n',toc);
         else    
            %Extract features.
            featureType='lbp_norm1';         
            fprintf('+++Extracting full image features...%s\n ',featureType);
            tic;
            images=extractFeatures(images,featureType); 
            fprintf('Time to extract full features %.2f\n',toc);
         end
         
         %Break into 80-20%
         numTrain=floor(0.8*numOfFiles);
         trainSet=zeros(size(images,1),numTrain);
         testSet=zeros(size(images,1),(numOfFiles-numTrain));
        
         trainSet=images(1:numTrain,:);
         testSet=images(numTrain+1:numOfFiles,:);
         
         trainLabels=labels(1:numTrain);
         testLabels=labels(numTrain+1:numOfFiles);
         
         %TRain and test DecisionTree
         trainTestDT(trainSet,testSet,trainLabels,testLabels);
         trainTestKnn(trainSet,testSet,trainLabels,testLabels);
         clusterTEmplateMatch(original,trainSet,testSet,trainLabels,testLabe