%% setup.
addpath(genpath('../../drtoolbox'));   
rng(666) % set random seed
shuffle = 1;

% roi = 0;
% pca_decomposition=0;
% nmf_decomposition=0;
% drtoolbox_decomp=0;

emotionIndexMap = getEmotionIndexMap();

%% data collect + process
% specify dataset folder: 'cohn-kanade' original, 'cohn-kanade-images' extended,
ck_dataset = '../data/cohn-kanade-combined/';
emotion_labels = get_filenames('../data/Emotion/');% text files w labels
numOfFiles = length(emotion_labels);

% build cropped dataset, simulataneously reading labels
[original, labels] = align_cohn_ims(ck_dataset);

% shuffle dataset
if shuffle == 1
    idx = randperm(numOfFiles);
    croppedImages = original(:, :, idx);
    labels = labels(idx);
end

%% feature extraction
% features = {'pixel', 'pixel_norm1', 'pixel_norm2', ...
%     'pixel_sharpen', 'pixel_sharpen_norm1', 'pixel_sharpen_norm2', ...
%     'pixel_gradient', 'pixel_gradient_norm1', 'pixel_gradient_norm2', ...
%     'hog', 'hog_norm1', 'hog_norm2', ...
%     'lbp', 'lbp_norm1', 'lbp_norm2', 'lbp2', ...
%     'gabor', 'gabor_norm1', 'gabor_norm2', ...
%     'soft_clustering', 'soft_clustering_norm1', 'soft_clustering_norm2', ...
%     'edge_features', 'edge_features_n1', 'edge_features_n2', ...
%     'fiducial_points', 'fiducial_points_n1'};

features = {'lbp_norm1', 'lbp2', 'edge_features', 'edge_features_n1', 'edge_features_n2', 'fiducial_points', 'fiducial_points_n1'};


% iterate over feature extraction
results = zeros(length(features),  4, 5);
for i = 2: length(features)
    featureType = features{i};
    fprintf('+++Extracting full image features...%s\n ', featureType);
    tic;
    image_feats = extractFeatures(croppedImages,featureType);
    fprintf('Time to extract full features %.2f\n', toc);
    
    % iterate over dim reduction
    dimReduceType = 'none';
    for j = 1:4
        % no dim reduction for gabor
        if ~isempty(strfind(featureType, 'gabor'))
            images = image_feats;
        elseif  j == 1
            images = image_feats;
        elseif j == 2
            images = pca_decomp(image_feats);
            dimReduceType = 'PCA';
        elseif j == 3
            images = nnmf_decomp(image_feats);
            dimReduceType = 'NMF';
        elseif j == 4
            dimReduceType = 'tSNE';
            images = drtoolbox_decomposition(image_feats, char(labels), dimReduceType);
        end
        fprintf('+++reducing dimensions with technique...%s\n ', dimReduceType);
        % train/test split
        numTrain = floor(0.8*numOfFiles);
        trainSet = zeros(size(images,1),numTrain);
        testSet = zeros(size(images,1),(numOfFiles-numTrain));
        
        trainSet = images(1:numTrain,:);
        testSet = images(numTrain+1:numOfFiles,:);
        trainLabels = labels(1:numTrain);
        testLabels = labels(numTrain+1:numOfFiles);
        
        %% run models
        dt_acc = trainTestDT(trainSet, testSet, trainLabels, testLabels, 'ck');
        knn_acc = trainTestKnn(trainSet, testSet, trainLabels, testLabels, 'ck');
        svm_acc = trainTestSVM(trainSet, testSet, trainLabels, testLabels, 'ck');
        [~, ~, nn_acc] = trainTestNeuralNet(trainSet, testSet, trainLabels, testLabels, [50,10], 'ck');
        [~, ~, ensemble_acc] = ensembleNeuralNet(trainSet, testSet, trainLabels, testLabels, 'ck');
        
        % save result in matrix
        results_ij = [dt_acc knn_acc svm_acc nn_acc ensemble_acc];
        results(i, j, :) = results_ij;
    end
    % for each feature write as file
    dlmwrite(strcat('results/', featureType, '_', dimReduceType, '.txt'), results(i, :, :))
end


% % lbp Time to extract full features 2579.03
% if roi == 1
%     featureType = 'gabor_norm1';
%     fprintf('+++Extracting ROI features...%s\n ', featureType);
%     tic;
%     image_feats = extractFeaturesROI(croppedImages, featureType);
%     fprintf('Time to extract ROI features %.2f\n', toc);
% else
%     featureType = features{22};
%     fprintf('+++Extracting full image features...%s\n ', featureType);
%     tic;
%     image_feats = extractFeatures(croppedImages,featureType);
%     fprintf('Time to extract full features %.2f\n', toc);
% end

%% feature selection / dim reduction
% if pca_decomposition==1
%     %Pca decomposition.
%     images = pca_decomp(image_feats); %If a float value, percent of features to retain.
%
% elseif nmf_decomposition == 1
%     %NNMF decomposition.
%     images = nnmf_decomp(image_feats); %If a float value, percent of features to retain.
%
% elseif drtoolbox_decomp == 1
%     % dim reduction
%     method = 'tSNE';
%     images = drtoolbox_decomposition(image_feats, labels, method);
% else
%     images = image_feats;
% end
%

%% train-test split 80-20%
% numTrain = floor(0.8*numOfFiles);
% fprintf('Num_train(%d) Num_test(%d)\n',numTrain,size(images,1)-numTrain);
% trainSet = zeros(size(images,1),numTrain);
% testSet = zeros(size(images,1),(numOfFiles-numTrain));
% 
% trainSet = images(1:numTrain,:);
% testSet = images(numTrain+1:numOfFiles,:);
% trainLabels = labels(1:numTrain);
% testLabels = labels(numTrain+1:numOfFiles);
% 
% %% run models
% trainTestDT(trainSet, testSet, trainLabels, testLabels, 'ck');
% trainTestKnn(trainSet, testSet, trainLabels, testLabels, 'ck');
% trainTestSVM(trainSet, testSet, trainLabels, testLabels, 'ck');
% trainTestNeuralNet(trainSet, testSet, trainLabels, testLabels, [50,10], 'ck');
% sprintf('----------- %s', '--------------')
% ensembleNeuralNet(trainSet, testSet, trainLabels, testLabels, 'ck');

%trainTestAdaBoost(trainSet, testSet, trainLabels, testLabels, 'ck');
%clusterTEmplateMatch(trainSet, testSet, trainLabels, testLabels, 'ck');