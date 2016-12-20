%Labels: AN-1{Anger} DI-2{Disgust} FE-3{Fear} HA-4{HAppy} NE-5{Neutral} SA-6{Sad} SU -7{Surprise}

function readCKImages()
%Container map --dictionary with the above emotionTag-index
%mapping
shuffle=1;
roi=0;
emotionIndexMap = getEmotionIndexMap();

% specify dataset folder: 'cohn-kanade' original, 'cohn-kanade-images' extended,
ck_dataset = '../data/cohn-kanade-combined/'; 
emotion_labels = get_filenames('../data/Emotion/');% text files w labels
numOfFiles = length(emotion_labels);

% Preprocessing
% align the raw data 
%   croppedImages   [350 450 numOfFiles] 
%   labels          cell array length numOfFiles
% build cropped dataset, simulataneously reading labels
[original, labels] = align_cohn_ims(ck_dataset);
images = original;

%shuffle dataset
if shuffle==1
    ix=randperm(numOfFiles);
    croppedImages=original(:,:,ix);
    labels=labels(ix);
end

if roi==1
    featureType='pixel';
    fprintf('+++Extracting ROI features...%s\n ',featureType);
    tic;
    images=extractFeaturesROI(croppedImages,featureType);
    fprintf('Time to extract ROI features %.2f\n',toc);
else
    %Extract features.
    featureType='pixel';
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

trainLabels=char(labels(1:numTrain));
testLabels=char(labels(numTrain+1:numOfFiles));

%TRain and test DecisionTree
x_train = trainSet;
x_test = testSet;
y_train = trainLabels;
y_test = testLabels;

trainTestDT(trainSet,testSet,trainLabels,testLabels);
trainTestKnn(trainSet,testSet,trainLabels,testLabels);
clusterTEmplateMatch(original,trainSet,testSet,trainLabels,testLabels);
trainTestSVM(trainSet,testSet,trainLabels,testLabels);
end