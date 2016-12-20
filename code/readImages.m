%Labels: AN-1{Anger} DI-2{Disgust} FE-3{Fear} HA-4{HAppy} NE-5{Neutral} SA-6{Sad} SU -7{Surprise} 

function readImages()
         %Container map --dictionary with the above emotionTag-index
         %mapping        
         %To leverage the Matlab Dimensionality Reduction Toolbox 
         %Downloaded teh following package and added it to the path.
         %addpath(genpath('./drtoolbox'));     
         
         %variables to set.
         shuffle=1;
         fprintf('shuffling or no:%d\n ',shuffle);
         roi=0;
         %only one of the following to be true.
         pca_decomposition=1;
         nmf_decomposition=0;
         drtoolbox_decomp=0;
         
         emotionIndexMap=containers.Map;
         emotionIndexMap('AN')=1;
         emotionIndexMap('DI')=2;
         emotionIndexMap('FE')=3;
         emotionIndexMap('HA')=4;
         emotionIndexMap('NE')=5;
         emotionIndexMap('SA')=6;
         emotionIndexMap('SU')=7;
         
         filepath='jaffe\';
         fileList=ls(strcat(filepath,'*.tiff'));
         numOfFiles=length(fileList);
         imageWidth=size(imread(fullfile('jaffe\',fileList(1,:))),2);
         imageHeight=size(imread(fullfile('jaffe\',fileList(1,:))),1);
         
         %Constants
         r1=90; r2=109; 
         c1=70;c2=59;
         resize_factor=0.9;
         croppedImHeight=(r1+r2+1)*0.9;%180;
         croppedImWidth=(c1+c2+1)*0.9;%117;   
         
         images=zeros(imageHeight,imageWidth,numOfFiles);
         croppedImages=zeros(croppedImHeight,croppedImWidth,numOfFiles);
         labels=zeros(1,numOfFiles);
         
         %Preprocessing
         for i=1:numOfFiles
             labels(i)=emotionIndexMap(fileList(i,4:5));
             images(:,:,i)=imread(fullfile(filepath,fileList(i,:)));            
             croppedImages(:,:,i)=imresize(images(128-r1:128+r2,128-c1:128+c2,i),resize_factor);
                      
         end
         %imtool(croppedImages(:,:,213)/100);
         
         %shuffle dataset
         if shuffle==1
            ix=randperm(numOfFiles);
            croppedImages=croppedImages(:,:,ix);
            labels=labels(ix);
         end
         %this will make a plot with images from each emotion.
         show_emotions(croppedImages,labels);
         
         original=croppedImages;
         if roi==1,  
            featureType='lbp2';         
            fprintf('+++Extracting ROI features...%s\n ',featureType);
            tic;
            images=extractFeaturesROI(croppedImages,featureType); 
            fprintf('Time to extract ROI features %.2f\n',toc);
         else    
            %Extract features.
            featureType='gabor_pixel_sharpen';         
            fprintf('+++Extracting full image features...%s\n ',featureType);
            tic;
            images=extractFeatures(croppedImages,featureType); 
            fprintf('Time to extract full features %.2f\n',toc);
         end
         
         %Pca decomposition.
         if pca_decomposition==1,
             images=pca_decomp(images); %If a float value, percent of features to retain.
         end
         
         %NNMF decomposition.
         if nmf_decomposition==1,
             images=nnmf_decomp(images); %If a float value, percent of features to retain.
         end
         
         if drtoolbox_decomp==1,
             method='tSNE';
             images=drtoolbox_decomposition(images,labels,method);
         end
         
         %Break into 80-20%
         numTrain=floor(0.8*numOfFiles);
         fprintf('Num_train(%d) Num_test(%d)\n',numTrain,size(images,1)-numTrain);
         trainSet=zeros(size(images,1),numTrain);
         testSet=zeros(size(images,1),(numOfFiles-numTrain));
        
         trainSet=images(1:numTrain,:);
         testSet=images(numTrain+1:numOfFiles,:);
         
         trainLabels=labels(1:numTrain);
         testLabels=labels(numTrain+1:numOfFiles);
         
         %TRain and test DecisionTree
         %trainTestAdaBoost(trainSet,testSet,trainLabels,testLabels);
         trainTestDT(trainSet,testSet,trainLabels,testLabels);
         trainTestKnn(trainSet,testSet,trainLabels,testLabels);
         clusterTEmplateMatch(original,trainSet,testSet,trainLabels,testLabels); 
         trainTestSVM(trainSet,testSet,trainLabels,testLabels);
         trainTestNeuralNet(trainSet,testSet,trainLabels,testLabels);
         ensembleNeuralNet(trainSet,testSet,trainLabels,testLabels);
end