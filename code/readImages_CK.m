%Labels: AN-1{Anger} DI-2{Disgust} FE-3{Fear} HA-4{HAppy} NE-5{Neutral} SA-6{Sad} SU -7{Surprise} 

function readImages()
         %Container map --dictionary with the above emotionTag-index
         %mapping        
         %To leverage the Matlab Dimensionality Reduction Toolbox 
         %Downloaded teh following package and added it to the path.
         addpath(genpath('./drtoolbox'));
         
         %variables to set.
         shuffle=1;
         fprintf('shuffling or no:%d\n ',shuffle);
         roi=0;
         %only one of the following to be true.
         pca_decomposition=0;
         nmf_decomposition=0;
         drtoolbox_decomp=0;
         %{
         emotionIndexMap=containers.Map;
         emotionIndexMap('AN')=1;
         emotionIndexMap('DI')=2;
         emotionIndexMap('FE')=3;
         emotionIndexMap('HA')=4;
         emotionIndexMap('NE')=5;
         emotionIndexMap('SA')=6;
         emotionIndexMap('SU')=7;
         %}
         filepath='CohnKanade\';
         fileList=ls(strcat(filepath,'*.png'));
         numOfFiles=length(fileList);
         imageWidth=size(imread(fullfile('CohnKanade\',fileList(1,:))),2);
         imageHeight=size(imread(fullfile('CohnKanade\',fileList(1,:))),1);
         
         %Constants
         r1=173; r2=173; 
         c1=170;c2=170;
         resize_factor=0.9;
         croppedImHeight=ceil((r1+r2+1)*0.9);%313;
         croppedImWidth=ceil((c1+c2+1)*0.9);%325;   
         
         images=zeros(imageHeight,imageWidth,numOfFiles);
         %croppedImages=zeros(imageHeight*0.5,imageWidth*0.5,numOfFiles);
         croppedImages=zeros(croppedImWidth,croppedImHeight,numOfFiles);
         labels=zeros(1,numOfFiles);
         
         %Preprocessing
         for i=1:numOfFiles
             labels(i)=str2num(fileList(i,8:8));
             images(:,:,i)=imread(fullfile(filepath,fileList(i,:)));  
             %images(:,:,i)=histeq(images(:,:,i));
             %croppedImages(:,:,i)=imresize(images(:,:,i),0.5);
             croppedImages(:,:,i)=imresize(images(252-c1:252+c2,372-r1:372+r2,i),resize_factor);
                      
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
            images=extractFeaturesROI_CK(croppedImages,featureType); 
            fprintf('Time to extract ROI features %.2f\n',toc);
         else    
            %Extract features.
            featureType='gabor_norm1';         
            fprintf('+++Extracting full image features...%s\n ',featureType);
            tic;
            images=extractFeatures_CK(croppedImages,featureType); 
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
         trainTestDT(trainSet,testSet,trainLabels,testLabels);
         trainTestKnn(trainSet,testSet,trainLabels,testLabels);
         clusterTEmplateMatch(original,trainSet,testSet,trainLabels,testLabels); 
         trainTestSVM(trainSet,testSet,trainLabels,testLabels);
         trainTestNeuralNet(trainSet,testSet,trainLabels,testLabels);
         ensembleNeuralNet(trainSet,testSet,trainLabels,testLabels);
end