function features=extractFeaturesROI(images,featureType)
         switch featureType
                case 'pixel'
                      features = PixelFeatures(images);
                case 'pixel_norm1'
                      features = PixelFeatures_norm1(images); 
                case 'pixel_norm2'
                      features = PixelFeatures_norm2(images);
                case 'pixel_sharpen'  
                      features = PixelFeatures_sharpened(images);
                case 'pixel_sharpen_norm1'  
                      features = PixelFeatures_sharpened_n1(images);
                case 'pixel_sharpen_norm2'  
                      features = PixelFeatures_sharpened_n2(images);
                case 'pixel_gradient'  
                      features = PixelFeatures_gradient(images);
                case 'pixel_gradient_norm1'  
                      features = PixelFeatures_gradient_n1(images);
                case 'pixel_gradient_norm2'  
                      features = PixelFeatures_gradient_n2(images);
                case 'hog'
                      features = HOGFeatures(images); 
                case 'hog_norm1'
                      features = HOGFeatures_norm1(images);      
                case 'hog_norm2'
                      features = HOGFeatures_norm2(images);  
                case 'lbp'
                      features = LBPFeatures(images); 
                case 'lbp_norm1'
                      features = LBPFeatures_norm1(images);      
                case 'lbp_norm2'
                      features = LBPFeatures_norm2(images); 
                case  'lbp2'
                    features=LBPFeatures_fullImage(images);
                case 'gabor'
                    features=GaborFeatures(images); 
                case 'gabor_norm1'
                    features=GaborFeatures_norm1(images);
                case 'gabor_norm2'
                     features=GaborFeatures_norm2(images);
                     
                case 'soft_clustering'
                     features=SoftClustering(images);
                case 'soft_clustering_norm1'
                     features=SoftClustering_n1(images);
                case 'soft_clustering_norm2'
                     features=SoftClustering_n2(images);
                     
                case 'edge_features'
                     features=EdgeFeatures(images);
                case 'edge_features_n1'
                     features=EdgeFeatures_norm1(images);
                case 'edge_features_n2'
                     features=EdgeFeatures_norm2(images);
                case 'gabor_v2'
                    %combine with pca/adaboost for dimensionality reduction
                    features=GaborFeatures_V2(images);     
                     
         end
         
function features=GaborFeatures_V2(images)
     block_c=[17,40,20,40]; %sums to 117
     block_r=[55,35,25,65]; %sums to 180
     blocks=[6,8,12,14,16];
     wavelengths=[2,4,8]; %Try other options
     orients=[0,45,90];%,135,180];
     gaborBank=gabor(wavelengths,orients);
     num_filters=length(wavelengths)*length(orients);
         %num_features=size(images,1)*size(images,2)*1*num_filters;
     num_blocks=length(blocks); 
     num_pixels=0; %Per image
     C=mat2cell(images(:,:,1),block_r,block_c);
     for i=1:num_blocks
         num_pixels=num_pixels+size(C{blocks(i)},1)*size(C{blocks(i)},2);
     end    
             
     num_features=num_pixels*num_filters;
     features=zeros(size(images,3),num_features);
         
     for im_num=1:size(images,3)
         im=images(:,:,im_num);
             %disp(size(im));
         im=imresize(im,1); 
             %disp(size(im));
             %figure;imshow(im/100
        [mag,phase]=imgaborfilt(im,gaborBank);
         i=1;
             %figure
         feats=[];
         for p=1:num_filters
             mag_p=mag(:,:,p);
             C=mat2cell(mag_p,block_r,block_c); 
             for block=blocks
                 block_im=C{block}.';
                 feats=[feats block_im(:).'];
             end              
              %  subplot(3,3,p);imshow(mag_p/100);
             end
            % disp(size(features)); disp(size(feats));
             features(im_num,:)=feats; 
         end
end  
         
         
function features=EdgeFeatures(images)
         for im_num=1:size(images,3) 
             %figure;imshow(images(:,:,im_num));
             images(:,:,im_num)=edge(images(:,:,im_num),'canny');
             %figure;imshow(images(:,:,im_num));
         end 
         features=PixelFeatures(images);
end  

function features=EdgeFeatures_norm1(images)
         for im_num=1:size(images,3) 
             images(:,:,im_num)=edge(images(:,:,im_num),'canny');
         end 
         features=PixelFeatures_norm1(images);
end  

function features=EdgeFeatures_norm2(images)
         for im_num=1:size(images,3) 
             images(:,:,im_num)=edge(images(:,:,im_num),'canny');
         end 
         features=PixelFeatures_norm2(images);
end  

function features=SoftClustering(images)
       %Get the sharpened features from the images 
        features=PixelFeatures_sharpened(images);
       %Put them through fuzzy clustering algorithm.
       Num_clusters=20;
       options=[3,25,0.001,0];
       [centers,U,objFn]=fcm(features,Num_clusters,options);
       feats=zeros(size(features,1),(size(features,2)+1)*Num_clusters);
       for im_num =1:size(features,1)
           f=[];
           for cluster_num=1:Num_clusters
                f=[f,U(cluster_num,im_num)];
                f=[f,sqrt((features(im_num,:)-centers(cluster_num,:)).^2)];      
           end
           %disp([size(feats) size(f)]);
           feats(im_num,:)=f; %membership in each cluster,distance to the cluster center 
       end    
    features=feats;     
    end

function features=SoftClustering_n1(images)
       %Get the sharpened features from the images 
        features=PixelFeatures_sharpened_n1(images);
       %Put them through fuzzy clustering algorithm.
       Num_clusters=7;
       options=[3,25,0.001,0];
       [centers,U,objFn]=fcm(features,Num_clusters,options);
       feats=zeros(size(features,1),(size(features,2)+1)*Num_clusters);
       for im_num =1:size(features,1)
           f=[];
           for cluster_num=1:Num_clusters
                f=[f,U(cluster_num,im_num)];
                f=[f,sqrt((features(im_num,:)-centers(cluster_num,:)).^2)];      
           end
           %disp([size(feats) size(f)]);
           feats(im_num,:)=f; %membership in each cluster,distance to the cluster center 
       end    
    features=feats;     
end


    
function features=SoftClustering_n2(images)
       features=SoftClustering(images);
       features=features./sqrt(sum((features.^2+0.01),2));
end
         
         
function features = LBPFeatures_fullImage(images)
 %Apply to each ROI in an image and concatenate the results;
 %Each block will give num_filt*2 outputs.
 block_c=[17,40,20,40]; %sums to 117
 block_r=[55,35,25,65]; %sums to 180
 blocks=[6,8,12,14,16];
 
 cell_size=3;
 cells=zeros(cell_size);
 stride=cell_size-1;

 features = zeros(size(images,3),255);
 for im_num=1:size(images,3)
     im=images(:,:,im_num);
      feat=[];
      C=mat2cell(im,block_r,block_c); 
      for block=blocks
          block_im=C{block};
          value_at_pixel=zeros(size(block_im,1),size(block_im,2));
     for row=1:size(block_im,1)-stride
         for col=1:size(block_im,2)-stride
             cell_patch=im(row:row+stride,col:col+stride);
             flattened_patch=cell_patch(:);
             mid_value=flattened_patch(ceil(cell_size*cell_size/2));
             cell_patch=cell_patch-mid_value;
             cell_patch(cell_patch>=0)=1;
             cell_patch(cell_patch<0)=0;
                              
             binary_val=num2str([cell_patch(2) cell_patch(3) cell_patch(6) cell_patch(8:9) ...
                                          cell_patch(7) cell_patch(4) cell_patch(1)]);
             num_val=bin2dec(binary_val);
             value_at_pixel(row,col)=num_val;
             
         end
     end
             %disp(size(feat)) 
             %disp(size(value_at_pixel(:).'))
             feat=[feat value_at_pixel(:).'];
             
      end
            f=[];
            for value=1:255
                 f=[f,numel(find(feat==value))];
             end
                 f=(f./(sqrt((f).^2+1)));
             features(im_num,:)=f;     
 end         
end 

function features=GaborFeatures(images)
       %Apply to each ROI in an image and concatenate the results;
       %Each block will give num_filt*2 outputs.
         block_c=[17,40,20,40]; %sums to 117
         block_r=[55,35,25,65]; %sums to 180
         
         blocks=[6,8,12,14,16];
        
         wavelengths=[2,3,4,5,6,7,8,9]; %Try other options
         orients=[0,15,30,45,60,90];%,135,180];
         gaborBank=gabor(wavelengths,orients);
         num_filters=length(wavelengths)*length(orients);
         num_features=length(blocks)*(2*num_filters);
         
         features=zeros(size(images,3),num_features);
         for im_num=1:size(images,3)
             im=images(:,:,im_num);
             C=mat2cell(im,block_r,block_c); 
             feat=[];
             for block=blocks
                 block_im=C{block};
                 [mag,phase]=imgaborfilt(block_im,gaborBank);
                 
                 for p=1:num_filters
                     mag_p=mag(:,:,p);
                     feat=[feat mean(mag_p(:)),var(mag_p(:))];
                 end
             end
             features(im_num,:)=feat;
         end         
   end
          
function features = GaborFeatures_norm1(x)
    %normalize to unit length. Blind to illumination effects.
    features = GaborFeatures(x);
    features=features./sqrt(sum((features.^2+0.01),2));
end         
   
function features = GaborFeatures_norm2(x)
    features = GaborFeatures(x);
    features=sqrt(features);
end

    function features=PixelFeatures(images)
             croppedImages=images;
             block_c=[17,40,20,40]; %sums to 117
             block_r=[55,35,25,65]; %sums to 180
             %blocks=[6,7,8,12,14,15,16];
             blocks=[6,8,12,14,16];
             im=croppedImages(:,:,1);
             C=mat2cell(im,block_r,block_c);
             num_blocks=length(blocks); 
             num_pixels=0; %Per image
             for i=1:num_blocks
                 num_pixels=num_pixels+size(C{blocks(i)},1)*size(C{blocks(i)},2);
             end    
             
             roi_features=zeros(size(croppedImages,3),num_pixels);
            
           
             for im_num=1:1%size(croppedImages,3),
                  im=croppedImages(:,:,im_num);
                  figure; imshow(im/100);
                  %disp(size(im));
                  C=mat2cell(im,block_r,block_c); 
                  feat=[];
                    figure;i=1;
                  for block=blocks
                      r=size(C{block},1); c=size(C{block},2);
                      
                     subplot(2,3,i); imshow(C{block}/100);i=i+1;
                 
                  %}
                     feat=[feat,reshape(C{block}.',[1,r*c])];
                  end
                  %disp(size(roi_features)); disp(size(feat));
                  roi_features(im_num,:)=feat; 
             end
             features=roi_features;
    end 

function features = PixelFeatures_norm1(x)
    %normalize to unit length. Blind to illumination effects.
    features = PixelFeatures(x);
    features=features./sqrt(sum((features.^2+0.01),2));
end         
   
function features = PixelFeatures_norm2(x)
    features = PixelFeatures(x);
    features(features<=0)=0;
    features=sqrt(features);
end

function features = PixelFeatures_sharpened(images)
        %sharpen each image and extract pixel features from it.
        figure; subplot(1,2,1);imshow(images(:,:,1)/100);
        
        for im_num=1:size(images,3)
             im=images(:,:,im_num);
             images(:,:,im_num)=imsharpen(im,'Amount',0.8,'Threshold',0.5,'Radius',2);   
        end
        subplot(1,2,2);imshow(images(:,:,1)/100);
        features=PixelFeatures(images);
end    

function features = PixelFeatures_sharpened_n1(images)
        %sharpen each image and extract pixel features from it.
        features=PixelFeatures_sharpened(images);
        features=features./sqrt(sum((features.^2+0.01),2));
end    

function features = PixelFeatures_sharpened_n2(x)
    features = PixelFeatures_sharpened(x);
    features(features<=0)=0;
    features=sqrt(features);
end

function features = PixelFeatures_gradient(images)
        %sharpen each image and extract pixel features from it.
         figure; subplot(1,2,1);imshow(images(:,:,1)/100);
        features = zeros(size(images,1)*size(images,2), size(images,3));
        for im_num=1:size(images,3)
             im=images(:,:,im_num);
             [gx,gy]=imgradient(im,'sobel'); %The method to use is a hyperparameter.
             images(:,:,im_num)=sqrt(gx.^2+gy.^2); 
             %images(:,:,im_num)=sqrt(gx.^2); 
        end
        subplot(1,2,2);imshow(images(:,:,1)/100);
        features=PixelFeatures(images);
end    

function features = PixelFeatures_gradient_n1(images)
        %sharpen each image and extract pixel features from it.
        features=PixelFeatures_gradient(images);
        features=features./sqrt(sum((features.^2+0.01),2));
end    

function features = PixelFeatures_gradient_n2(x)
    features = PixelFeatures_gradient(x);
    features(features<=0)=0;
    features=sqrt(features);
end

function features =HOGFeatures(images)
        %croppedImages as input;
        num_bins=9;
        bin_orients=[0 45 90 135 180 225 270 315 360];
        %cell_size=2;
        block_size=9;        
        %{
        num_cells_per_block=floor(block_size/cell_size);
        cells=zeros(cell_size);
        cells_mag=zeros(cell_size);
        cells_orient=zeros(cell_size);
        features=zeros([size(images,3) (num_blocks*num_bins*num_cells_per_block*num_cells_per_block)]);
        %}
        
         %stride=cell_size-1;
             block_stride=block_size-1;
             block_c=[17,40,20,40]; %sums to 117
             block_r=[55,35,25,65]; %sums to 180
             block_num=[6,8,12,14,16];
             
             
             num_blocks=length(block_num); 
             num_pixels=num_blocks*num_bins;
             features=zeros(size(images,3),num_pixels);
            
             for im_num=1:size(images,3),
                  im=images(:,:,im_num);
                                    
                 %calculate the x and y gradient of an image.
                 x_filter=[1 -1];
                 y_filter=[1;-1];    
                 x_grad=imfilter(im,x_filter,'same');
                 y_grad=imfilter(im,y_filter,'same');
                 im_mag=sqrt(x_grad.^2 + y_grad.^2);
                 im_orient=atan2d(y_grad,(x_grad+0.01));
                 %Make all the values to exist between 0 and 360
                 im_orient(im_orient<0)=im_orient(im_orient<0)+360;
                 %sweep the blocks and cells across the image
                 C_mag=mat2cell(im_mag,block_r,block_c); %6x6 shape
                 C_orient=mat2cell(im_orient,block_r,block_c); %6x6 shape
                 feat_=[];
                 for block=block_num
                     block_mag=C_mag{block};
                     block_orient=C_orient{block};              
                     %{
                     for row=1:stride+1:size(block_mag,2)-stride
                        for col=1:stride+1:size(block_mag,1)-stride
                            cells_mag=block_mag(row:row+stride,col:col+stride);
                            cells_orient=block_orient(row:row+stride,col:col+stride);
                      %}                  
                            rems=mod(block_orient,45); %These give how much above 
                            %the current bin orientation the given orientation exceeds.
                            quos=fix(block_orient/45)+1; %which bins they belong to.
                            histr=zeros([1 num_bins]);
            
                  for i=1:block_stride
                      for j=1:block_stride
                           % disp([i j]);
                           % disp([quos(i,j) rems(i,j) cells_mag(i,j)]);
                          histr(quos(i,j))=histr(quos(i,j))+ceil(((45-rems(i,j))/45)*block_mag(i,j));
                            %add to the bin it belongs to.
                          histr(quos(i,j)+1)=histr(quos(i,j)+1)+ceil((rems(i,j)/45)*block_mag(i,j));
                            %add to the next bin.
                            
                        end 
                    end
                    histr=((histr)./sqrt((histr)+0.01));
                    feat_=[feat_,histr];          
                 end 
            features(im_num,:)=feat_;     
            end
                 
        %features(im_num,:)=(features(im_num,:)./(sqrt(features(im_num,:).^2)+0.01));
        %scaling of the features.
        %{
        feature_mean=mean(features,1);
        feature_var=var(features,1);
        features=((features-feature_mean)./feature_var);
        %}
               

end 

function features = HOGFeatures_norm1(images)
    %normalize to unit length. Blind to illumination effects.
    features = HOGFeatures(images);
    features=features./sqrt(sum((features.^2+0.01),2));
end         
   
function features = HOGFeatures_norm2(images)
    features = HOGFeatures(images);
    features=sqrt(features);
end

function features = LBPFeatures(images)
         
         block_c=[17,40,20,40]; %sums to 117
         block_r=[55,35,25,65]; %sums to 180
         block_num=[6,8,12,14,16];
         stride=2;cell_size=3;
         num_blocks=length(block_num); 
         num_pixels=num_blocks*255;
         features=zeros(size(images,3),num_pixels);
 
        for im_num=1:size(images,3)
            im=images(:,:,im_num); 
             C=mat2cell(im,block_r,block_c);
             blocks=[];     
             for block=block_num
                     block_im=C{block};
                     value_at_pixel=zeros(size(block_im));
                     feat=[];
                     for row=1:size(block_im,1)-stride
                         for col=1:size(block_im,2)-stride
                              cell_patch=block_im(row:row+stride,col:col+stride);
                              flattened_patch=cell_patch(:);
                              mid_value=flattened_patch(ceil(cell_size*cell_size/2));
                              cell_patch=cell_patch-mid_value;
                              cell_patch(cell_patch>=0)=1;
                              cell_patch(cell_patch<0)=0;
                              %cell_patch=permute(cell_patch,[2 1]);
                              binary_val=num2str([cell_patch(2) cell_patch(3) cell_patch(6) cell_patch(8:9) ...
                                           cell_patch(7) cell_patch(4) cell_patch(1)]);
            
                              num_val=bin2dec(binary_val);
                              value_at_pixel(row,col)=num_val;
             
                         end
                        end
                               for value=1:255
                                 feat=[feat,numel(find(value_at_pixel==value))];
                               end
                               feat=(feat./(sqrt((feat).^2+1)));
                               blocks=[blocks,feat];              
             end
        features(im_num,:)=blocks;       
        end
                          
end                  
 


function features = LBPFeatures_BoW(x)
 cell_size=3;
 cells=zeros(cell_size);
 stride=cell_size-1;
 block_size=9;
 num_blocks=floor(size(x,1)/block_size)^2;
 
 patches=zeros(size(x,4)*num_blocks,255);
 patch_num=1;
 
 features = zeros(size(x,4),255*num_blocks);
 for im_num=1:size(x,4)
     im=x(:,:,1,im_num); 
     block_stride=block_size-1;
                
             for i=1:block_stride+1:size(im,1)-block_stride
                 for j=1:block_stride+1:size(im,1)-block_stride
                     block_im=im(i:i+block_stride,j:j+block_stride);
                     value_at_pixel=zeros(size(block_im));
                     feat=[];
                     for row=1:size(block_im,1)-stride
                         for col=1:size(block_im,2)-stride
                              cell_patch=block_im(row:row+stride,col:col+stride);
                              flattened_patch=cell_patch(:);
                              mid_value=flattened_patch(ceil(cell_size*cell_size/2));
                              cell_patch=cell_patch-mid_value;
                              cell_patch(cell_patch>=0)=1;
                              cell_patch(cell_patch<0)=0;
                              %cell_patch=permute(cell_patch,[2 1]);
                              binary_val=num2str([cell_patch(2) cell_patch(3) cell_patch(6) cell_patch(8:9) ...
                                           cell_patch(7) cell_patch(4) cell_patch(1)]);
            
                              num_val=bin2dec(binary_val);
                              value_at_pixel(row,col)=num_val;
             
                         end
                        end
                               for value=1:255
                                 feat=[feat,numel(find(value_at_pixel==value))];
                               end
                               %feat=(feat./(sqrt((feat).^2+1)));
							   patches(patch_num,:)=feat;
                               patch_num=patch_num+1;
                                            
                 end
             end   
                   
          end                  
k=10;
[cluster_indices,cluster_centers]=kmeans(patches,k); %patches is a nx255 matrix and k is number of clusters;   
new_features=zeros(size(features,1),k);

im_num=1;
disp(size(patches));
disp((size(patches,1)-num_blocks));
 for im=1:num_blocks:((size(patches,1)-num_blocks)+1)%2*255:255
    % disp(im);
     patch=cluster_indices(im:im+num_blocks-1);
     disp(im_num);disp(patch);
	for cc=1:k
	   new_features(im_num,cc)=new_features(im_num,cc)+numel(patch(patch==cc)) ;
       %disp(numel(patch(patch==cc)));
    end
    %disp(new_features(im,:));
    im_num=im_num+1;
 end
 disp(new_features(1:5,:));
 
 features=new_features.';
 
 features=sqrt(features);
  
 %features=features./sqrt(sum((features.^2+0.01),2));
end

function features = LBPFeatures_BoW_norm1(x)
    features = LBPFeatures_BoW(x);
    %features(:,im_num)=features(:,im_num)./(sqrt((features(:,im_num).^2)+0.01));
    features=features./sqrt(sum((features.^2+0.01),2));
end

function features = LBPFeatures_BoW_norm2(x)
    features = LBPFeatures_BoW(x);
    features=sqrt(features);
end


function features = LBPFeatures_norm1(x)
    features = LBPFeatures(x);
    %features(:,im_num)=features(:,im_num)./(sqrt((features(:,im_num).^2)+0.01));
    features=features./sqrt(sum((features.^2+0.01),2));
end

function features = LBPFeatures_norm2(x)
    features = LBPFeatures(x);
    features=sqrt(features);
end




end