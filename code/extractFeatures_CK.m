function features=extractFeatures_CK(images,featureType)
         %Input images is of the form imageHeight,imageWidth,numImages 
         %256x256x213-- for jaffe dataset;
      
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
                case 'fiducial_points'
                    features=FiducialPoints(images);
                case 'fiducial_points_n1'
                    features=FiducialPoints_norm1(images);    
                     
         end

         
function features=FiducialPoints(images)
         fid_x=[43,52,56,59,63,65,63,73,30,56,82,109,19,25,34,40,44,50,52,78,82,84,82,77,73,...
                79,85,90,95,101,105,109,111,50,40,30,25,24,30,37,44,47,70,65,60,55,51,48,46,80,...
                87,95,102,103,98,92,86,82,62,13,115,55,56,62,65,70,74,77,76,23,22,23,98,99,66,55,51,...
                46,55,62,84,75,80,75,23,22,23,98,99,48,65,78,68,90,67];
         fid_y=[148,142,139,138,140,140,138,140,46,45,43,46,71,66,67,67,68,68,70,142,144,150,153,...
                156,158,69,66,65,64,64,66,67,70,86,86,86,86,83,79,77,79,83,160,161,161,160,156,...
                153,149,85,86,85,85,80,78,77,79,81,175,79,75,127,121,126,127,122,122,125,18,120,...
                131,140,120,132,147,145,143,140,136,135,140,134,144,147,120,131,140,120,132,...
                116,117,119,160,144,179]; 
            
         gabor_scale=[2,4,8,16];   
         gabor_orient=[0,30,60,90,120,150,180];   
         
         gaborBank=gabor(gabor_scale,gabor_orient);
         num_filters=length(gabor_scale)*length(gabor_orient);
         num_features=num_filters*length(fid_x); %gabor wavelets at the fiducial points
         features=zeros(size(images,3),num_features);
         im=images(:,:,2);
         imtool(im/100);
         figure; imshow(im/100);
         hold on; 
         plot(fid_x,fid_y,'*');
         
         for im_num=1:size(images,3)
             im=images(:,:,im_num);
             fid_point=[];
             %figure;imshow(im/100);
             [mag,phase]=imgaborfilt(im,gaborBank);
             for p=1:num_filters
                 mag_p=mag(:,:,p);
                 %disp(size(mag_p));
                 sub=sub2ind(size(mag_p.'),fid_x,fid_y);
                 fid_val=mag_p(sub);
                 
                 fid_point=[fid_point fid_val(:).'];
             end     
             features(im_num,:)=fid_point;
         end     
         
end     
         
function features=FiducialPoints_norm1(images)
     features=FiducialPoints(images);
     features=features./sqrt(sum((features.^2+0.01),2));
end

function features=EdgeFeatures(images)
         for im_num=1:size(images,3) 
             images(:,:,im_num)=edge(images(:,:,im_num),'canny');
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
      Num_clusters=7;
      options=[3,25,0.001,0];
      [centers,U,objFn]=fcm(features,Num_clusters,options);
      feats=zeros(size(features,1),(size(features,2)+1)*Num_clusters);
      for im_num =1:size(features,1)
          f=[];
          for cluster_num=1:Num_clusters
               f=[f,U(cluster_num,im_num)];
               %f=[f,sqrt((features(im_num,:)-centers(cluster_num,:)).^2)]; 
               f=[f,((features(im_num,:)-centers(cluster_num,:)).^2)]; 
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
                %f=[f,sqrt((features(im_num,:)-centers(cluster_num,:)).^2)];
                f=[f,((features(im_num,:)-centers(cluster_num,:)).^2)];
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

function features=GaborFeatures(images)
         wavelengths=[2,4,8]; %Try other options
         orients=[0,45,90];%,135,180];
         gaborBank=gabor(wavelengths,orients);
         num_filters=length(wavelengths)*length(orients);
         num_features=2*num_filters;
         
         features=zeros(size(images,3),num_features);
         
         for im_num=1:size(images,3)
             im=images(:,:,im_num);
             
             %figure;imshow(im/100);
             [mag,phase]=imgaborfilt(im,gaborBank);
             i=1;
             %figure;
             for p=1:num_filters
                 mag_p=mag(:,:,p);
                 
              %   subplot(3,3,p);imshow(mag_p/100);
                 
                 features(im_num,i)=mean(p(:));
                 features(im_num,i+1)=var(p(:));
                 i=i+2;
             end
         end
end         
         
function features = GaborFeatures_norm1(x)
    %normalize to unit length. Blind to illumination effects.
    features = GaborFeatures(x);
    features=features./sqrt(sum((features.^2+0.01),2));
end         
   
function features = GaborFeatures_norm2(x)
    features = GaborFeatures(x);
    features(features<=0)=0;
    features=sqrt(features);
end 


function features = PixelFeatures(images)
        features = zeros(size(images,1)*size(images,2), size(images,3));
        for im_num=1:size(images,3)
             im=images(:,:,im_num);
             features(:,im_num)=reshape(permute(im,[2 1]),[size(im,1)*size(im,2) 1]);     
        end
        features=features.';
end 
    

function features = PixelFeatures_norm1(x)
    %normalize to unit length. Blind to illumination effects.
    features = PixelFeatures(x);
    features=features./sqrt(sum((features.^2+0.01),2));
end         
   
function features = PixelFeatures_norm2(x)
    features = PixelFeatures(x);
    features=sqrt(features);
end
    
function features = PixelFeatures_sharpened(images)
        %sharpen each image and extract pixel features from it.
         figure; subplot(1,2,1);imshow(images(:,:,1)/100);
        features = zeros(size(images,1)*size(images,2), size(images,3));
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
    features=sqrt(features);
end


function features = PixelFeatures_gradient(images)
        %sharpen each image and extract pixel features from it.
         figure; subplot(1,2,1);imshow(images(:,:,1)/100);
        features = zeros(size(images,1)*size(images,2), size(images,3));
        for im_num=1:size(images,3)
             im=images(:,:,im_num);
             [gx,gy]=imgradient(im,'sobel'); %The method to use is a hyperparameter.
             %images(:,:,im_num)=sqrt(gx.^2+gy.^2);
             images(:,:,im_num)=sqrt(gy.^2);
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
        num_bins=9;
        bin_orients=[0 45 90 135 180 225 270 315 360];
        cell_size=3;
        block_size=num_bins;
        num_blocks=floor(size(images,1)/(block_size))*floor(size(images,2)/(block_size));
        num_cells_per_block=floor(block_size/cell_size);
 
       cells=zeros(cell_size);
       cells_mag=zeros(cell_size);
       cells_orient=zeros(cell_size);
       features=zeros([size(images,3) (num_blocks*num_bins*num_cells_per_block*num_cells_per_block)]);
       stride=cell_size-1;
       block_stride=block_size-1;
 
       %calculate the x and y gradient of an image.
       for im_num=1:size(images,3)
           im=images(:,:,im_num);
           x_filter=[1 -1];
           y_filter=[1;-1];    
           x_grad=imfilter(im,x_filter,'same');
           y_grad=imfilter(im,y_filter,'same');
           im_mag=sqrt(x_grad.^2 + y_grad.^2);
           im_orient=atan2d(y_grad,(x_grad+0.01));
           %Make all the values to exist between 0 and 360
           im_orient(im_orient<0)=im_orient(im_orient<0)+360;
           %sweep the blocks and cells across the image
    
           feat=[];
           for block_row=1:block_stride+1:size(im,1)-block_stride
               for block_col=1:block_stride+1:size(im,2)-block_stride
                   block_mag=im_mag(block_row:block_row+block_stride,block_col:block_col+block_stride);
                   block_orient=im_orient(block_row:block_row+block_stride,block_col:block_col+block_stride);
                   blocks=[];
                   for row=1:stride+1:size(block_mag,1)-stride
                       for col=1:stride+1:size(block_mag,2)-stride
                           cells_mag=block_mag(row:row+stride,col:col+stride);
                           cells_orient=block_orient(row:row+stride,col:col+stride);
                           rems=mod(cells_orient,45); %These give how much above 
                           %the current bin orientation the given orientation exceeds.
                           quos=fix(cells_orient/45)+1; %which bins they belong to.
                           histr=zeros([1 num_bins]);
            
                    
                           for i=1:stride
                              for j=1:stride
                                  histr(quos(i,j))=histr(quos(i,j))+ceil(((45-rems(i,j))/45)*cells_mag(i,j));
                                  %add to the bin it belongs to.
                                  histr(quos(i,j)+1)=histr(quos(i,j)+1)+ceil((rems(i,j)/45)*cells_mag(i,j));
                                  %add to the next bin.
                            
                              end 
                           end
                           blocks=[blocks histr];
          
                end
            end
           
           %blocks=((blocks)./sqrt((blocks)+0.01));
           
            feat=[feat,blocks];
            end
        end      
           %disp(size(feat));
           %disp(size(features(im_num,:)));
        features(im_num,:)=feat;
                     
end

end 
    
function features = HOGFeatures_norm1(x)
    features = HOGFeatures(x);
    features=features./sqrt(sum((features.^2+0.01),2));
end

function features = HOGFeatures_norm2(x)
    features = HOGFeatures(x);
    features=sqrt(features);
end


function features = LBPFeatures(images)
 cell_size=3;
 cells=zeros(cell_size);
 stride=cell_size-1;
 block_size=30;
 num_blocks=floor(size(images,1)/(block_size))*floor(size(images,2)/(block_size));
 block_stride=block_size-1;
 features = zeros(size(images,3),255*num_blocks);
 for im_num=1:size(images,3)
     im=images(:,:,im_num);      
             blocks=[];     
             for i=1:block_stride+1:size(im,1)-block_stride
                 for j=1:block_stride+1:size(im,2)-block_stride
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
             end   
             features(im_num,:)=blocks;       
          end                  
 
end

function features = LBPFeatures_norm1(x)
    features = LBPFeatures(x);
    features=features./sqrt(sum((features.^2+0.01),2));
end

function features = LBPFeatures_norm2(x)
    features = LBPFeatures(x);
    features=sqrt(features);
end

function features = LBPFeatures_fullImage(images)
 cell_size=3;
 cells=zeros(cell_size);
 stride=cell_size-1;

 features = zeros(size(images,3),255);
 for im_num=1:size(images,3)
     im=images(:,:,im_num);
     value_at_pixel=zeros(size(im,1),size(im,2));
     for row=1:size(im,1)-stride
         for col=1:size(im,2)-stride
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
             feat=[];
             for value=1:255
                 feat=[feat,numel(find(value_at_pixel==value))];
             end
                 feat=(feat./(sqrt((feat).^2+1)));
             features(im_num,:)=feat;
          end                  
end



end