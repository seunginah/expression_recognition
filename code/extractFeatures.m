function features=extractFeatures(images,featureType)
         %Input images is of the form imageHeight,imageWidth,numImages 
         %256x256x213-- for jaffe dataset;
      
         switch featureType
                case 'pixel'
                      features = PixelFeatures(images);
                case 'pixel_norm1'
                      features = PixelFeatures_norm1(images); 
                case 'pixel_norm2'
                      features = PixelFeatures_norm2(images);
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
                case 'gabor'
                    features=GaborFeatures(images);
                case 'gabor_norm1'
                    features=GaborFeatures_norm1(images);
                case 'gabor_norm2'
                     features=GaborFeatures_norm2(images);
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
             [mag,phase]=imgaborfilt(im,gaborBank);
             i=1;
             for p=1:num_filters
                 mag_p=mag(:,:,p);
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
           for block_row=1:block_stride+1:size(im,2)-block_stride
               for block_col=1:block_stride+1:size(im,2)-block_stride
                   block_mag=im_mag(block_row:block_row+block_stride,block_col:block_col+block_stride);
                   block_orient=im_orient(block_row:block_row+block_stride,block_col:block_col+block_stride);
                   blocks=[];
                   for row=1:stride+1:size(block_mag,2)-stride
                       for col=1:stride+1:size(block_mag,1)-stride
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



end