function images=drtoolbox_decomposition(images,labels,method)
         switch method
             case 'tSNE'
                 reducedImages=compute_mapping(images,method,2);
                 
                 colors=zeros(1,size(images,1));
                 cr=linspace(1,10,size(images,1));
                
                 
                 for lid=1:7
                     colors(labels==lid)=cr(lid);
                 end    
                 %disp(colors(labels==3));
                 %disp(size(reducedImages));
                 %disp(reducedImages(1:5,:));
                 
                 %Scatter plot with circles of different colors.
                 figure;scatter(reducedImages(:,1),reducedImages(:,2),[],colors,'filled');
                 %figure;scatter(reducedImages(:,1),reducedImages(:,2),'filled');
                 images=images;
         end 
end