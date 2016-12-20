function images=pca_decomp(images)
         %Pca from matlab takes in input of shape NxP and
         %returns PxK where K<=P(strictly) where each column is a data vector and
         %each element of column weight for corresponding eigen vector.
        
         %images here is a NxP shape, transpose.
         %N=size(images,2);
         fprintf('Original number of features:\n'); disp(size(images));
       
         im=pca(images,'NumComponents',50);%,'NumComponents',floor(num_comp*N));
         images=images*im;
         fprintf('Final number of features:\n ');disp(size(images));
end