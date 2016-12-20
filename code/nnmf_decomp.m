function images=nnmf_decomp(images)
         %Input of shape MxN --outputs W MxK {reduced input} and H KxN{basis}
         disp(size(images));
         %k<min(M,N)
         k=100;
         opt = statset('MaxIter',10,'Display','final');

         [W,H]=nnmf(images,k,'replicates',10,'options',opt,'algorithm','als');
         images=W;
         
         disp(size(images)); 

end