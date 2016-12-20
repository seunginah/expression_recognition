function show_emotions(images,labels)
         %Pick one image with an emotion from 1-7 classes and 
         %put it in a subplot;
         emotions=['Angry   ';'Disgust ';'Fear    ';'Happy   ';'Neutral ';'Sad     ';'Surprise'];
         %emotions=['Surprise';'Sad     ';'Disgust ';'Anger   ';'Fear    ';'Happy   '];
         %Make sure all the character vectors are of equal length--pad with
         %empty characters to force equal length.
         celldata = cellstr(emotions);
         figure;
         for i =1:7
             l=find(labels==i);
             
             l=l(1);
             im=images(:,:,l);
             subplot(3,3,i);imshow(im/100); title(celldata{i});
         end    
         
end