function fig2png()
         filepath='trials\';
         fileList=ls(strcat(filepath,'*.fig'));
         numOfFiles=length(fileList);
         %disp(fileList);
         
         for i=1:numOfFiles             
             figName=fullfile(filepath,fileList(i,:));
             disp(figName);
             outName=fullfile(filepath,fileList(i,1:end-4)); 
             disp(outName);
             h=openfig(figName,'new','invisible');
             saveas(h,outName,'jpg'); 
             close(h);
         end 

end