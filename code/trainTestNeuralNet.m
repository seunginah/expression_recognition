function [trainIndices,testIndices]=trainTestNeuralNet(x_train,x_test,y_train,y_test,input)
         setdemorandstream(0);
         if nargin<5
             input=[50,10];
         end    
         net = patternnet(input,'trainscg','crossentropy'); % alist of n inputs, n+1 layers created
         %default solver rule,loss function; 
         %view(net);
         x_train=x_train.';
         x_test=x_test.';
         %disp(size(x_train));
         %disp(size(y_train));
         %Build t of shape [num_target_classes,num_samples]
         t=zeros(7,size(x_train,2));
         for row=1:7
             t(row,(y_train==row))=1;
         end
         
         [net,tr] = train(net,x_train,t);
         %plotperform(tr);
         
         pred_y=net(x_train);
         trainIndices = vec2ind(pred_y);
         %disp(trainIndices)
         %disp(y_train)
         %disp([size(trainIndices) size(y_train)]);
         fprintf('train set accuracy: %.2f\n',mean(trainIndices==y_train));
%        plotconfusion(trainIndices,y_train.');
         
         pred_y = net(x_test);
         testIndices = vec2ind(pred_y);
         %disp(testIndices);
         %disp(y_test);
         %disp([size(testIndices) size(y_test)]);
         fprintf('test set accuracy: %.2f\n',mean(testIndices==y_test));
        % plotconfusion(testIndices,y_test.');
         

end