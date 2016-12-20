function ensembleNeuralNet(x_train,x_test,y_train,y_test)
         [trainIndices1,testIndices1]=trainTestNeuralNet(x_train,x_test,y_train,y_test,[60,10]);
         [trainIndices2,testIndices2]=trainTestNeuralNet(x_train,x_test,y_train,y_test,[80,30]);
         trainIndices=floor((trainIndices1+trainIndices2)/2);
         testIndices=floor((testIndices1+testIndices2)/2);
         fprintf('train set accuracy: %.2f\n',mean(trainIndices==y_train));
         fprintf('test set accuracy: %.2f\n',mean(testIndices==y_test));
end