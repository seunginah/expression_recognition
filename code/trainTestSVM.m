function trainTestSVM(x_train,x_test,y_train,y_test)
         %standardize the input features;   
         t = templateSVM('Standardize',1);  
         model = fitcecoc(x_train,y_train,'Learners',t);
               %'ClassNames',{'Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise'});
         insample_loss= resubLoss(model);  
         cv_model=crossval(model);
         outofsample_loss = kfoldLoss(cv_model);
         fprintf('In sample loss %.2f\n',insample_loss);
         fprintf('Out of sample loss %.2f\n',outofsample_loss);
         rng(1); % For reproducibility
         
         label = predict(model,x_train);
         label=label.';
         fprintf('Accuracy of svm on train set: %.2f\n',mean(label==y_train));
         
         label = predict(model,x_test);
         label=label.';
         fprintf('Accuracy of svm on test set: %.2f\n',mean(label==y_test));
end