function trainTestAdaBoost(x_train,x_test,y_train,y_test)
         t = templateTree('MaxNumSplits',5);

         model = fitcensemble(x_train,y_train,'Method','AdaBoostM2','Learners',t,'CrossVal','on');
         label = predict(model,x_train);
         label=label.';
         fprintf('Accuracy of AdaboostM2 on train set: %.2f\n',mean(label==y_train));
         
         label = predict(model,x_test);
         label=label.';
         fprintf('Accuracy of AdaboostM2 on test set: %.2f\n',mean(label==y_test));
         
         model = fitcensemble(x_train,y_train,'Method','RUSBoost','Learners',t,'CrossVal','on');
         label = predict(model,x_train);
         label=label.';
         fprintf('Accuracy of RUSBoost on train set: %.2f\n',mean(label==y_train));
         
         label = predict(model,x_test);
         label=label.';
         fprintf('Accuracy of RUSBoost on test set: %.2f\n',mean(label==y_test));
         
          model = fitcensemble(x_train,y_train,'Method','Bag','Learners',t,'CrossVal','on');
         label = predict(model,x_train);
         label=label.';
         fprintf('Accuracy of Bagging on train set: %.2f\n',mean(label==y_train));
         
         label = predict(model,x_test);
         label=label.';
         fprintf('Accuracy of Bagging on test set: %.2f\n',mean(label==y_test));
end