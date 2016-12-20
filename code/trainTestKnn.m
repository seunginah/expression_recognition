function trainTestKnn(x_train,x_test,y_train,y_test)      
         model = fitcknn(x_train,y_train,'NumNeighbors',3,...
              'Standardize',1,'Distance','cosine'); 
            [label,score,cost] = predict(model,x_train);
            label=label.';
            fprintf('Accuracy of knn on train set:%.2f\n',mean(label==y_train));
           % disp(label);
           % disp(y_test); 
          
            [label,score,cost] = predict(model,x_test);
            label=label.';
            fprintf('Accuracy of knn on test set:%.2f\n',mean(label==y_test));
            %disp(score);
           % disp(label);
           % disp(y_test);
end