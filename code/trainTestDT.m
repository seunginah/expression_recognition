function trainTestDT(x_train,x_test,y_train,y_test)
     
          model=fitctree(x_train,y_train);
          ypred=predict(model,x_test);
          
          ypred=ypred.';
          
          if ~all(size(ypred) == size(y_test))
              y_test = y_test.';
          end
          
          fprintf('Accuracy of decision tree on test set:%.2f\n', sum(mean(ypred==y_test)));
          
          fprintf('++++++++++Prune tree based on least cvloss...\n'); 
          [~,~,~,bestlevel] = cvLoss(model,...
                'SubTrees','All','TreeSize','min');
          fprintf('Pruning to %d\n',bestlevel);
         
          model = prune(model,'Level',bestlevel);
          ypred=predict(model,x_test);
          ypred=ypred.';
          
          fprintf('Accuracy after pruning on test set:%.2f\n', sum(mean(ypred==y_test)));
end