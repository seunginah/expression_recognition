function trainTestKnn(x_train, x_test, y_train, y_test, varargin)
%% parse inputs -- default is jaffe dataset
if length(varargin) == 0
    sprintf('* using %s *', 'jaffe')
    jaffe(x_train, x_test, y_train, y_test)
elseif length(varargin) == 1 && ~isempty(strmatch('ck', varargin{1}, 'exact'))
    sprintf('* using %s *', 'cohn-kanade')
    ck(x_train, x_test, y_train, y_test)
else
    sprintf('trainTestKnn(x_train, x_test, y_train, y_test, *%s) \n    * optional param to use cohn-kanade', '"ck"')
end 
end

%% for jaffe images
function jaffe(x_train, x_test, y_train, y_test)  

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


%% for for cohn-kanade images
function ck(x_train, x_test, y_train, y_test)

 model = fitcknn(x_train,y_train,'NumNeighbors',3,...
        'Standardize',1,'Distance','cosine');
    [label,score,cost] = predict(model,x_train);
    
    correct = 0;
    n = size(label, 1);
    for i = 1:n
        if all(label(i, :) == y_train(i,:))
            correct = correct + 1;
        end
    end

    fprintf('Accuracy of knn on train set:%.2f\n', (correct/n));
    % disp(label);
    % disp(y_test);

    [label,score,cost] = predict(model,x_test);
    correct = 0;
    n = size(label, 1);
    for i = 1:n
        if all(label(i, :) == y_test(i,:))
            correct = correct + 1;
        end
    end
    fprintf('Accuracy of knn on test set:%.2f\n',mean(label==y_test));
    %disp(score);
    % disp(label);
    % disp(y_test);
end

