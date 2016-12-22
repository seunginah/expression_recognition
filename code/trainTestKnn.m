function accuracy = trainTestKnn(x_train, x_test, y_train, y_test, varargin)
%% parse inputs -- default is jaffe dataset
if length(varargin) == 0
    sprintf('* using %s *', 'jaffe')
    jaffe(x_train, x_test, y_train, y_test)
    accuracy =  [];
elseif length(varargin) == 1 && ~isempty(strmatch('ck', varargin{1}, 'exact'))
    %sprintf('* using %s *', 'cohn-kanade')
    accuracy = ck(x_train, x_test, y_train, y_test)
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
function accuracy = ck(x_train, x_test, y_train, y_test)

model = fitcknn(x_train, y_train,'NumNeighbors',3,...
    'Standardize',1,'Distance','cosine');

% train
fprintf('\n...training...\n');
[label, score, cost] = predict(model, x_train);
ypred = cellstr(label);

accuracy = sum(strcmp(ypred, y_train.') / length(ypred));
fprintf('Accuracy of knn on train set: %.2f\n', accuracy);

% test

fprintf('\n...predicting...\n');
[label, score, cost] = predict(model,x_test);
ypred = cellstr(label);

accuracy = sum(strcmp(ypred, y_test.') / length(ypred));
fprintf('Accuracy of knn on test set:%.2f\n', accuracy);
end

