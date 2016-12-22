function [trainIndices1, testIndices1, accuracy] = ensembleNeuralNet(x_train,x_test,y_train,y_test, varargin)
%% parse inputs -- default is jaffe dataset
if length(varargin) == 0
    sprintf('* using %s *', 'jaffe')
    [trainIndices1, testIndices1] = jaffe(x_train, x_test, y_train, y_test)
    accuracy = [];
elseif length(varargin) == 1 && ~isempty(strmatch('ck', varargin{1}, 'exact'))
    %sprintf('* using %s *', 'cohn-kanade')
    [trainIndices1, testIndices1, accuracy] = ck(x_train, x_test, y_train, y_test)
else
    sprintf('ensembleNeuralNet(x_train, x_test, y_train, y_test, *%s) \n    * optional param to use cohn-kanade', '"ck"')
end
end

%% for jaffe images
function [trainIndices1,testIndices1] = jaffe(x_train, x_test, y_train, y_test)
[trainIndices1,testIndices1]=trainTestNeuralNet(x_train,x_test,y_train,y_test,[60,10]);
[trainIndices2,testIndices2]=trainTestNeuralNet(x_train,x_test,y_train,y_test,[80,30]);
trainIndices=floor((trainIndices1+trainIndices2)/2);
testIndices=floor((testIndices1+testIndices2)/2);
fprintf('train set accuracy: %.2f\n',mean(trainIndices==y_train));
fprintf('test set accuracy: %.2f\n',mean(testIndices==y_test));
end

%% for for cohn-kanade images
function [trainIndices1,testIndices1, accuracy] = ck(x_train, x_test, y_train, y_test)
[trainIndices1,testIndices1]=trainTestNeuralNet(x_train,x_test,y_train,y_test,[60,10], 'ck');
[trainIndices2,testIndices2]=trainTestNeuralNet(x_train,x_test,y_train,y_test,[80,30], 'ck');
trainIndices=floor((trainIndices1+trainIndices2)/2);
testIndices=floor((testIndices1+testIndices2)/2);

emotions = {'NE', 'AN', 'CO', 'DI', 'FE', 'HA', 'SA', 'SU', 'NA'};
ypred = cellstr(emotions(trainIndices));
accuracy = sum(strcmp(ypred.', y_train.') / length(ypred)); 
fprintf('train set accuracy: %.2f\n', accuracy);

ypred = cellstr(emotions(testIndices));
accuracy = sum(strcmp(ypred.', y_test.') / length(ypred)); 
fprintf('test set accuracy: %.2f\n', accuracy);
end