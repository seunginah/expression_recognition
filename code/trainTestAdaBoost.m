function trainTestAdaBoost(x_train, x_test, y_train, y_test, varargin)
%% parse inputs -- default is jaffe dataset
if length(varargin) == 0
    sprintf('* using %s *', 'jaffe')
    jaffe(x_train, x_test, y_train, y_test)
elseif length(varargin) == 1 && ~isempty(strmatch('ck', varargin{1}, 'exact'))
    sprintf('* using %s *', 'cohn-kanade')
    ck(x_train, x_test, y_train, y_test)
else
    sprintf('trainTestAdaBoost(x_train, x_test, y_train, y_test, *%s) \n * optional param to use cohn-kanade', '"ck"')
end
end

%% for jaffe images
function jaffe(x_train, x_test, y_train, y_test)

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

%% for for cohn-kanade images
function ck(x_train, x_test, y_train, y_test)
t = templateTree('MaxNumSplits',5);

model = fitcensemble(x_train,y_train,'Method','AdaBoostM2','Learners',t,'CrossVal','on');
ypred = cellstr(predict(model, x_train));
accuracy = sum(strcmp(ypred, y_train.') / length(ypred)); 
fprintf('Accuracy of AdaboostM2 on train set: %.2f\n', accuracy);

ypred = cellstr(predict(model, x_test));
accuracy = sum(strcmp(ypred, y_test.') / length(ypred));
fprintf('Accuracy of AdaboostM2 on test set: %.2f\n', accuracy);

model = fitcensemble(x_train,y_train,'Method','RUSBoost','Learners',t,'CrossVal','on');
ypred = cellstr(predict(model, x_train));
accuracy = sum(strcmp(ypred, y_train.') / length(ypred)); 
fprintf('Accuracy of RUSBoost on train set: %.2f\n', accuracy);

ypred = cellstr(predict(model, x_test));
accuracy = sum(strcmp(ypred, y_test.') / length(ypred));
fprintf('Accuracy of RUSBoost on test set: %.2f\n', accuracy);

model = fitcensemble(x_train,y_train,'Method','Bag','Learners',t,'CrossVal','on');
ypred = cellstr(predict(model, x_train));
accuracy = sum(strcmp(ypred, y_train.') / length(ypred)); 
fprintf('Accuracy of Bagging on train set: %.2f\n', accuracy);

ypred = cellstr(predict(model, x_test));
accuracy = sum(strcmp(ypred, y_test.') / length(ypred));
fprintf('Accuracy of Bagging on test set: %.2f\n', accuracy);


end
