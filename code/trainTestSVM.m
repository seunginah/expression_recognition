function accuracy = trainTestSVM(x_train, x_test, y_train, y_test, varargin)
%% parse inputs -- default is jaffe dataset
if length(varargin) == 0
    sprintf('* using %s *', 'jaffe')
    jaffe(x_train, x_test, y_train, y_test)
    accuracy = [];
elseif length(varargin) == 1 && ~isempty(strmatch('ck', varargin{1}, 'exact'))
    %sprintf('* using %s *', 'cohn-kanade')
    accuracy = ck(x_train, x_test, y_train, y_test)
else
    sprintf('trainTestSVM(x_train, x_test, y_train, y_test, *%s) \n    * optional param to use cohn-kanade', '"ck"')
end
end

%% for jaffe images
function jaffe(x_train, x_test, y_train, y_test)

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

%% for for cohn-kanade images
function accuracy =  ck(x_train, x_test, y_train, y_test)

%standardize the input features;
t = templateSVM('Standardize',1);
model = fitcecoc(x_train,y_train,'Learners',t);

insample_loss = resubLoss(model);
cv_model = crossval(model);
outofsample_loss = kfoldLoss(cv_model);
fprintf('In sample loss: %.2f, Out of sample loss: %.2f\n', insample_loss, outofsample_loss);

rng(1); % For reproducibility
ypred = cellstr(predict(model, x_train));
accuracy = sum(strcmp(ypred, y_train.') / length(ypred)); 
fprintf('Accuracy of svm on train set: %.2f\n', accuracy);

ypred = cellstr(predict(model, x_test));
accuracy = sum(strcmp(ypred, y_test.') / length(ypred));
fprintf('Accuracy of svm on test set: %.2f\n', accuracy);
end

