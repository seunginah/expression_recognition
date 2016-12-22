function [trainIndices, testIndices, accuracy] = trainTestNeuralNet(x_train,x_test,y_train,y_test, input, varargin)
%% parse inputs -- default is jaffe dataset
if length(varargin) == 0
    sprintf('* using %s *', 'jaffe')
    accuracy = 0;
    [trainIndices, testIndices, accuracy] = jaffe(x_train, x_test, y_train, y_test, input);
elseif length(varargin) == 1 && ~isempty(strmatch('ck', varargin{1}, 'exact'))
    %sprintf('* using %s *', 'cohn-kanade')
    [trainIndices, testIndices, accuracy] = ck(x_train, x_test, y_train, y_test, input);
else
    sprintf('trainTestNeuralNet(x_train, x_test, y_train, y_test, input, *%s) \n    * optional param to use cohn-kanade', '"ck"')
end 
end

%% for jaffe images
function [trainIndices, testIndices] = jaffe(x_train, x_test, y_train, y_test, input)  

setdemorandstream(0);
if nargin<5
    input=[50,10];
end
net = patternnet(input,'trainscg','crossentropy'); % alist of n inputs, n+1 layers created
%default solver rule,loss function;
%view(net);
x_train=x_train.';
x_test=x_test.';
%disp(size(x_train));
%disp(size(y_train));
%Build t of shape [num_target_classes,num_samples]
t=zeros(7,size(x_train,2));
for row=1:7
    t(row,(y_train==row))=1;
end

[net,tr] = train(net,x_train,t);
%plotperform(tr);

pred_y=net(x_train);
trainIndices = vec2ind(pred_y);
%disp(trainIndices)
%disp(y_train)
%disp([size(trainIndices) size(y_train)]);
fprintf('train set accuracy: %.2f\n',mean(trainIndices==y_train));
%        plotconfusion(trainIndices,y_train.');

pred_y = net(x_test);
testIndices = vec2ind(pred_y);
%disp(testIndices);
%disp(y_test);
%disp([size(testIndices) size(y_test)]);
fprintf('test set accuracy: %.2f\n',mean(testIndices==y_test));
% plotconfusion(testIndices,y_test.');


end

%% for for cohn-kanade images
function [trainIndices, testIndices, accuracy] = ck(x_train, x_test, y_train, y_test, input)

rng(0);
if nargin<5
    input=[50,10];
end

% alist of n inputs, n+1 layers created
net = patternnet(input,'trainscg','crossentropy');

x_train = x_train.';
x_test = x_test.';

%Build t of shape [num_target_classes,num_samples]
t=zeros(7,size(x_train,2));
emotions = {'NE', 'AN', 'CO', 'DI', 'FE', 'HA', 'SA', 'SU', 'NA'};

for row = 1:length(emotions)
    emotion = emotions{row};
    % idxes where y_train == emotion
    emotion_idxs = contains(y_train, emotion);
    t(row, emotion_idxs) = 1;
end

[net, tr] = train(net,x_train,t);

pred_y=net(x_train);
trainIndices = vec2ind(pred_y);
ypred = cellstr(emotions(trainIndices));
train_accuracy = sum(strcmp(ypred.', y_train.') / length(ypred)); 
fprintf('train set accuracy: %.2f\n', train_accuracy);

pred_y = net(x_test);
testIndices = vec2ind(pred_y);
ypred = cellstr(emotions(testIndices));
accuracy = sum(strcmp(ypred.', y_test.') / length(ypred)); 
fprintf('test set accuracy: %.2f\n', accuracy);

end