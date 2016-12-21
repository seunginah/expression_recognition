function clusterTEmplateMatch(x_train, x_test, y_train, y_test, varargin)
%% parse inputs -- default is jaffe dataset
if length(varargin) == 0
    sprintf('* using %s *', 'jaffe')
    jaffe(x_train, x_test, y_train, y_test)
elseif length(varargin) == 1 && ~isempty(strmatch('ck', varargin{1}, 'exact'))
    sprintf('* using %s *', 'cohn-kanade')
    ck(x_train, x_test, y_train, y_test)
else
    sprintf('clusterTEmplateMatch(x_train, x_test, y_train, y_test, *%s) \n * optional param to use cohn-kanade', '"ck"')
end
end

%% for jaffe
function jaffe(x_train, x_test, y_train, y_test)

k=7;
[idx,centers] = kmeans(x_train,k);
%{
 for cnum=1:k,
     cluster=find(idx==cnum);
     fprintf('cluster %d has %d elements\n',cnum,numel(cluster));
     im_c3=original(:,:,cluster);

     for i=1:length(cluster)
         %The conglomeration here based on face shape and not
         %expression
         figure;imshow(im_c3(:,:,i)/100);
     end
 end
%}
%Test the accuracy with teh closest neighbor or k=2
%{
 dist_mat=zeros(size(x_test,1),size(centers,1));
 for test=1:size(x_test,1),
     %disp(size(centers));disp(size(x_test(test,:)));
     dist_mat(test,:)=(sqrt(sum((centers-x_test(test,:)).^2,2))).';
 end

 [M,min_ix]=min(dist_mat,[],2);
 disp(dist_mat);
 disp(min_ix);
 ypred=min_ix.';
 fprintf('ypred= \n');
 disp(ypred);
 fprintf('ytest= \n');
 disp(y_test);
 fprintf('Accuracy of template matching on test set:%.2f\n',mean(ypred==y_test));
%}

%Fit a KNN on the cluster centers and cluster id values.
y_c=1:k;
model = fitcknn(centers,y_c,'NumNeighbors',1,...
    'Standardize',1,'Distance','cosine');
[label, score, cost] = predict(model,x_test);
label=label.';
fprintf('Accuracy of template matching on test set:%.2f\n',mean(label==y_test));
%disp(score);
%disp(label);
%disp(y_test);
end

%% for cohn-kanade
function ck(x_train, x_test, y_train, y_test)
k = 7;
[idx, centers] = kmeans(x_train, k);

% Fit a KNN on the cluster centers and cluster id values.
y_c = 1:k;
model = fitcknn(centers,y_c,'NumNeighbors',1,...
    'Standardize',1,'Distance','cosine');
[label, score, cost] = predict(model,x_test);

ypred = cellstr(label)
accuracy = sum(strcmp(ypred, y_test.') / length(ypred));
fprintf('Accuracy of template matching on test set:%.2f\n', accuracy);
end