function trainTestHCRF(x_train, x_test, y_train, y_test, varargin)
%% parse inputs -- default is jaffe dataset
if length(varargin) == 0
    sprintf('* using %s *', 'jaffe')
    jaffe(x_train, x_test, y_train, y_test)
elseif length(varargin) == 1 && ~isempty(strmatch('ck', varargin{1}, 'exact'))
    sprintf('* using %s *', 'cohn-kanade')
    ck(x_train, x_test, y_train, y_test)
else
    sprintf('trainTestHCRF(x_train, x_test, y_train, y_test, *%s) \n    * optional param to use cohn-kanade', '"ck"')
end 
end

%% for jaffe images
function jaffe(x_train, x_test, y_train, y_test)

disp(size(x_train));
disp(size(y_train));

end

%% for for cohn-kanade images
function ck(x_train, x_test, y_train, y_test)
end