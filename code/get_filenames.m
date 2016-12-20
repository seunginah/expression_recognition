function files = get_filenames(top_level)
    % input: top_level is directory of a folder
    % output: list of all files under top_level (including all its
    % subdirectories)
    dir_top = dir(top_level);
    dir_idx = [dir_top.isdir];
    files = {dir_top(~dir_idx).name}';
    if ~isempty(files)
        files = cellfun(@(x) fullfile(top_level, x), files, 'UniformOutput', false);
    end
    % recurse through subdirectories and add them to file list
    sub_dirs = {dir_top(dir_idx).name};
    valid_subdir = ~ismember(sub_dirs,{'.','..'});
    for i = find(valid_subdir)                 
        sub_dir = fullfile(top_level, sub_dirs{i});
        files = [files; get_filenames(sub_dir)];  
        files = files(~contains(files, '.DS_Store'));  
    end
end