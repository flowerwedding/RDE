%{
function [pop,pop_NA] =  gene_layout_by_indices_one(wf,indices)

    pop_NA = zeros(1, wf.rows*  wf.cols);
    pop = pop_NA;
    pop(indices) = 1;
    pop_NA(indices) = 1;
    pop_NA(wf.NA_loc)= 2;

end
%}
function [pop, pop_NA] = gene_layout_by_indices_one(wf, indices)
    % 去重并确保数量正确
    unique_indices = unique(indices);
    unique_indices = unique_indices(unique_indices > 0);
    
    % 如果数量不对，报警告
    if length(unique_indices) ~= wf.turbine_num
        % 修正数量
        if length(unique_indices) > wf.turbine_num
            unique_indices = unique_indices(randperm(length(unique_indices), wf.turbine_num));
        else
            all_positions = 1:(wf.rows * wf.cols);
            available = setdiff(all_positions, [unique_indices, wf.NA_loc]);
            needed = wf.turbine_num - length(unique_indices);
            if length(available) >= needed
                new_pos = available(randperm(length(available), needed));
                unique_indices = [unique_indices, new_pos];
            end
        end
    end
    
    pop_NA = zeros(1, wf.rows * wf.cols);
    pop = pop_NA;
    pop(unique_indices) = 1;
    pop_NA(unique_indices) = 1;
    pop_NA(wf.NA_loc) = 2;
end