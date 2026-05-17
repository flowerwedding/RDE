function [pop_indices,lu] = windfarm_init(popsize, turbine_num,wf)

    index =1:wf.rows* wf.cols;
    index(wf.NA_loc)=[];
    rand_index = randperm(length(index));
    index = index(rand_index);
    pop_indices = zeros(popsize, turbine_num);
    for i = 1: popsize
        rand_index = randperm(length(index));
        index = index(rand_index);
        pop_indices(i, :) = index(1:turbine_num);
    end
    lu = [1 * ones(1, turbine_num); wf.rows* wf.cols * ones(1, turbine_num)];
    
end