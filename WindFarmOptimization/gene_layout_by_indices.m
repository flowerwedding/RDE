function [framlayout] = gene_layout_by_indices(wf,pop)
        pop_size = size(pop,1);
        framlayout = zeros(pop_size, wf.rows*wf.cols);

        for i =1:pop_size
            framlayout(i, pop(i, :)) = 1;

        end
end