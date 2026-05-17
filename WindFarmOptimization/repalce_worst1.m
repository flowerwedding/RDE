% function [population] = repalce_worst(population, popsize,power_order,NA_loc,popu_power)
% 
% 
% for i =1: popsize
%     turbine_pos = power_order(i, 1);
%     rnd  = randi(144);
% %     [~,ind] = sort(popu_power, 'descend');
% %     len = floor(length(ind) / 2);%length(find(popu_power(i,:)>0));
% %     new_index = ind(randi(len));
%     while sum(population(i,:)==rnd)>0
%         rnd  = randi(144);
%     end
% %     while (sum(new_index == NA_loc)>0 ) || (turbine_pos == new_index) || (sum(population(i,:) == new_index)>0)
% %         new_index = ind(randi(len));
% %     end
%     population(i,population(i,:)==turbine_pos) = rnd;
% end
% end


% function [population] = repalce_worst(population, popsize,power_order,NA_loc,popu_power)
% 
% 
% for i =1: popsize
%     turbine_pos = power_order(i, 1);
%     
%     [~,ind] = sort(popu_power, 'descend');
%     len = floor(length(ind) / 2);%length(find(popu_power(i,:)>0));
%     new_index = ind(randi(len));
% 
%     while (sum(new_index == NA_loc)>0 ) || (turbine_pos == new_index) || (sum(population(i,:) == new_index)>0)
%         new_index = ind(randi(len));
%     end
%     population(i,population(i,:) == turbine_pos) = new_index;
% end
% end


function [population] = repalce_worst(population, popsize,power_order,NA_loc,popu_power)
%% tournament selection
rand('seed',now);
for i =1: popsize
    pop_constraint  = zeros(1,144);
    pop_constraint(NA_loc)=1;
    turbine_pos = power_order(i, 1);
    
    [~,ind] = sort(popu_power(i,:), 'descend');
    len = floor(length(ind) *0.2);%length(find(popu_power(i,:)>0));
    new_index = ind(randi(len));
    pop_constraint(population(i,:))=1;
%     while (sum(new_index == NA_loc)>0 ) || (turbine_pos == new_index) || (sum(population(i,:) == new_index)>0)
% disp(pop_constraint(1,new_index))
    while (pop_constraint(1,new_index)== 1) || (turbine_pos == new_index)
        new_index = ind(randi(len));
%         fprintf('new_index:%d %d %d \n',new_index,pop_constraint(1,new_index)== 1,turbine_pos == new_index)
    end
    population(i,population(i,:) == turbine_pos) = new_index;
end
end

% function [population] = repalce_worst(population, popsize,power_order,NA_loc,popu_power)
% %% roulette wheel selection
% 
% 
% for i =1: popsize
%     q = cumsum(popu_power(i,:)) ./ (sum(popu_power(i,:)));
%     pop_constraint  = zeros(1,144);
%     pop_constraint(NA_loc)=1;
%     pop_constraint(population(i,:))=1;
%     turbine_pos = power_order(i, 1);
%     index = find(q>rand);
%     new_index = index(1);
%     
%     
% %     while (na_pop(new_index)==1sum(new_index == NA_loc)>0 ) || (turbine_pos == new_index) || (sum(population(i,:) == new_index)>0)
%     while (pop_constraint(1,new_index)==1) || (turbine_pos == new_index)
%         new_index = new_index + 1;
%         if new_index> 144
%             new_index = randi(144);
%         end
%     end
% %     disp([num2str(size(pop_constraint)),' ',num2str(new_index)])
%     population(i,population(i,:) == turbine_pos) = new_index;
% end
% end