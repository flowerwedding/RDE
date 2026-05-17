function [popu,fitness,pfit,pop_power,temp_lep] = Chaotic_selection(popu,fitness,down,up,Chaos_p,wf,D,pfit,temp_lep,Strategy,iter,radius,success_num,fail_num,LEP,pop_power,cls_type,c_r)
% lb=lu(1,:);
% ub=lu(2,:);
% randPopuList = randperm(popsize);
% randPopuList = setdiff(randPopuList,1,'stable');
% indiR1 = popold(randPopuList(1),:);
% indiR2 = popold(randPopuList(2),:);

% rand('seed',sum(100*clock));
z=0.5;
g=iter;
% radius = 0.5 * (1-iter/iterations);
ub = up;
lb = down;
switch Strategy
    case 1
        %% CJADE-R
        j=randi([1,12],1,1);
        temp_X = popu + radius * (up-down) .* (Chaos_p(j,g)-z);%r=0.1
        temp_X = windfarm_constraint(temp_X, wf.NA_loc, D,down,up);
        [fit_temp,~,~]= wf_fitness(wf,temp_X);

        if fit_temp>fitness
            popu=temp_X;
            fitness=fit_temp;
        end

    case 2
        %% CJADE-P
        temp_X=rand(12,D);
        for j=1:12
            temp_X(j,:)=popu + radius * (ub-lb) * (Chaos_p(j,g)-z);
            temp_X = windfarm_constraint(temp_X, wf.NA_loc, D,down,up);
            [fit_temp,~,~]= wf_fitness(wf,temp_X);
        end
        [num_Fbest, num_X]=max(fit_temp);
        if num_Fbest>fitness
            popu=temp_X(num_X,:);
            fitness=num_Fbest;
        end

    case 4
        %% CJADE-M
        % success and failure
        % Stochastic universal sampling
        rr = rand;
        len = floor(size(pop_power,2)/10);
        %  cumulative probability
        normfit1 = cumsum(pfit)/sum(pfit);
        index = find(rr < normfit1);
        select = index(1);
        
%         if length(index) < len
%             select(1:length(index)) = index;
%         else
%             select = index(1:len);
%         end

         switch cls_type
             case 1
                %%  individual 2d
                turbine_index = pop_power(1:len);
                ind = find(popu == turbine_index);
                
                x_index = floor( (popu(1,:) -1) / wf.rows);
                y_index = floor(popu(1,:) - 1 - x_index * wf.rows);
                for temp_i  = 1:floor(size(pop_power,2)/10)
                    ind = find(popu == turbine_index(temp_i));
                    x_index(ind) = x_index(ind) +  radius * (wf.rows) * (Chaos_p(select(temp_i),g)-z);
                    y_index(ind) = y_index(ind) +  radius * (wf.cols) * (Chaos_p(select(temp_i),g)-z);
                end
                a = rand(1,40) * radius * (wf.rows) * (Chaos_p(select,g) - z);
                b = rand(1,40) * radius * (wf.cols) * (Chaos_p(select,g) - z);
                x_index(ind) = x_index + radius * (wf.rows) * (Chaos_p(select,g)-z);
                y_index(ind) = y_index + radius * (wf.cols) * (Chaos_p(select,g)-z);
             case 2
                %% population 2D
                x_index = floor( (popu(1,:) -1) / wf.rows);
                y_index = floor(popu(1,:) - 1 - x_index * wf.rows);
                rand_temp = rand;
                switch c_r
                    case 1
                        x_index = x_index + radius * (wf.rows) * (Chaos_p(select,g)-z);
                    case 2
                        y_index = y_index + radius * (wf.cols) * (Chaos_p(select,g)-z);
                    case 3
                        x_index = x_index + radius * (wf.rows) * (Chaos_p(select,g)-z);
                        y_index = y_index + radius * (wf.cols) * (Chaos_p(select,g)-z);
                    case 4
                        if rand_temp >= 1/3
                            x_index = x_index + radius * (wf.rows) * (Chaos_p(select,g)-z);
                            y_index = y_index + radius * (wf.cols) * (Chaos_p(select,g)-z);
                        elseif (rand_temp > 1/3) &&( rand_temp < 2/3)           
                            y_index = y_index + radius * (wf.cols) * (Chaos_p(select,g)-z);
                        elseif rand_temp >= 2/3            
                            x_index = x_index + radius * (wf.rows) * (Chaos_p(select,g)-z);
                        end
                end
                temp_X =  x_index * wf.rows + y_index + 1;
             case 3
                %% population 1D
                temp_X = popu +  radius .* (ub-lb) .* (Chaos_p(select,g)-z);
         end

        temp_X = windfarm_constraint(temp_X, wf.NA_loc, D,down,up);
        [fit_temp,temp_power_order,~]= wf_fitness(wf,temp_X);


        if fit_temp>fitness
            popu=temp_X;
            fitness=fit_temp;
            pop_power = temp_power_order;
            success_num(temp_lep,select) = 1;
            fail_num(temp_lep,~select)=1;
        else
            success_num(temp_lep,:) = 0;
            fail_num(temp_lep,:)=1;
        end
        ns = success_num;
        nf = fail_num;

        for i = 1 : 12
            if (sum(ns(:, i)) + sum(nf(:, i))) == 0
                pfit(i) = 0.01; % to avoid the possible null success rates
            else
                pfit(i) = sum(ns(:, i)) / (sum(ns(:, i)) + sum(nf(:, i))) + 0.01;
            end
        end

        temp_lep = temp_lep +1;
        if temp_lep > LEP
            temp_lep = 1;
        end

    case 3
        % only success memonry
        % Stochastic universal sampling
        rr = rand;
        normfit1 = cumsum(pfit)/sum(pfit);
        index = find(rr<normfit1);
        select = index(1);

        lpcount=[];
        temp_X = popu +(radius)*Chaos_p(select,g)*0.01;%r=0.1

        temp_X = windfarm_constraint(temp_X, wf.NA_loc, D,down,up);
        [fit_temp,~]= wf_fitness(wf,temp_X);


        if fit_temp > fitness
            popu = temp_X;
            tlpcount = zeros(1, 12);
            tlpcount(select) = 1; %fitness-fit_temp;
            success_num(temp_lep,select) = 1; %fitness-fit_temp;
            lpcount = [lpcount;tlpcount];
            fitness = fit_temp;
        end
        temp_lep = temp_lep + 1;
        ns = success_num; %[ns; sum(lpcount, 1)];

        %success and failure memory
        if temp_lep + 1 >= LEP
            for i = 1 : 12
                if sum(sum(ns, 1))== 0
                    pfit(i) = 1/12;%to avoid the possible null success rates
                else
                    pfit(i) = sum(ns(:, i)) / sum(sum(ns, 1))  + 1/12;
                end
            end
            
            if temp_lep > LEP
                temp_lep = 1;
            end
        end
end
end
function temp_X = cls_wflo(popu,radius,ub,lb,Chaos_p,select,g,pop_power,cls_type)
switch cls_type
    case 1
        %%  individual 2d
        turbine_index = pop_power(1:len);
        ind = find(popu == turbine_index);
        x_index = floor( (popu(1,:) -1) / wf.rows);
        y_index = floor(popu(1,:) - 1 - x_index * wf.rows);
        for temp_i  = 1:floor(size(pop_power,2)/10)
            ind = find(popu == turbine_index(temp_i));
            x_index(ind) = x_index(ind) +  radius * (wf.rows) * (Chaos_p(select(temp_i),g)-z);
            y_index(ind) = y_index(ind) +  radius * (wf.cols) * (Chaos_p(select(temp_i),g)-z);
        end
        a = rand(1,40) * radius * (wf.rows) * (Chaos_p(select,g) - z);
        b = rand(1,40) * radius * (wf.cols) * (Chaos_p(select,g) - z);
        x_index(ind) = x_index + radius * (wf.rows) * (Chaos_p(select,g)-z);
        y_index(ind) = y_index + radius * (wf.cols) * (Chaos_p(select,g)-z);
    case 2
        %% population 2D
        x_index = floor( (popu(1,:) -1) / wf.rows);
        y_index = floor(popu(1,:) - 1 - x_index * wf.rows);
        rand_temp = rand;
        if rand_temp >= 1/3
            x_index = x_index + radius * (wf.rows) * (Chaos_p(select,g)-z);
            y_index = y_index + radius * (wf.cols) * (Chaos_p(select,g)-z);
        elseif (rand_temp > 1/3) &&( rand_temp < 2/3)
            y_index = y_index + radius * (wf.cols) * (Chaos_p(select,g)-z);
        elseif rand_temp >= 2/3
            x_index = x_index + radius * (wf.rows) * (Chaos_p(select,g)-z);
        end
        temp_X =  x_index * wf.rows + y_index + 1;
    case 3
        %% population 1D
        temp_X = popu +  radius .* (ub-lb) .* (Chaos_p(select,g)-z);
end
end
