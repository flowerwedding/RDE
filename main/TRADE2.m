% ----------------------------------------------------------------------------
% TRADE Algorithm for Wind Farm Layout Optimization
% Modified for Wind Dataset
% ----------------------------------------------------------------------------

clear all;
clc;

format long;
format compact;

%% path setup
problem_path = '../WindFarmOptimization/';
ws_path =  '../WindFarmOptimization/windscenarios';
save_path = '../Results/';
addpath(problem_path)
addpath(ws_path)

%% experiment setup
runTime = 3;
popsize = 50;
max_it = 400;

%% wind farm parameters
rows = 21;
cols = 21;
cell_width = 77.0 * 3;

%% 3种风机规模
turbine_num = [30]; %风力发电机组的规模
NA_type_list = 0;

%% 4种风电场场景wind scenarios
n_speeds = [3, 3, 4, 6]; %风速数量
n_directions = [12, 12, 12, 12]; %风向数量
unifrom = [0, 1, 0, 0]; %是否均匀分布

%% 设置标记间隔
marker_interval = 5000; % 每 5,000 评估标记一次（根据最大迭代次数调整）
marker_points = [0:marker_interval:max_it*popsize, max_it*popsize];

%% Para
algorithmDir = 'TRADE_Wind';
total_st = tic;

text_path = sprintf('%s/%s/', save_path, algorithmDir);
text_file = sprintf('%s/cost_time.txt', text_path);
if ~exist(text_path, 'dir')
    mkdir(text_path)
end
f = fopen(text_file, 'a');
fprintf(f, datestr(now));
fprintf(f, '\n');

for wt = 1:length(n_speeds)
    for tn = turbine_num
        for NA_type = NA_type_list
            NA_loc_array = gene_NA_loc(NA_type);
            [wf, ws_folder] = gene_windfram(rows, cols, tn, cell_width, NA_loc_array, n_speeds(wt), n_directions(wt), unifrom(wt));
            
            % 初始化存储所有运行结果的数组
            all_run_results = zeros(length(marker_points), runTime); % [标记点数×运行次数]
            result = zeros(1, 5); % 统计结果：[最小值,最大值,中位数,均值,标准差]
            
            D = wf.turbine_num; % 变量维度（风机数量）
            lu = [ones(1, D); wf.rows * wf.cols * ones(1, D)]; % 边界[1, 网格总数]
            
            folder = sprintf('%s/%s/%s/tn%d_NA%d', save_path, algorithmDir, ws_folder, tn, NA_type);
            if ~exist(folder, 'dir')
                mkdir(folder)
            end
            
            fprintf('\n-------------------------------------------------------\n')
            fprintf('Wind Scenario = %s, Turbine Num = %d\n', ws_folder, tn)
            
            outcome = [];
            
            for run = 1:runTime
                fprintf('运行 %d/%d...\n', run, runTime);
                run_st = tic;
                
                %% TRADE Algorithm Initialization
                popSize = popsize;
                nFES = 0;
                
                % 随机初始化种群（风电场布局）
                Pop = zeros(popSize, D);
                for i = 1:popSize
                    available_positions = setdiff(1:wf.rows*wf.cols, wf.NA_loc);
                    selected_positions = available_positions(randperm(length(available_positions), D));
                    Pop(i, :) = sort(selected_positions);
                end
                
                % 计算适应度（注意：风电场问题是最大化问题）
                popFit = zeros(popSize, 1);
                for i = 1:popSize
                    [fitness, ~, ~] = wf_fitness(wf, Pop(i, :));
                    popFit(i) = fitness;
                end
                nFES = nFES + popSize;
                
                [bsff, bsfi] = max(popFit); % 风电场是最大化问题
                bsfp = Pop(bsfi, :);
                
                %% Initialize TRADE-specific parameters
                lPopSize = 0.5 * popSize;
                gPopSize = popSize - lPopSize;
                gFlag = zeros(gPopSize, 1);
                gR = ones(gPopSize, 1);
                lSeed = randperm(popSize, lPopSize);
                gSeed = setdiff(1:popSize, lSeed);
                lPop = Pop(lSeed, :);
                lPopFit = popFit(lSeed);
                gPop = Pop(gSeed, :);
                gPopFit = popFit(gSeed);
                subSize = ones(lPopSize, 1);
                flag = zeros(lPopSize, 1);
                theta = 1.5 * D;
                if theta < 30
                    theta = 30;
                end
                gTh = popSize;
                
                %% DE parameters
                CR = 0.9;
                F = 0.5;
                
                %% Initialize history recording arrays
                fes_history = zeros(1, max_it*popsize);
                eta_history = zeros(1, max_it*popsize); % 存储eta值（归一化适应度）
                history_index = 1;
                
                %% Record initial point
                fes_history(history_index) = nFES;
                eta_history(history_index) = bsff / wf.power_total; % 归一化适应度
                history_index = history_index + 1;
                
                %% Main loop
                while nFES < max_it*popsize && history_index <= length(fes_history)
                    remove = [];
                    improve = [];
                    change = 0;
                    
                    %% DE/rand/1 for local search
                    newFit = zeros(lPopSize, 1);
                    newPop = zeros(lPopSize, D);
                    
                    if lPopSize > 3
                        for i = 1:lPopSize
                            Offs = zeros(subSize(i), D);
                            for j = 1:subSize(i)
                                % 随机选择三个不同的个体
                                rnd = randperm(lPopSize);
                                rnd(rnd == i) = [];
                                r1 = rnd(1);
                                r2 = rnd(2);
                                r3 = rnd(3);
                                
                                % 变异（DE/rand/1）
                                mutate = lPop(r1, :) + F * (lPop(r2, :) - lPop(r3, :));
                                
                                % 边界处理（风电场特殊约束）
                                mutate = round(mutate); % 风机位置需要是整数
                                mutate = max(mutate, 1); % 确保不小于1
                                mutate = min(mutate, wf.rows * wf.cols); % 确保不大于网格总数
                                
                                % 移除重复和禁止区域位置
                                mutate = unique(mutate, 'stable');
                                available_positions = setdiff(1:wf.rows*wf.cols, wf.NA_loc);
                                mutate = intersect(mutate, available_positions);
                                
                                % 如果位置不足，随机补充
                                if length(mutate) < D
                                    additional_positions = setdiff(available_positions, mutate);
                                    needed = D - length(mutate);
                                    if needed > 0 && length(additional_positions) >= needed
                                        additional = additional_positions(randperm(length(additional_positions), needed));
                                        mutate = [mutate, additional];
                                    elseif length(mutate) < D
                                        % 如果仍然不足，使用原有个体
                                        mutate = lPop(i, :);
                                    end
                                elseif length(mutate) > D
                                    mutate = mutate(1:D);
                                end
                                
                                mutate = sort(mutate);
                                Offs(j, :) = mutate;
                            end
                            
                            % 评估子代
                            OffsFit = zeros(subSize(i), 1);
                            for j = 1:subSize(i)
                                [fitness, ~, ~] = wf_fitness(wf, Offs(j, :));
                                OffsFit(j) = fitness;
                            end
                            
                            [newFit(i), pos] = max(OffsFit); % 最大化问题
                            newPop(i, :) = Offs(pos, :);
                            nFES = nFES + subSize(i);
                        end
                    else
                        % 小种群情况使用高斯扰动
                        for i = 1:lPopSize
                            Offs = zeros(subSize(i), D);
                            for j = 1:subSize(i)
                                % 高斯扰动
                                perturbed = lPop(i, :) + round(randn(1, D));
                                perturbed = max(perturbed, 1);
                                perturbed = min(perturbed, wf.rows * wf.cols);
                                perturbed = unique(perturbed, 'stable');
                                available_positions = setdiff(1:wf.rows*wf.cols, wf.NA_loc);
                                perturbed = intersect(perturbed, available_positions);
                                
                                if length(perturbed) < D
                                    additional_positions = setdiff(available_positions, perturbed);
                                    needed = D - length(perturbed);
                                    if needed > 0 && length(additional_positions) >= needed
                                        additional = additional_positions(randperm(length(additional_positions), needed));
                                        perturbed = [perturbed, additional];
                                    else
                                        perturbed = lPop(i, :);
                                    end
                                elseif length(perturbed) > D
                                    perturbed = perturbed(1:D);
                                end
                                
                                perturbed = sort(perturbed);
                                Offs(j, :) = perturbed;
                            end
                            
                            OffsFit = zeros(subSize(i), 1);
                            for j = 1:subSize(i)
                                [fitness, ~, ~] = wf_fitness(wf, Offs(j, :));
                                OffsFit(j) = fitness;
                            end
                            
                            [newFit(i), pos] = max(OffsFit);
                            newPop(i, :) = Offs(pos, :);
                            nFES = nFES + subSize(i);
                        end
                    end
                    
                    %% 局部搜索更新
                    for i = 1:lPopSize
                        if newFit(i) >= lPopFit(i) % 最大化问题
                            flag(i) = 0;
                            lPopFit(i) = newFit(i);
                            lPop(i, :) = newPop(i, :);
                            improve = [improve, i];
                        else
                            flag(i) = flag(i) + 1;
                            if flag(i) >= theta
                                flag(i) = 0;
                                if i ~= bsfi
                                    change = change + subSize(i);
                                    remove = [remove, i];
                                else
                                    if subSize(i) > 1
                                        subSize(i) = subSize(i) - 1;
                                        change = change + 1;
                                    end
                                end
                            end
                        end
                    end
                    
                    %% 检查是否有改进的个体
                    if ~isempty(improve)
                        if gPopSize > gTh && sum(subSize) < popSize
                            assign = improve(randperm(length(improve), 1));
                            change = change - 1;
                            subSize(assign) = subSize(assign) + 1;
                        end
                    end
                    
                    [maxSf, maxSi] = max(lPopFit); % 最大化问题
                    if maxSf > bsff
                        bsff = maxSf;
                        bsfp = lPop(maxSi(1), :);
                    end
                    
                    %% 移除表现差的个体
                    subSize(remove) = [];
                    lPop(remove, :) = [];
                    lPopFit(remove) = [];
                    flag(remove) = [];
                    lPopSize = lPopSize - length(remove);
                    
                    %% Random search for global population
                    if gPopSize > 0
                        gOffs = zeros(gPopSize, D);
                        for i = 1:gPopSize
                            if gFlag(i)
                                % 完全随机搜索
                                available_positions = setdiff(1:wf.rows*wf.cols, wf.NA_loc);
                                selected = available_positions(randperm(length(available_positions), D));
                                gOffs(i, :) = sort(selected);
                            else
                                % 高斯扰动搜索
                                perturbed = gPop(i, :) + round(gR(i) * randn(1, D));
                                perturbed = max(perturbed, 1);
                                perturbed = min(perturbed, wf.rows * wf.cols);
                                perturbed = unique(perturbed, 'stable');
                                available_positions = setdiff(1:wf.rows*wf.cols, wf.NA_loc);
                                perturbed = intersect(perturbed, available_positions);
                                
                                if length(perturbed) < D
                                    additional_positions = setdiff(available_positions, perturbed);
                                    needed = D - length(perturbed);
                                    if needed > 0 && length(additional_positions) >= needed
                                        additional = additional_positions(randperm(length(additional_positions), needed));
                                        perturbed = [perturbed, additional];
                                    else
                                        perturbed = gPop(i, :);
                                    end
                                elseif length(perturbed) > D
                                    perturbed = perturbed(1:D);
                                end
                                
                                gOffs(i, :) = sort(perturbed);
                            end
                        end
                        
                        % 评估全局种包子代
                        gOffsFit = zeros(gPopSize, 1);
                        for i = 1:gPopSize
                            [fitness, ~, ~] = wf_fitness(wf, gOffs(i, :));
                            gOffsFit(i) = fitness;
                        end
                        nFES = nFES + gPopSize;
                        
                        for i = 1:gPopSize
                            if gFlag(i) == 1
                                gFlag(i) = 0;
                                gPop(i, :) = gOffs(i, :);
                                gPopFit(i) = gOffsFit(i);
                            else
                                if gOffsFit(i) >= gPopFit(i) % 最大化问题
                                    gPopFit(i) = gOffsFit(i);
                                    gPop(i, :) = gOffs(i, :);
                                    gR(i) = 1.1 * gR(i);
                                    if gR(i) > 20
                                        gR(i) = 1;
                                    end
                                else
                                    gR(i) = 0.9 * gR(i);
                                    if gR(i) < 10e-8
                                        gR(i) = 1;
                                    end
                                end
                            end
                        end
                        
                        [sFit, sI] = max(gPopFit);
                        if sFit > bsff
                            bsff = sFit;
                            bsfp = gPop(sI(1), :);
                        end
                        
                        %% 个体迁移
                        if rand < gPopSize / popSize
                            if sum(subSize) < popSize
                                lPopSize = lPopSize + 1;
                                subSize = [subSize; 1];
                                gPopSize = gPopSize - 1;
                                lPop = [lPop; gPop(sI(1), :)];
                                lPopFit = [lPopFit; sFit];
                                flag = [flag; 0];
                                gPop(sI(1), :) = [];
                                gPopFit(sI(1)) = [];
                                gFlag(sI(1)) = [];
                                gR(sI(1)) = [];
                            end
                        end
                    end
                    
                    %% 调整全局种群大小
                    gPopSize = gPopSize + change;
                    if change >= 0
                        gFlag = [gFlag; ones(change, 1)];
                        gR = [gR; ones(change, 1)];
                        gPop = [gPop; zeros(change, D)];
                        gPopFit = [gPopFit; -inf*ones(change, 1)]; % 初始化适应度为负无穷
                    else
                        for i = 1:abs(change)
                            [~, sortIndex] = sort(gPopFit, 'descend'); % 最大化问题，降序排序
                            deleIndex = sortIndex(end);
                            gPopFit(deleIndex) = [];
                            gPop(deleIndex, :) = [];
                            gFlag(deleIndex) = [];
                            gR(deleIndex) = [];
                        end
                    end
                    
                    %% 调整全局阈值
                    gTh = popSize * exp(1 - max_it*popsize/(max_it*popsize + 1 - nFES));
                    
                    if mod(nFES, popsize) == 0 || nFES == popsize  % 每代结束时打印一次
                        iter_num = floor(nFES / popsize);
                        if mod(iter_num, 50) == 0 || iter_num == 400
                            current_eta = bsff / wf.power_total;
                            fprintf('运行 %d/%d, 迭代 %d, FES: %d/%d, 适应度: %.2f, eta: %.6f\n', run, runTime, iter_num, nFES, max_it*popsize, bsff, current_eta);
                        end
                    end
                    
                    %% 记录当前最佳归一化适应度和函数评估次数
                    if history_index <= length(fes_history)
                        fes_history(history_index) = nFES;
                        eta_history(history_index) = bsff / wf.power_total;
                        history_index = history_index + 1;
                    end
                        
                    if nFES >= max_it*popsize
                        break;
                    end
                end
                
                %% 裁剪历史记录数组
                fes_history = fes_history(1:history_index-1);
                eta_history = eta_history(1:history_index-1);
                
                %% 在标记点找到对应的eta值
                marker_eta = zeros(size(marker_points));
                for i = 1:length(marker_points)
                    [~, idx] = min(abs(fes_history - marker_points(i)));
                    marker_eta(i) = eta_history(idx);
                end
                
                %% 保存该次运行的结果
                all_run_results(:, run) = marker_eta;
                
                %% 计算最终适应度
                bsf_eta_val = bsff / wf.power_total;
                outcome = [outcome, bsf_eta_val];
                
                %% 保存和输出
                run_end_t = toc(run_st);
                cost_t = seconds(run_end_t);
                cost_t.Format = 'hh:mm:ss';
                fprintf('第 %d 次运行完成，最佳归一化适应度: %f，耗时: %s\n', run, bsf_eta_val, cost_t);
                fprintf(f, '%s - %s TN %d 第%d次运行完成，最佳eta: %f，耗时: %s\n', ...
                    algorithmDir, ws_folder, tn, run, bsf_eta_val, cost_t);
                
                %% 保存布局结果
                [best_farmlayout, best_farmlayout_NA] = gene_layout_by_indices_one(wf, bsfp);
                save_results(best_farmlayout, best_farmlayout_NA, run, folder);
            end            
           
            %% 输出统计信息
            fprintf('\n======= 统计结果 =======\n');
            fprintf('场景: %s, 风机数: %d\n', ws_folder, tn);
            fprintf('最大eta值: %f\n', min(outcome));
            fprintf('最小eta值: %f\n', max(outcome));
            fprintf('平均eta值: %f\n', mean(outcome));
            fprintf('eta值标准差: %f\n', std(outcome));
            fprintf('eta值中位数: %f\n', median(outcome));
            fprintf('========================\n\n');
            
            %% 保存统计结果
            result(1, :) = [min(outcome), max(outcome), median(outcome), mean(outcome), std(outcome)];
            stat_filename = sprintf('%s/statistics.mat', folder);
            save(stat_filename, 'result', 'outcome');
            
            %% 保存详细结果
            eta_matrix = all_run_results';
            save(sprintf('%s/eta.mat', folder), 'eta_matrix');         
        end
    end
end

%% 记录总时间
total_cost = toc(total_st);
total_cost_t = seconds(total_cost);
total_cost_t.Format = 'hh:mm:ss';
fprintf(f, '总耗时: %s\n', total_cost_t);
fclose(f);

fprintf('所有实验完成！\n');
toc;