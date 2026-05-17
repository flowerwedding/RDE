clear all;
clc;
tic;
format long;

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
turbine_num = [30]; % 风力发电机组的规模
n_speeds = 6;
n_directions = 12;
unifrom = 0;
NA_type_list = 0;

%% 4种风电场场景wind scenarios
% n_speeds = [3, 3, 4, 6]; % 风速数量
% n_directions = [12, 12, 12, 12]; % 风向数量
% unifrom = [0, 1, 0, 0]; % 是否均匀分布

%% 算法参数
u_NS = 0.3 * popsize; % 邻居大小的上界 = 0.3 × 种群大小
c = 0.1; % 均值更新的权重值
sigma = 0.1; % 标准差
sigma_NS = 1; % 邻居大小的标准差

%% 设置标记间隔
marker_interval = 5000; % 每 5,000 FES 标记一次
marker_fes = [0:marker_interval:max_it*popsize, max_it*popsize];

%% 算法名称
algorithmDir = 'Modified_Algorithm_Wind';

total_st = tic;

text_path = sprintf('%s/%s/', save_path, algorithmDir);
text_file = sprintf('%s/cost_time.txt', text_path);
if ~exist(text_path, 'dir')
    mkdir(text_path)
end
f = fopen(text_file, 'a');
fprintf(f, datestr(now));
fprintf(f, '\n');

%% 主循环：遍历所有风场景、风机规模和NA类型
for wt = 1:length(n_speeds)
    for tn = turbine_num
        for NA_type = NA_type_list
            fprintf('\n-------------------------------------------------------\n')
            fprintf('开始处理：风场景 %d, 风机数 %d, NA类型 %d\n', wt, tn, NA_type)
            
            % 生成风电场结构和风场景数据
            NA_loc_array = gene_NA_loc(NA_type);
            [wf, ws_folder] = gene_windfram(rows, cols, tn, cell_width, NA_loc_array, ...
                n_speeds(wt), n_directions(wt), unifrom(wt));
            
            % 变量维度（风机数量）
            D = wf.turbine_num;
            
            % 初始化存储所有运行结果的数组
            all_run_results = zeros(length(marker_fes), runTime);
            outcome = zeros(1, runTime);
            
            % 创建保存目录
            folder = sprintf('%s/%s/%s/tn%d_NA%d', save_path, algorithmDir, ws_folder, tn, NA_type);
            if ~exist(folder, 'dir')
                mkdir(folder)
            end
            
            %% 对每个运行次数进行循环
            for run_id = 1:runTime
                fprintf('运行 %d/%d...\n', run_id, runTime);
                run_st = tic;
                
                %% 初始化种群
                % 风电场布局初始化：每个个体是一组风机位置的索引
                pop = zeros(popsize, D);
                for i = 1:popsize
                    available_positions = setdiff(1:wf.rows*wf.cols, wf.NA_loc);
                    selected_positions = available_positions(randperm(length(available_positions), D));
                    pop(i, :) = sort(selected_positions); % 排序便于比较
                end
                
                %% 初始化参数
                mu_NS = randi([0.1*popsize, 0.3*popsize]); % 邻居大小的均值
                mu_CR = 0.5; % 交叉概率的均值
                mu_F = 0.5; % 缩放因子的均值
                A = []; % 档案集
                nfes = 0;
                
                %% 评估初始种群
                f = zeros(popsize, 1); % 适应度（注意：风电场是最大化问题）
                for i = 1:popsize
                    [fitness, ~, ~] = wf_fitness(wf, pop(i, :));
                    f(i) = fitness;
                end
                nfes = nfes + popsize;
                
                [best_value, best_index] = max(f); % 风电场是最大化问题
                best_solution = pop(best_index, :);
                
                %% 初始化历史记录数组
                fes_history = zeros(1, max_it*popsize);
                eta_history = zeros(1, max_it*popsize); % 记录归一化适应度
                history_index = 1;
                
                % 记录初始点
                fes_history(history_index) = nfes;
                eta_history(history_index) = best_value / wf.power_total; % 归一化适应度
                history_index = history_index + 1;
                
                %% 主优化循环
                while nfes < max_it*popsize
                    Sns = []; % 成功邻居大小集合
                    Scr = []; % 成功交叉概率集合
                    Sf = []; % 成功缩放因子集合
                    
                    % 生成参数
                    NS = Rand_NS(mu_NS, sigma_NS, popsize, u_NS); % 生成邻居大小
                    F = Rand_F(mu_F, sigma, popsize); % 生成缩放因子
                    CR = Rand_CR(mu_CR, sigma, popsize); % 生成交叉概率
                    CR = sort(CR);
                    
                    % 对适应度排序，获取索引顺序
                    [~, order] = sort(f, 'descend'); % 降序排序（最大化问题）
                    
                    for i = 1:popsize
                        ns = round(NS(i)); % 四舍五入邻居大小
                        index_i = find(order == i); % 找到当前个体在排序中的位置
                        CR_i = CR(index_i); % 获取对应的交叉概率
                        
                        % 获取邻居索引
                        index_neighbor = randperm(popsize);
                        if ismember(i, index_neighbor(1:ns))
                            index_neighbor(find(index_neighbor(1:ns) == i)) = index_neighbor(ns+1);
                        end
                        
                        % 在邻居中找到最佳个体
                        neighbor_indices = index_neighbor(1:ns);
                        neighbor_fitness = f(neighbor_indices);
                        [best_value_neighbor, best_index_neighbor] = max(neighbor_fitness); % 最大化问题
                        
                        optimum_vector = pop(i, :);
                        optimum_index = i;
                        if best_value_neighbor >= f(i) % 最大化问题，使用 >=
                            optimum_vector = pop(neighbor_indices(best_index_neighbor), :);
                            optimum_index = neighbor_indices(best_index_neighbor);
                        end
                        
                        %% 构建合并种群（当前种群+档案集）
                        P_A = [pop; A];
                        [P_A_PS, ~] = size(P_A);
                        
                        %% 选择父代
                        parent1 = randi(popsize);
                        while parent1 == i || parent1 == optimum_index
                            parent1 = randi(popsize);
                        end
                        
                        parent2 = randi(P_A_PS);
                        while parent2 == i || parent2 == parent1 || parent2 == optimum_index
                            parent2 = randi(P_A_PS);
                        end
                        
                        %% DE变异操作
                        % 注意：对于风电场问题，我们不能直接进行数值加减
                        % 需要特殊处理变异操作
                        if rand < 0.5
                            % 方案1：基于位置的交叉变异
                            v = windfarm_mutation(pop(i, :), optimum_vector, ...
                                pop(parent1, :), P_A(parent2, :), F(i), wf);
                        else
                            % 方案2：随机交换位置
                            v = windfarm_random_swap(pop(i, :), wf);
                        end
                        
                        %% 二项式交叉
                        u = windfarm_crossover(pop(i, :), v, CR_i);
                        
                        %% 边界约束处理（确保是有效风机位置）
                        u = windfarm_constraint(u, wf.NA_loc, D, 1, wf.rows*wf.cols);
                        
                        %% 评估子代
                        f1 = 0;
                        try
                            [fitness, ~, ~] = wf_fitness(wf, u);
                            f1 = fitness;
                        catch
                            % 如果计算失败，使用父代适应度
                            f1 = f(i);
                        end
                        nfes = nfes + 1;
                        
                        %% 选择操作
                        if f1 > f(i) % 风电场是最大化问题
                            % 更新档案集
                            [A_line, ~] = size(A);
                            if A_line == popsize
                                % 随机替换
                                rand_line = randi(popsize);
                                A(rand_line, :) = pop(i, :);
                            else
                                % 直接添加
                                A = [A; pop(i, :)];
                            end
                            
                            % 更新个体
                            f(i) = f1;
                            pop(i, :) = u;
                            
                            % 记录成功参数
                            Sns = [Sns, NS(i)];
                            Scr = [Scr, CR_i];
                            Sf = [Sf, F(i)];
                            
                            % 更新全局最优
                            if f1 > best_value
                                best_value = f1;
                                best_solution = u;
                            end
                        end
                        
                        %% 记录当前最佳归一化适应度
                        if history_index <= length(fes_history)
                            current_eta = best_value / wf.power_total;
                            fes_history(history_index) = nfes;
                            eta_history(history_index) = current_eta;
                            history_index = history_index + 1;
                        end
                       
                        %% 显示进度
                        if mod(nfes, 100) == 0
                            iter_num = floor(nfes / popsize);
                            if mod(iter_num, 50) == 0 || iter_num == 400
                                current_eta = best_value / wf.power_total;
                                fprintf('  运行 %d, 迭代 %d, FES: %d/%d, 当前适应度: %.2f, eta: %.6f\n', ...
                                    run_id, iter_num, nfes, max_it*popsize, best_value, current_eta);
                            end
                        end
                       
                        if nfes >= max_it*popsize
                            break;
                        end
                    end
                    
                    %% 更新参数均值
                    [~, column] = size(Scr);
                    if column ~= 0
                        meanA = sum(Scr) / column;
                        meanL = sum(Sf.^2) / sum(Sf);
                        meanNS = sum(Sns.^2) / sum(Sns);
                    else
                        meanA = 0;
                        meanL = 0;
                        meanNS = 0;
                    end
                    
                    % 使用指数平滑更新参数均值
                    mu_CR = (1 - c) * mu_CR + c * meanA;
                    mu_F = (1 - c) * mu_F + c * meanL;
                    mu_NS = (1 - c) * mu_NS + c * meanNS;
                    
                    if nfes >= max_it*popsize
                        break;
                    end
                end
                
                %% 裁剪历史记录数组
                fes_history = fes_history(1:history_index-1);
                eta_history = eta_history(1:history_index-1);
                
                %% 在标记点找到对应的eta值
                marker_etas = zeros(size(marker_fes));
                for i = 1:length(marker_fes)
                    [~, idx] = min(abs(fes_history - marker_fes(i)));
                    marker_etas(i) = eta_history(idx);
                end
                
                %% 保存该次运行的结果
                all_run_results(:, run_id) = marker_etas;
                outcome(run_id) = best_value / wf.power_total;
                
                %% 保存和输出
                run_end_t = toc(run_st);
                cost_t = seconds(run_end_t);
                cost_t.Format = 'hh:mm:ss';
                
                fprintf('第 %d 次运行完成, 最佳适应度: %.2f, eta: %.6f, 耗时: %s\n', run_id, best_value, outcome(run_id), cost_t);
                
                % 保存布局结果
                [best_farmlayout, best_farmlayout_NA] = gene_layout_by_indices_one(wf, best_solution);
                save_results(best_farmlayout, best_farmlayout_NA, run_id, folder);
                
            end
            
            %% 输出统计信息
            fprintf('\n======= 统计结果 =======\n');
            fprintf('场景: %s, 风机数: %d\n', ws_folder, tn);
            fprintf('最小eta值: %f\n', min(outcome));
            fprintf('最大eta值: %f\n', max(outcome));
            fprintf('中位数eta值: %f\n', median(outcome));
            fprintf('平均eta值: %f\n', mean(outcome));
            fprintf('eta值标准差: %f\n', std(outcome));
            fprintf('========================\n\n');
            
            %% 保存统计结果
            result_stats = [min(outcome), max(outcome), median(outcome), mean(outcome), std(outcome)];
            stat_filename = sprintf('%s/statistics.mat', folder);
            save(stat_filename, 'result_stats', 'outcome');
            
            %% 保存eta矩阵
            eta_matrix = all_run_results';
            save(sprintf('%s/eta.mat', folder), 'eta_matrix');
            
            %% 保存详细结果
            detailed_results = [marker_fes', all_run_results, mean(all_run_results, 2), median(all_run_results, 2)];
            detailed_filename = sprintf('%s/detailed_results.txt', folder);
            save(detailed_filename, 'detailed_results', '-ascii');
        end
    end
end

%% 记录总时间
total_cost = toc(total_st);
total_cost_t = seconds(total_cost);
total_cost_t.Format = 'hh:mm:ss';
fprintf('总耗时: %s\n', total_cost_t);
fclose(f);

fprintf('所有实验完成！\n');
toc;

%% ========== 辅助函数 ==========

function NS = Rand_NS(mu, sigma, PS, u_NS)
    % 生成服从截断正态分布的邻居大小
    NS = mu + sigma * randn(PS, 1);
    NS = max(NS, 0.1*PS);
    NS = min(NS, u_NS);
end

function F = Rand_F(mu, sigma, PS)
    % 生成缩放因子
    F = mu + sigma * randn(PS, 1);
    F = max(F, 0.1);
    F = min(F, 1.0);
end

function CR = Rand_CR(mu, sigma, PS)
    % 生成交叉概率
    CR = mu + sigma * randn(PS, 1);
    CR = max(CR, 0);
    CR = min(CR, 1);
end

function v = windfarm_mutation(current, optimum, parent1, parent2, F, wf)
    % 风电场变异操作：基于位置的交叉
    D = length(current);
    v = current; % 初始化为当前个体
    
    % 随机选择一些位置进行替换
    num_replace = round(F * D);
    if num_replace < 1
        num_replace = 1;
    elseif num_replace > D
        num_replace = D;
    end
    
    % 从其他个体中随机选择位置
    all_sources = [optimum, parent1, parent2];
    for i = 1:num_replace
        source_idx = randi(3); % 随机选择一个源个体
        pos_idx = randi(D); % 随机选择一个位置索引
        
        % 获取新位置
        new_pos = all_sources((source_idx-1)*D + pos_idx);
        
        % 确保新位置有效且不重复
        if ~ismember(new_pos, v) && ~ismember(new_pos, wf.NA_loc)
            % 随机替换一个现有位置
            replace_idx = randi(D);
            v(replace_idx) = new_pos;
        end
    end
    
    % 排序
    v = sort(v);
end

function v = windfarm_random_swap(current, wf)
    % 随机交换变异
    D = length(current);
    v = current;
    
    % 随机选择两个位置交换
    idx1 = randi(D);
    idx2 = randi(D);
    while idx2 == idx1
        idx2 = randi(D);
    end
    
    % 交换位置
    temp = v(idx1);
    v(idx1) = v(idx2);
    v(idx2) = temp;
    
    % 排序
    v = sort(v);
end

function u = windfarm_crossover(parent, mutant, CR)
    % 风电场交叉操作：基于概率选择位置
    D = length(parent);
    u = parent; % 初始化为父代
    
    % 对每个位置进行交叉
    for j = 1:D
        if rand < CR || j == randi(D) % 确保至少有一个维度交叉
            u(j) = mutant(j);
        end
    end
    
    % 处理重复位置
    u = unique(u, 'stable');
    
    % 如果长度不足，补充随机位置
    if length(u) < D
        available_positions = setdiff(1:max(parent), [u, parent]);
        needed = D - length(u);
        if length(available_positions) >= needed
            additional = available_positions(randperm(length(available_positions), needed));
            u = [u, additional];
        else
            % 如果仍然不足，使用父代位置
            u = parent;
        end
    end
    
    % 排序
    u = sort(u);
end