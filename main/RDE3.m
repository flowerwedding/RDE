function [Fbest, BestChart, BestFitness, farmlayout, farmlayout_NA] = RDE2(popsize, wf, iterations, NA_type, tn, wt, t)
% 初始化记录数组 消融实验 没有多变异策略
BestChart = zeros(iterations, 1);                  % 最佳归一化适应度
BestFitness = zeros(iterations, 1);               % 最佳实际适应度
farmlayout = zeros(iterations, wf.rows * wf.cols);       % 最佳风电场布局
farmlayout_NA = zeros(iterations, wf.rows * wf.cols);    % 最佳NA的风电场布局

% 风机数量、变量上下界
D = wf.turbine_num;        % 变量维度（风机数量）
down = 1;                  % 下界
up = wf.rows * wf.cols;    % 上界（网格总单元数）

%% ========== RDE参数初始化 ==========
% 种群初始化
[popu, lu] = windfarm_init(popsize, wf.turbine_num, wf);

% 计算适应度
[popuFitness, power_order, lp_power_accum] = wf_fitness(wf, popu);
optimal = max(popuFitness);  % 记录当前最佳适应度

% F和CR参数
genForChange = 50;              % 策略更新频率
memory_size = 5;                % 记忆库大小
memory_sf = 0.5 .* ones(memory_size, 1);    % 缩放因子记忆
memory_cr = 0.5 .* ones(memory_size, 1);    % 交叉概率记忆
memory_pos = 1;                 % 记忆库当前位置

% 策略参数
n_opr = 3;                      % 操作符数量
arrayDiff = 0.1 * ones(1, 3);   % 改进量统计
arrayRate = 0.1 * ones(1, 3);   % 成功率统计
indexLN = [1, 1, 1];            % 策略排序
numViaLN = [1, 1, 1];           % 策略使用次数
current_G = 1;                  % 当前代数

% 邻域参数
sigma = 0.1;                    % 标准差
sigma_NS = 1;                   % 邻居大小的标准差
u_NS = 0.3 * popsize;           % 邻居大小的上界
mu_NS = randi([0.1 * popsize, 0.3 * popsize]); % 邻居大小的均值
c = 0.1;                        % 均值更新权重
A = [];                         % 档案集
parent1_list = zeros(popsize);
parent2_list = zeros(popsize * 2);

% 神经网络参数
netF = [];                      % F值预测神经网络
netCr = [];                     % CR值预测神经网络
ArchdataF = [];                 % 存储F参数相关数据的归档
ArchdataCr = [];                % 存储CR参数相关数据的归档
Archnum=0;
Arch_size= popsize;
pl = 0.5;
LearnFlag = 0;                  % 神经网络训练标志
LearnuseFlag = 0;               % 神经网络使用标志
trigger_points = round(iterations * [0.2  0.4 0.6 0.8]); % 在20%、40%、60%、80%处触发
trigger_points = round(iterations * 0.5); % 只在50%处触发一次 TODO
trigger_points = round(iterations + 1);% 不触发神经网络的训练

% 记录
jilu.rank = zeros(ceil(iterations/genForChange)+10, 3);
jilu.probs = zeros(ceil(iterations/genForChange)+10, 3);
jilu.G = zeros(ceil(iterations/genForChange)+10, 1);
jilu.rank(1, :) = indexLN;
jilu.probs(1, :) = numViaLN / sum(numViaLN);
jilu.G(1) = current_G;

%% ========== RDE核心算法 ==========
% 初始化
pBest = popu;                          % 个体历史最佳位置
pBestFit = popuFitness;                % 个体历史最佳适应度
[~, gBestId] = max(pBestFit);          % 全局最佳个体索引
gBest = pBest(gBestId, :);             % 全局最佳位置
gBest_fitness = pBestFit(gBestId);     % 全局最佳适应度
Fbest = gBest_fitness;                 % 当前最佳适应度

% 记录第一代结果
BestChart(1) = gBest_fitness / wf.power_total;
BestFitness(1) = gBest_fitness;
[best_farmlayout, best_farmlayout_NA] = gene_layout_by_indices_one(wf, gBest);
farmlayout(1, :) = best_farmlayout;
farmlayout_NA(1, :) = best_farmlayout_NA;

%% ========== 主循环 ==========
for iter = 1:iterations
    current_G = current_G + 1;
    
    % 当前种群排序
    [popuFitness_sorted, sorted_index] = sort(popuFitness, 'descend');
    popu_sorted = popu(sorted_index, :);
    
    X = popu_sorted;                % 当前种群（排序后）
    val_X = popuFitness_sorted;     % 当前适应度（排序后）
    
    %% ========== F和CR参数 ==========
    % 从记忆库选择参数基值
    mem_rand_index = ceil(memory_size * rand(popsize, 1));
    mu_sf = memory_sf(mem_rand_index);
    mu_cr = memory_cr(mem_rand_index);
    
    % 生成交叉概率CR（正态分布）
    cr = mu_cr + 0.1 * randn(popsize, 1);
    term_pos = find(mu_cr == -1);
    cr(term_pos) = 0;
    cr = min(cr, 1);
    cr = max(cr, 0);
    
    % 生成缩放因子F（柯西分布）
    sf = mu_sf + 0.1 * tan(pi * (rand(popsize, 1) - 0.5));
    pos = find(sf <= 0);
    while ~isempty(pos)
        sf(pos) = mu_sf(pos) + 0.1 * tan(pi * (rand(length(pos), 1) - 0.5));
        pos = find(sf <= 0);
    end
    sf = min(sf, 1);
    
    %% ========== 邻居大小生成 ==========
    % 生成服从截断正态分布的邻居大小
    NS = min(max(mu_NS + sigma_NS * randn(popsize, 1), 0.1 * popsize), u_NS);
    ns_list = round(NS);
    
    %% ========== 变异 ==========  
    
    vi = zeros(popsize, D); % 为每个个体生成变异向量
            
    for i = 1:popsize
         % 获取当前个体的邻居大小
         ns = ns_list(i);
                
         % 获取邻居索引
         index_neighbor = randperm(popsize);% 获取种群随机排列的索引
         if ismember(i, index_neighbor(1:ns))% 确保自己不在前ns个邻居中
             replace_idx = find(index_neighbor(1:ns) == i);
             index_neighbor(replace_idx) = index_neighbor(ns + 1);
         end
                
         neighbor_indices = index_neighbor(1:ns);
         neighbor_fitness = val_X(neighbor_indices);
         [best_value_neighbor, best_index_neighbor] = min(neighbor_fitness);

         % 策略3：选取邻居中最优个体
         if best_value_neighbor <= val_X(i)
             % 如果邻居中的最佳个体比当前个体好，选择最优的个体
              optimum_vector = X(neighbor_indices(best_index_neighbor), :);
              optimum_index = neighbor_indices(best_index_neighbor);
         else
              % 计算与邻居的欧式距离，找到最近的个体
              neighbor_pop = X(neighbor_indices, :);
              diff = repmat(X(i, :), ns, 1) - neighbor_pop;
              distances = sqrt(sum(diff.^2, 2));
              [~, min_dist_idx] = min(distances);
              optimum_vector = neighbor_pop(min_dist_idx, :);
              optimum_index = neighbor_indices(min_dist_idx);
         end
                
         %% 建立合并种群（当前种群+档案集）
         P_A = [X; A];
         [P_A_PS, ~] = size(P_A);

         % 将种群分为优秀和较差两部分
         N_half = round(P_A_PS / 2);
         goodX = P_A(1:N_half, :);
         worstX = P_A(N_half:end, :);
                
         % 重新组织种群
         a = randperm(N_half);
         b = randperm(N_half);
         index2 = floor(0.5 * N_half);
                
         gX = goodX(a, :);
         gx1 = gX(1:index2, :);
         gx2 = gX(index2 + 1:N_half, :);
                
         wX = worstX(b, :);
         wx1 = wX(1:N_half - index2, :);
         wx2 = wX(N_half - index2 + 1:N_half, :);
                
         P_A = [gx1; wx1; gx2; wx2];

         % 策略2：选取前p=0.11的最优个体
         p = 0.3 - 0.2*(iter/iterations)^1;% 精英比例p线性递减
         pNP = max(round(p * P_A_PS), 2);
         randindex = ceil(rand(1, P_A_PS) .* pNP) + 1; 
         randindex = max(1, randindex); 
         pbX = P_A(randindex, :);      
                
         % 选择父代
         parent1 = randi(popsize);
         while parent1 == i || parent1 == optimum_index
             parent1 = randi(popsize);
         end
         parent1_list(i) = parent1;
                
         parent2 = randi(P_A_PS);
         while parent2 == i || parent2 == parent1 || parent2 == optimum_index
             parent2 = randi(P_A_PS);
         end
         parent2_list(i) = parent2;
                
         
         % 策略1：DE/rand/1 普通差分变异
         % 当前个体 + F×(父代1-父代2)
         % vi(i, :) = X(i, :) + sf(i) .* (X(parent1, :) - P_A(parent2, :));
             

         % 策略3：邻域双变异策略-最优或最近邻居
         % 当前个体 + F×(opt-当前) + F×(父代1-父代2)
         vi(i, :) = X(i, :) + sf(i) .* (optimum_vector - X(i, :)) + sf(i) .* (X(parent1, :) - P_A(parent2, :));     
          
    end
    
    %% ========== 边界处理 ==========
    vi = windfarm_constraint(vi, wf.NA_loc, D, down, up);
    
    %% ========== 交叉操作 ==========
    mask = rand(popsize, D) > repmat(cr, 1, D);
    rows = (1:popsize)';
    cols = floor(rand(popsize, 1) * D) + 1;
    jrand = sub2ind([popsize, D], rows, cols);
    mask(jrand) = false;
    
    ui = vi;
    ui(mask) = X(mask);
    
    %% ========== 评估子代 ==========
    [fit_U, power_order, lp_power_accum] = wf_fitness(wf, ui);
    
    % 更新全局最佳
    [max_fit_U, max_idx] = max(fit_U);
    if max_fit_U > gBest_fitness
        gBest = ui(max_idx, :);
        gBest_fitness = max_fit_U;
        Fbest = gBest_fitness;
    end
    
    %% ========== 更新档案集A ==========
    for i = 1:popsize    
        if fit_U(i) > val_X(sorted_index(i))
            A = [A; X(i, :)];
        end
    end
    [A, ~, ~] = unique(A, 'rows'); % 去重
    [A_line, ~] = size(A);
    if A_line > popsize % 最差替换
        [A_fit, ~, ~] = wf_fitness(wf, A);
        [~, A_index] = sort(A_fit, 'descend');
        A = A(A_index(1:popsize), :);
    end
    
    %% ========== 更新邻居大小参数均值 ==========
    Sns = [];
    for i = 1:popsize
        if fit_U(i) > val_X(sorted_index(i))
            Sns = [Sns, NS(i)];
        end
    end
    
    [~, column] = size(Sns);
    if column ~= 0
        meanNS = sum(Sns.^2) / (sum(Sns) + eps);
        mu_NS = (1 - c) * mu_NS + c * meanNS;
    end
    
    %% ========== 选择操作 ==========
    I = (fit_U > val_X);  % 子代比父代好的个体（最大化问题）
    
    % 收集成功参数
    goodCR = [];
    goodF = [];
    dif_val = [];
    
    if any(I)
        goodCR = cr(I);
        goodF = sf(I);
        dif_val = abs(fit_U(I) - val_X(I));
        ArchdataF = [ArchdataF; [X(I,:), X(parent1_list(I), :), P_A(parent2_list(I),:), sf(I)]];
        ArchdataCr = [ArchdataCr; [ui(I,:), cr(I)]];
    end
    if size(ArchdataF, 1) > Arch_size% 限制归档大小  
        ArchdataF = ArchdataF(end-Arch_size +1:end, :);  
        ArchdataCr = ArchdataCr(end-Arch_size +1:end, :);  
        Archnum=Archnum+1;
    end

    
    %% ========== 更新参数记忆库 ==========
    num_success_params = numel(goodCR);
    if num_success_params > 0
        sum_dif = sum(dif_val);
        if sum_dif > 0
            dif_val = dif_val / sum_dif;
            
            % 更新缩放因子记忆
            memory_sf(memory_pos) = (dif_val' * (goodF .^ 2)) / (dif_val' * goodF + eps);
            
            % 更新交叉概率记忆
            if max(goodCR) == 0
                memory_cr(memory_pos) = -1;
            else
                memory_cr(memory_pos) = (dif_val' * (goodCR .^ 2)) / (dif_val' * goodCR + eps);
            end
            
            memory_pos = memory_pos + 1;
            if memory_pos > memory_size
                memory_pos = 1;
            end
        end
    else
        memory_cr(memory_pos) = 0.5;
        memory_sf(memory_pos) = 0.5;
    end
    
    
    %% ========== 更新种群 ==========
    % 更优的个体替换原有个体（最大化问题）
    for i = 1:popsize
        if fit_U(i) > popuFitness(sorted_index(i))% 精英一对一替换，不用去重
            popu(sorted_index(i), :) = ui(i, :);
            popuFitness(sorted_index(i)) = fit_U(i);
        end
    end
    
    %% ========== 更新最佳值 ==========
    current_best = max(popuFitness);
    if current_best > optimal
        optimal = current_best;
        [~, new_best_idx] = max(popuFitness);
        gBest = popu(new_best_idx, :);
        gBest_fitness = current_best;
    end
    
    %% ========== 记录当前代结果 ==========
    Fbest = gBest_fitness;
    BestChart(iter) = gBest_fitness / wf.power_total;
    BestFitness(iter) = gBest_fitness;
    [best_farmlayout, best_farmlayout_NA] = gene_layout_by_indices_one(wf, gBest);
    farmlayout(iter, :) = best_farmlayout;
    farmlayout_NA(iter, :) = best_farmlayout_NA;
    
    %% ========== 输出进度 ==========
    if mod(iter, 50) == 0 || iter == iterations
        fprintf('NA %d Turbine Num:%d Wind %s run: %d iteration: %d  eta %f fitness %f\n', ...
            NA_type, tn, wt, t, iter, (Fbest / wf.power_total), Fbest);
    end
end

fprintf('RDE优化完成! 运行次数:%d, 最佳适应度: %f (归一化: %f)\n', ...
    t, Fbest, BestChart(end));
end
