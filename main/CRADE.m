function [Fbest, BestChart, BestFitness, farmlayout, farmlayout_NA] = CRADE(popsize, wf, iterations, NA_type, tn, wt, t)
% CRADE - 基于CRADE算法的风电场布局优化，与RDE接口完全兼容
% 修正版本 - 修复索引维度错误

% 初始化记录数组（与RDE完全相同）
BestChart = zeros(iterations, 1);                  % 最佳归一化适应度
BestFitness = zeros(iterations, 1);               % 最佳实际适应度
farmlayout = zeros(iterations, wf.rows * wf.cols);       % 最佳风电场布局
farmlayout_NA = zeros(iterations, wf.rows * wf.cols);    % 最佳NA的风电场布局

% 风机数量、变量上下界（与RDE相同）
D = wf.turbine_num;        % 变量维度（风机数量）
down = 1;                  % 下界
up = wf.rows * wf.cols;    % 上界（网格总单元数）

%% ========== CRADE参数初始化 ==========
% 种群初始化（使用RDE的函数）
[popu, lu] = windfarm_init(popsize, wf.turbine_num, wf);

% 计算适应度（使用RDE的函数）
[popuFitness, power_order, lp_power_accum] = wf_fitness(wf, popu);
optimal = max(popuFitness);  % 记录当前最佳适应度

% CRADE参数设置
genForChange = 50;              % 策略更新频率
memory_size = 5;                % 记忆库大小
memory_sf = 0.5 .* ones(memory_size, 1);    % 缩放因子记忆
memory_cr = 0.5 .* ones(memory_size, 1);    % 交叉概率记忆
memory_pos = 1;                 % 记忆库当前位置

% 策略相关参数
n_opr = 3;                      % 操作符数量
arrayDiff = 0.1 * ones(1, 3);   % 改进量统计
arrayRate = 0.1 * ones(1, 3);   % 成功率统计
indexLN = [1, 1, 1];            % 策略排序
numViaLN = [1, 1, 1];           % 策略使用次数
current_G = 1;                  % 当前代数

% 策略记录（与RDE中的jilu类似）
jilu.rank = zeros(ceil(iterations/genForChange)+10, 3);
jilu.probs = zeros(ceil(iterations/genForChange)+10, 3);
jilu.G = zeros(ceil(iterations/genForChange)+10, 1);
jilu.rank(1, :) = indexLN;
jilu.probs(1, :) = numViaLN / sum(numViaLN);
jilu.G(1) = current_G;

%% ========== CRADE核心算法 ==========
% 初始化最佳值（与RDE相同）
pBest = popu;                          % 个体历史最佳位置
pBestFit = popuFitness;                % 个体历史最佳适应度
[~, gBestId] = max(pBestFit);          % 全局最佳个体索引
gBest = pBest(gBestId, :);             % 全局最佳位置
gBest_fitness = pBestFit(gBestId);     % 全局最佳适应度
Fbest = gBest_fitness;                 % 当前最佳适应度

% 记录第一代结果（与RDE相同）
BestChart(1) = gBest_fitness / wf.power_total;
BestFitness(1) = gBest_fitness;
[best_farmlayout, best_farmlayout_NA] = gene_layout_by_indices_one(wf, gBest);
farmlayout(1, :) = best_farmlayout;
farmlayout_NA(1, :) = best_farmlayout_NA;

fprintf('CRADE算法初始化完成 - 风机数: %d, 网格: %dx%d\n', tn, wf.rows, wf.cols);

%% ========== 主循环 ==========
for iter = 1:iterations
    current_G = current_G + 1;
    
    % 当前种群排序（与RDE相同方式）
    [popuFitness_sorted, sorted_index] = sort(popuFitness, 'descend');
    popu_sorted = popu(sorted_index, :);
    
    X = popu_sorted;                % 当前种群（排序后）
    val_X = popuFitness_sorted;     % 当前适应度（排序后）
    
    %% ========== 参数生成 ==========
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
    
    %% ========== CRADE变异策略 ==========
    % 生成随机索引
    R = Gen_R_CRADE(popsize, 3);
    r1 = R(:, 2); r2 = R(:, 3); r3 = R(:, 4);
    
    % 策略分配
    [op_1, op_2, op_3, numViaLN] = vaiOP_CRADE(indexLN, popsize, numViaLN);
    
    % 初始化变异向量
    vi = zeros(popsize, D);
    
    % 策略1: DE/current-to-rand/1
    if any(op_1)
        srand = rand(popsize, 1);
        idx_op1 = find(op_1 == 1);
        if ~isempty(idx_op1)
            vi(idx_op1, :) = X(idx_op1, :) + ...
                srand(idx_op1, ones(1, D)) .* (X(r1(idx_op1), :) - X(idx_op1, :)) + ...
                sf(idx_op1, ones(1, D)) .* (X(r2(idx_op1), :) - X(r3(idx_op1), :));
        end
    end
    
    % 策略2: DE/current-to-p-best/1
    if any(op_2)
        pNP = max(round(0.11 * popsize), 2);
        randindex = ceil(rand(1, popsize) .* pNP) + 1;
        randindex = max(1, randindex);
        pbX = X(randindex, :);
        
        idx_op2 = find(op_2 == 1);
        if ~isempty(idx_op2)
            vi(idx_op2, :) = X(idx_op2, :) + ...
                sf(idx_op2, ones(1, D)) .* (pbX(idx_op2, :) - X(idx_op2, :) + ...
                X(r1(idx_op2), :) - X(r2(idx_op2), :));
        end
    end
    
    % 策略3: DE/rand/1
    if any(op_3)
        idx_op3 = find(op_3 == 1);
        if ~isempty(idx_op3)
            vi(idx_op3, :) = X(r1(idx_op3), :) + ...
                sf(idx_op3, ones(1, D)) .* (X(r2(idx_op3), :) - X(r3(idx_op3), :));
        end
    end
    
    %% ========== 边界约束处理 ==========
    % 使用RDE的边界处理函数
    vi = windfarm_constraint(vi, wf.NA_loc, D, down, up);
    
    %% ========== 交叉操作 ==========
    mask = rand(popsize, D) > cr(:, ones(1, D));
    rows = (1:popsize)';
    cols = floor(rand(popsize, 1) * D) + 1;
    jrand = sub2ind([popsize, D], rows, cols);
    mask(jrand) = false;
    
    ui = vi;
    ui(mask) = X(mask);
    
    %% ========== 评估子代 ==========
    % 使用RDE的适应度函数
    [fit_U, power_order, lp_power_accum] = wf_fitness(wf, ui);
    
    % 更新全局最佳（与RDE逻辑相同）
    [max_fit_U, max_idx] = max(fit_U);
    if max_fit_U > gBest_fitness
        gBest = ui(max_idx, :);
        gBest_fitness = max_fit_U;
        Fbest = gBest_fitness;
    end
    
    %% ========== 选择操作 ==========
    % 修正：确保维度匹配
    I = (fit_U > val_X);  % 子代更好的个体（都是列向量）
    
    % 收集成功参数 - 修复维度问题
    goodCR = [];
    goodF = [];
    dif_val = [];
    
    if any(I)
        goodCR = cr(I);
        goodF = sf(I);
        dif_val = abs(fit_U(I) - val_X(I));
    end
    
    %% ========== 更新策略概率 ==========
    % 确保diff2是列向量
    diff2 = max(0, (fit_U - val_X)) ./ max(1e-10, abs(val_X));
    
    % 分别计算各策略的改进量
    if any(op_1)
        op1_indices = find(op_1);
        valid_op1 = op1_indices(1:min(length(op1_indices), length(diff2)));
        if ~isempty(valid_op1)
            arrayDiff(1) = arrayDiff(1) + max(0, mean(diff2(valid_op1)));
        end
    end
    
    if any(op_2)
        op2_indices = find(op_2);
        valid_op2 = op2_indices(1:min(length(op2_indices), length(diff2)));
        if ~isempty(valid_op2)
            arrayDiff(2) = arrayDiff(2) + max(0, mean(diff2(valid_op2)));
        end
    end
    
    if any(op_3)
        op3_indices = find(op_3);
        valid_op3 = op3_indices(1:min(length(op3_indices), length(diff2)));
        if ~isempty(valid_op3)
            arrayDiff(3) = arrayDiff(3) + max(0, mean(diff2(valid_op3)));
        end
    end
    
    % 计算成功率
    if any(op_1)
        arrayRate(1) = arrayRate(1) + sum(I(op_1)) / max(1, sum(op_1));
    end
    if any(op_2)
        arrayRate(2) = arrayRate(2) + sum(I(op_2)) / max(1, sum(op_2));
    end
    if any(op_3)
        arrayRate(3) = arrayRate(3) + sum(I(op_3)) / max(1, sum(op_3));
    end
    
    % 定期更新策略选择
    if mod(current_G, genForChange) == 0
        countDiff = arrayDiff;
        countProb1 = countDiff ./ (sum(countDiff) + eps);
        countRate = arrayRate;
        countProb2 = countRate ./ (sum(countRate) + eps);
        countProbs = 0.5 * countProb1 + 0.5 * countProb2;
        [~, indexLN] = sort(countProbs, 'descend');
        
        % 重置统计
        arrayDiff = 0.1 * ones(1, 3);
        arrayRate = 0.1 * ones(1, 3);
        
        % 记录
        jilu_idx = floor(current_G/genForChange) + 1;
        if jilu_idx <= size(jilu.rank, 1)
            jilu.rank(jilu_idx, :) = indexLN;
            jilu.probs(jilu_idx, :) = numViaLN / (sum(numViaLN) + eps);
            jilu.G(jilu_idx) = current_G;
        end
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
    
    %% ========== 更新种群（类似RDE的选择机制） ==========
    % 更优的个体替换原有个体
    for i = 1:popsize
        if fit_U(i) > popuFitness(sorted_index(i))
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
    
    %% ========== 记录当前代结果（与RDE完全相同） ==========
    Fbest = gBest_fitness;
    BestChart(iter) = gBest_fitness / wf.power_total;
    BestFitness(iter) = gBest_fitness;
    [best_farmlayout, best_farmlayout_NA] = gene_layout_by_indices_one(wf, gBest);
    farmlayout(iter, :) = best_farmlayout;
    farmlayout_NA(iter, :) = best_farmlayout_NA;
    
    %% ========== 显示进度（使用RDE的输出格式） ==========
    if mod(iter, 50) == 0 || iter == iterations
        fprintf('NA %d Turbine Num:%d Wind %s run: %d iteration: %d  eta %f fitness %f\n', ...
            NA_type, tn, wt, t, iter, (Fbest / wf.power_total), Fbest);
    end
end

fprintf('CRADE优化完成! 风机数:%d, 最佳适应度: %f (归一化: %f)\n', ...
    tn, Fbest, BestChart(end));
end

%% ========== CRADE辅助函数 ==========

function R = Gen_R_CRADE(popsize, num_indices)
% 生成随机索引矩阵（替代原始CRADE中的Gen_R）
% 输入: popsize - 种群大小, num_indices - 需要的索引数量
% 输出: R - popsize x (num_indices+1) 的随机索引矩阵

R = zeros(popsize, num_indices + 1);
for i = 1:popsize
    R(i, :) = randperm(popsize, num_indices + 1);
end
end

function [op_1, op_2, op_3, numViaLN] = vaiOP_CRADE(indexLN, popsize, numViaLN)
% 策略分配函数（替代原始CRADE中的vaiOP）
% 输入: indexLN - 策略排序, popsize - 种群大小, numViaLN - 策略使用计数
% 输出: op_1, op_2, op_3 - 策略分配标志, numViaLN - 更新后的使用计数

% 初始化输出
op_1 = false(popsize, 1);
op_2 = false(popsize, 1);
op_3 = false(popsize, 1);

% 策略分配比例（基于排序动态调整）
if indexLN(1) == 1
    ratio1 = 0.5; ratio2 = 0.3; ratio3 = 0.2;
elseif indexLN(1) == 2
    ratio1 = 0.3; ratio2 = 0.5; ratio3 = 0.2;
else
    ratio1 = 0.3; ratio2 = 0.2; ratio3 = 0.5;
end

% 计算各策略分配的个体数
num_op1 = round(popsize * ratio1);
num_op2 = round(popsize * ratio2);
num_op3 = popsize - num_op1 - num_op2;

% 确保num_op3不为负
if num_op3 < 0
    num_op3 = 0;
    num_op2 = popsize - num_op1;
end

% 随机分配个体到策略
all_indices = randperm(popsize);
if num_op1 > 0
    op_1(all_indices(1:num_op1)) = true;
end
if num_op2 > 0 && num_op1+num_op2 <= popsize
    op_2(all_indices(num_op1+1:num_op1+num_op2)) = true;
end
if num_op3 > 0 && num_op1+num_op2+num_op3 <= popsize
    op_3(all_indices(num_op1+num_op2+1:end)) = true;
end

% 更新策略使用计数
numViaLN(1) = numViaLN(1) + sum(op_1);
numViaLN(2) = numViaLN(2) + sum(op_2);
numViaLN(3) = numViaLN(3) + sum(op_3);
end
% NA 0 Turbine Num:30 Wind 3speed_12direction run: 1 iteration: 400  eta 0.984299 fitness 7909.162346