function [Fbest, BestChart, BestFitness, farmlayout, farmlayout_NA] = HHWOA_RDE(popsize, wf, iterations, NA_type, tn, wt, t)
    % HHWOA算法结合RDE局部搜索
    % https://github.com/suya0310/Hyper-heuristic-whale-optimization-algorithm-HHWOA
    
    % 初始化记录数组
    BestChart = zeros(iterations, 1);                  % 最佳归一化适应度
    BestFitness = zeros(iterations, 1);                % 最佳实际适应度
    farmlayout = zeros(iterations, wf.rows * wf.cols); % 最佳风电场布局
    farmlayout_NA = zeros(iterations, wf.rows * wf.cols); % 最佳NA的风电场布局
    
    % 问题维度设置
    D = wf.turbine_num;        % 变量维度（风机数量）
    down = 1;                  % 下界
    up = wf.rows * wf.cols;    % 上界（网格总单元数）
    
    %% ========== 种群初始化 ==========
    [Positions, ~] = windfarm_init(popsize, wf.turbine_num, wf);
    WOA_Positions = Positions;  % 复制一份用于WOA更新
    
    %% ========== 计算初始适应度 ==========
    fitness = zeros(popsize, 1);
    for i = 1:popsize
        fitness(i) = wf_fitness(wf, Positions(i, :));
    end
    
    %% ========== 初始化最优解 ==========
    [Leader_score, leader_idx] = max(fitness);  % 最大化问题用max
    Leader_pos = Positions(leader_idx, :);
    Fbest = Leader_score;
    
    % 记录第一代结果
    BestChart(1) = Leader_score / wf.power_total;
    BestFitness(1) = Leader_score;
    [best_farmlayout, best_farmlayout_NA] = gene_layout_by_indices_one(wf, Leader_pos);
    farmlayout(1, :) = best_farmlayout;
    farmlayout_NA(1, :) = best_farmlayout_NA;
    
    %% ========== HHWOA参数初始化 ==========
    w = 3;                      % 混沌映射的阶数（Chebyshev映射参数）
    p = rand();                 % 混沌映射初始值
    Tfitness = zeros(2 * popsize, 1); % 临时存储合并种群的适应度
    
    %% ========== RDE参数初始化 ==========
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
    
    %% ========== 主循环 ==========
    for t_iter = 1:iterations
        %% ========== 混沌映射更新p值 ==========
        p = abs(cos(w * acos(p)));
        
        %% ========== 反向学习初始化 ==========
        OBLPositions = OBL_initialization(popsize, D, down, up, Positions);
        
        %% ========== 合并种群 ==========
        TPositions = zeros(2 * popsize, D);
        for i = 1:popsize
            TPositions(i, :) = Positions(i, :);
            TPositions(i + popsize, :) = OBLPositions(i, :);
        end
        
        %% ========== 评估合并种群 ==========
        for i = 1:2 * popsize
            Tfitness(i) = wf_fitness(wf, TPositions(i, :));
        end
        
        %% ========== 选择最优的popsize个个体 ==========
        [~, Tindex] = sort(Tfitness, 'descend');
        for newindex = 1:popsize
            Positions(newindex, :) = TPositions(Tindex(newindex), :);
            fitness(newindex) = Tfitness(Tindex(newindex));
        end
        
        %% ========== RDE局部搜索 ==========
        [Positions, fitness, Leader_score, Leader_pos] = RDE_LocalSearch(...
            Positions, fitness, Leader_score, Leader_pos, down, up, wf, D, ...
            t_iter, iterations, current_G, genForChange, memory_size, ...
            memory_sf, memory_cr, memory_pos, arrayDiff, arrayRate, indexLN, ...
            numViaLN, sigma_NS, u_NS, mu_NS, c, A);
        
        %% ========== 更新WOA参数 ==========
        a = 2 - t_iter * (2 / iterations);
        a2 = -1 + t_iter * ((-1) / iterations);
        
        %% ========== WOA位置更新 ==========
        for i = 1:popsize
            r1 = rand();
            r2 = rand();
            
            A_coef = 2 * a * r1 - a;
            C = 2 * r2;
            
            b = 1;
            l = (a2 - 1) * rand() + 1;
            
            for j = 1:D
                if p < 0.5
                    if abs(A_coef) >= 1
                        rand_leader_index = randi(popsize);
                        X_rand = Positions(rand_leader_index, :);
                        D_X_rand = abs(C * X_rand(j) - Positions(i, j));
                        WOA_Positions(i, j) = round(X_rand(j) - A_coef * D_X_rand);
                    else
                        D_Leader = abs(C * Leader_pos(j) - Positions(i, j));
                        WOA_Positions(i, j) = round(Leader_pos(j) - A_coef * D_Leader);
                    end
                else
                    Distance2Leader = abs(Leader_pos(j) - Positions(i, j));
                    WOA_Positions(i, j) = round(Distance2Leader * exp(b * l) .* cos(l * 2 * pi) + Leader_pos(j));
                end
            end
            
            % 边界处理和约束检查
            WOA_Positions(i, :) = max(min(WOA_Positions(i, :), up), down);
            WOA_Positions(i, :) = windfarm_constraint(WOA_Positions(i, :), wf.NA_loc, D, down, up);
            
            % 评估新位置
            new_fitness = wf_fitness(wf, WOA_Positions(i, :));
            
            % 贪婪选择
            if new_fitness > fitness(i)
                Positions(i, :) = WOA_Positions(i, :);
                fitness(i) = new_fitness;
                
                if new_fitness > Leader_score
                    Leader_score = new_fitness;
                    Leader_pos = WOA_Positions(i, :);
                    Fbest = Leader_score;
                end
            end
        end
        
        %% ========== 记录当前代结果 ==========
        BestChart(t_iter) = Leader_score / wf.power_total;
        BestFitness(t_iter) = Leader_score;
        [best_farmlayout, best_farmlayout_NA] = gene_layout_by_indices_one(wf, Leader_pos);
        farmlayout(t_iter, :) = best_farmlayout;
        farmlayout_NA(t_iter, :) = best_farmlayout_NA;
        
        %% ========== 输出进度 ==========
        if mod(t_iter, 50) == 0 || t_iter == iterations
            fprintf('HHWOA-RDE - NA %d Turbine Num:%d Wind %s run: %d iteration: %d  eta %f fitness %f\n', ...
                NA_type, tn, wt, t, t_iter, (Fbest / wf.power_total), Fbest);
        end
    end
    
    fprintf('HHWOA-RDE优化完成! 运行次数:%d, 最佳适应度: %f (归一化: %f)\n', ...
        t, Fbest, BestChart(end));
end

%% ========== 反向学习初始化函数 ==========
function OBLPositions = OBL_initialization(N, D, lb, ub, Positions)
    OBLPositions = zeros(N, D);
    for i = 1:N
        for j = 1:D
            OBLPositions(i, j) = lb + ub - Positions(i, j);
        end
    end
    OBLPositions = max(min(OBLPositions, ub), lb);
end

%% ========== RDE局部搜索函数 ==========
function [Positions, fitness, Leader_score, Leader_pos] = RDE_LocalSearch(...
    Positions, fitness, Leader_score, Leader_pos, lb, ub, wf, D, ...
    iter, max_iter, current_G, genForChange, memory_size, ...
    memory_sf, memory_cr, memory_pos, arrayDiff, arrayRate, indexLN, ...
    numViaLN, sigma_NS, u_NS, mu_NS, c, A)
    
    % RDE参数
    RDE_iter = 20;  % RDE迭代次数
    popsize = size(Positions, 1);
    
    for t = 1:RDE_iter
        % 当前种群排序（按适应度降序）
        [fitness_sorted, sorted_index] = sort(fitness, 'descend');
        X_sorted = Positions(sorted_index, :);
        
        %% ========== F和CR参数生成 ==========
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
        NS = min(max(mu_NS + sigma_NS * randn(popsize, 1), 0.1 * popsize), u_NS);
        ns_list = round(NS);
        
        %% ========== 策略分配 ==========
        [op_1, op_2, op_3, numViaLN] = vaiOP(indexLN, popsize, numViaLN);
        
        %% ========== 变异操作 ==========
        vi = zeros(popsize, D);
        
        % 建立档案集（合并当前种群）
        P_A = [X_sorted; A];
        [P_A_PS, ~] = size(P_A);
        
        % p-best参数（精英比例线性递减）
        p_best = 0.3 - 0.2 * (iter / max_iter)^1;
        pNP = max(round(p_best * P_A_PS), 2);
        
        for i = 1:popsize
            % 获取邻居索引
            ns = ns_list(i);
            index_neighbor = randperm(popsize);
            if ismember(i, index_neighbor(1:min(ns, popsize)))
                replace_idx = find(index_neighbor(1:min(ns, popsize)) == i);
                if ns + 1 <= popsize
                    index_neighbor(replace_idx) = index_neighbor(ns + 1);
                else
                    index_neighbor(replace_idx) = index_neighbor(1);
                end
            end
            
            neighbor_indices = index_neighbor(1:min(ns, popsize));
            neighbor_fitness = fitness_sorted(neighbor_indices);
            [best_value_neighbor, best_index_neighbor] = max(neighbor_fitness); 

            % 策略3：选取邻居中最优个体
            if best_value_neighbor > fitness_sorted(i)  % 邻居最优更好
                optimum_vector = X_sorted(neighbor_indices(best_index_neighbor), :);
                optimum_index = neighbor_indices(best_index_neighbor);
            else
                % 计算与邻居的欧式距离，找到最近的个体
                neighbor_pop = X_sorted(neighbor_indices, :);
                diff = repmat(X_sorted(i, :), size(neighbor_pop, 1), 1) - neighbor_pop;
                distances = sqrt(sum(diff.^2, 2));
                [~, min_dist_idx] = min(distances);
                optimum_vector = neighbor_pop(min_dist_idx, :);
                optimum_index = neighbor_indices(min_dist_idx);
            end
            
            % 策略2：p-best选择
            randindex = ceil(rand(1, P_A_PS) .* pNP) + 1;
            randindex = max(1, min(randindex, P_A_PS));
            pbX = P_A(randindex, :);
            
            % 选择父代
            parent1 = randi(popsize);
            while parent1 == i || parent1 == optimum_index
                parent1 = randi(popsize);
            end
            
            parent2 = randi(P_A_PS);
            while parent2 == i || parent2 == parent1 || parent2 == optimum_index
                parent2 = randi(P_A_PS);
            end
            
            % 根据策略进行变异
            if op_1(i)
                % 策略1：DE/rand/1
                vi(i, :) = round(X_sorted(i, :) + sf(i) .* (X_sorted(parent1, :) - P_A(parent2, :)));
            elseif op_2(i)
                % 策略2：DE/current-to-p-best/1
                vi(i, :) = round(X_sorted(i, :) + sf(i) .* (pbX(i, :) - X_sorted(i, :)) + ...
                    sf(i) .* (X_sorted(parent1, :) - P_A(parent2, :)));
            elseif op_3(i)
                % 策略3：邻域双变异策略
                vi(i, :) = round(X_sorted(i, :) + sf(i) .* (optimum_vector - X_sorted(i, :)) + ...
                    sf(i) .* (X_sorted(parent1, :) - P_A(parent2, :)));
            end
        end
        
        %% ========== 边界处理 ==========
        vi = max(min(vi, ub), lb);
        vi = windfarm_constraint(vi, wf.NA_loc, D, lb, ub);
        
        %% ========== 交叉操作 ==========
        mask = rand(popsize, D) > repmat(cr, 1, D);
        rows = (1:popsize)';
        cols = floor(rand(popsize, 1) * D) + 1;
        jrand = sub2ind([popsize, D], rows, cols);
        mask(jrand) = false;
        
        ui = vi;
        ui(mask) = X_sorted(mask);
        
        %% ========== 评估子代 ==========
        fit_U = zeros(popsize, 1);
        for i = 1:popsize
            fit_U(i) = wf_fitness(wf, ui(i, :));
        end
        
        %% ========== 更新档案集A ==========
        for i = 1:popsize
            if fit_U(i) > fitness_sorted(i)
                A = [A; X_sorted(i, :)];
            end
        end
        
        if ~isempty(A)
            [A, ~, ~] = unique(A, 'rows');
            [A_line, ~] = size(A);
            if A_line > popsize
                A_fit = zeros(A_line, 1);
                for i = 1:A_line
                    A_fit(i) = wf_fitness(wf, A(i, :));
                end
                [~, A_index] = sort(A_fit, 'descend');
                A = A(A_index(1:popsize), :);
            end
        end
        
        %% ========== 更新邻居大小参数均值 ==========
        Sns = [];
        for i = 1:popsize
            if fit_U(i) > fitness_sorted(i)
                Sns = [Sns, NS(i)];
            end
        end
        
        if ~isempty(Sns)
            meanNS = sum(Sns.^2) / (sum(Sns) + eps);
            mu_NS = (1 - c) * mu_NS + c * meanNS;
        end
        
        %% ========== 选择操作和参数更新 ==========
        I = fit_U > fitness_sorted;
        goodCR = [];
        goodF = [];
        dif_val = [];
        
        if any(I)
            goodCR = cr(I);
            goodF = sf(I);
            dif_val = abs(fit_U(I) - fitness_sorted(I));
        end
        
        %% ========== 更新策略概率 ==========
        diff2 = max(0, (fit_U - fitness_sorted)) ./ (abs(fitness_sorted) + eps);
        
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
        for i = 1:popsize
            if fit_U(i) > fitness(sorted_index(i))
                Positions(sorted_index(i), :) = ui(i, :);
                fitness(sorted_index(i)) = fit_U(i);
                
                if fit_U(i) > Leader_score
                    Leader_score = fit_U(i);
                    Leader_pos = ui(i, :);
                end
            end
        end
        
        current_G = current_G + 1;
    end
end

%% ========== 策略分配函数 ==========
function [op_1, op_2, op_3, numViaLN] = vaiOP(indexLN, popsize, numViaLN)
    % 策略分配函数
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