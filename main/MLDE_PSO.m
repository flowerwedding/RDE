function [Fbest,BestChart,BestFitness,farmlayout,farmlayout_NA] = MLDE_PSO(popsize,wf,iterations,NA_type,tn,wt,t)
    % popsize: 种群大小, wf: 风电场结构体, iterations: 最大迭代次数, NA_type: 禁止区域类型, tn: 风机数量, wt: 风况标识, t: 运行次数索引
    
    % 初始化记录数组
    BestChart = zeros(iterations, 1); % 最佳归一化适应度
    BestFitness = zeros(iterations, 1); % 最佳实际适应度
    farmlayout = zeros(iterations, wf.rows * wf.cols); % 最佳风电场布局
    farmlayout_NA = zeros(iterations, wf.rows * wf.cols); % 最佳NA的风电场布局
    
    % 风电场参数
    D = wf.turbine_num; % 变量维度（风机数量）
    down = 1; % 下界
    up = wf.rows * wf.cols; % 上界（网格总单元数）
    
    %% DE参数设置
    NP = popsize; % DE种群大小
    F = 0.5; % 缩放因子
    CR = 0.9; % 交叉概率
    strategy = 2; % DE策略：DE/rand/1
    
    %% PSO参数设置
    ps = 7; % PSO种群大小
    w_max = 0.9; % 最大惯性权重
    w_min = 0.4; % 最小惯性权重
    c1 = 2.5; % 认知因子初始值
    c2 = 0.5; % 社会因子初始值
    
    %% 初始化DE种群
    [pop, lu] = windfarm_init(NP, wf.turbine_num, wf);
    popold = zeros(size(pop));
    val = zeros(1, NP);
    
    % 评估初始DE种群
    for i = 1:NP
        [val(i), ~, ~] = wf_fitness(wf, pop(i,:));
    end
    
    % 记录DE最佳值
    [bestval, ibest] = max(val); % 风电场优化是最大化问题
    bestmem = pop(ibest, :);
    bestmemit = bestmem;
    Fbest = bestval; % 初始化全局最佳
    
    %% 初始化PSO种群
    % 使用windfarm_init初始化PSO种群
    [pos, ~] = windfarm_init(ps, wf.turbine_num, wf);
    pbest = pos;
    vm = 0.2 * (up - down); % 速度限制
    vel = vm * (rand(ps, D) - 0.5); % 初始速度
    
    % 评估初始PSO种群
    pbestval = zeros(1, ps);
    for i = 1:ps
        [pbestval(i), ~, ~] = wf_fitness(wf, pbest(i,:));
    end
    
    % 合并种群用于后续操作
    popex = [pop; pbest];
    valex = [val, pbestval];
    
    % 初始化示例种群
    exemp = pbest;
    exempval = pbestval;
    
    % 初始化全局最佳
    [gbestval, id1] = max(pbestval); % 最大化问题
    gbest = pbest(id1, :);
    
    %% 主循环
    for iter = 1:iterations
        % 更新PSO参数
        w = w_max - (w_max - w_min) * iter / iterations;
        c1_current = c1 - 2 * iter / iterations;
        c2_current = 0.5 + 2 * iter / iterations;
        
        %% DE部分 - 简化版本避免索引错误
        popold = pop;
        
        % 简化DE/rand/1变异策略，避免复杂的索引操作
        for i = 1:NP
            % 随机选择三个不同的个体
            r = randperm(NP, 3);
            while any(r == i) % 确保不选择自己
                r = randperm(NP, 3);
            end
            
            % DE/rand/1变异
            mutant = pop(r(1), :) + F * (pop(r(2), :) - pop(r(3), :));
            
            % 整数化和边界处理
            for j = 1:D
                % 四舍五入为整数（网格索引）
                mutant(j) = round(mutant(j));
                
                % 边界约束
                if mutant(j) < down
                    mutant(j) = down;
                elseif mutant(j) > up
                    mutant(j) = up;
                end
            end
            
            % 风电场特定约束处理
            mutant = windfarm_constraint(mutant, wf.NA_loc, D, down, up);
            
            % 二项式交叉
            trial = pop(i, :);
            j_rand = randi(D); % 确保至少一个维度交叉
            for j = 1:D
                if rand() < CR || j == j_rand
                    trial(j) = mutant(j);
                end
            end
            
            % 评估新个体
            [tempval, ~, ~] = wf_fitness(wf, trial);
            
            % 选择操作
            if tempval >= val(i) % 最大化问题
                pop(i, :) = trial;
                val(i) = tempval;
                if tempval > bestval % 更新DE最佳值
                    bestval = tempval;
                    bestmem = trial;
                    bestmemit = bestmem;
                end
            end
        end
        
        %% PSO部分
        for i = 1:ps
            % 基于DE种群生成新示例
            rdx = ceil(rand * (NP + ps));
            while rdx == (i + NP)
                rdx = ceil(rand * (NP + ps));
            end
            
            % 确保rdx在有效范围内
            if rdx > length(valex)
                rdx = length(valex);
            end
            
            % 自适应交叉概率
            if valex(rdx) > pbestval(i) % 最大化问题
                crx = rand(1, D) < 0.1; % 如果DE个体更好，使用较小的交叉概率
            else
                crx = rand(1, D) < 0.9; % 否则使用较大的交叉概率
            end
            
            % 生成新个体
            oo = crx .* pbest(i,:) + (1 - crx) .* popex(rdx,:);
            
            % 整数化和边界处理
            for j = 1:D
                oo(j) = round(oo(j));
                if oo(j) < down
                    oo(j) = down;
                elseif oo(j) > up
                    oo(j) = up;
                end
            end
            oo = windfarm_constraint(oo, wf.NA_loc, D, down, up);
            
            % 评估新个体
            [ooval, ~, ~] = wf_fitness(wf, oo);
            
            % 更新示例
            if ooval > exempval(i) % 最大化问题
                exempval(i) = ooval;
                exemp(i,:) = oo;
            end
            
            % 更新速度 - 使用当前最佳个体
            if gbestval > bestval % 使用更好的全局最优
                vel(i,:) = w * vel(i,:) + ...
                           c1_current * rand(1, D) .* (exemp(i,:) - pos(i,:)) + ...
                           c2_current * rand(1, D) .* (gbest - pos(i,:));
            else
                vel(i,:) = w * vel(i,:) + ...
                           c1_current * rand(1, D) .* (exemp(i,:) - pos(i,:)) + ...
                           c2_current * rand(1, D) .* (bestmemit - pos(i,:));
            end
            
            % 速度限制
            vel(i,:) = max(min(vel(i,:), vm), -vm);
            
            % 更新位置
            pos(i,:) = pos(i,:) + vel(i,:);
            
            % 位置整数化和边界处理
            for j = 1:D
                pos(i,j) = round(pos(i,j));
                if pos(i,j) < down
                    pos(i,j) = down;
                elseif pos(i,j) > up
                    pos(i,j) = up;
                end
            end
            pos(i,:) = windfarm_constraint(pos(i,:), wf.NA_loc, D, down, up);
            
            % 评估新位置
            [tempval, ~, ~] = wf_fitness(wf, pos(i,:));
            
            % 更新个体最优
            if tempval > pbestval(i) % 最大化问题
                pbestval(i) = tempval;
                pbest(i,:) = pos(i,:);
                
                % 更新全局最优
                if tempval > gbestval
                    gbestval = tempval;
                    gbest = pos(i,:);
                end
            end
        end
        
        % 更新合并种群
        popex = [pop; pbest];
        valex = [val, pbestval];
        
        %% 记录最佳值
        % 确定当前全局最佳
        current_best = max([bestval, gbestval]);
        
        % 更新历史最佳
        if current_best > Fbest
            Fbest = current_best;
        end
        
        % 记录迭代结果
        BestChart(iter) = current_best / wf.power_total;
        BestFitness(iter) = current_best;
        
        % 确定最佳个体
        if bestval >= gbestval
            current_best_individual = bestmem;
            current_best_value = bestval;
        else
            current_best_individual = gbest;
            current_best_value = gbestval;
        end
        
        % 生成布局并记录
        [best_farmlayout, best_farmlayout_NA] = gene_layout_by_indices_one(wf, current_best_individual);
        farmlayout(iter, :) = best_farmlayout;
        farmlayout_NA(iter, :) = best_farmlayout_NA;
        
        %% 输出进度信息
        if mod(iter, 50) == 0 || iter == 1 || iter == iterations
            fprintf('NA %d Turbine Num:%d Wind %s run: %d iteration: %d  eta %f fitness %f\n', ...
                NA_type, tn, wt, t, iter, (current_best / wf.power_total), current_best);
        end
    end
end