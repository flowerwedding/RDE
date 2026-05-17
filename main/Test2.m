function [Bestr1r2, Fbest, BestChart, BestFitness, farmlayout, farmlayout_NA] = HPSO2(popsize, wf, iterations, NA_type, tn, wt, t, Chaos_p, Strategy, rad_value, cls_type, c_r_type) 
    % 初始化记录数组
    BestChart = zeros(iterations, 1);                  % 最佳归一化适应度
    BestFitness = zeros(iterations, 1);                % 最佳实际适应度
    Bestr1r2 = zeros(iterations, 2);                   % r1和r2参数记录
    farmlayout = zeros(iterations, wf.rows * wf.cols); % 最佳风电场布局
    farmlayout_NA = zeros(iterations, wf.rows * wf.cols); % 最佳NA的风电场布局
    pop_power = zeros(popsize, wf.cols * wf.rows);     % 种群功率分布
    
    % 问题维度设置
    D = wf.turbine_num;        % 变量维度（风机数量）
    down = 1;                  % 下界
    up = wf.rows * wf.cols;    % 上界（网格总单元数）
    
    %% ========== 固定参数设置 ==========
    r1 = 0.5;                   % 固定学习率参数1
    r2 = 0.5;                   % 固定学习率参数2
        
    %% ========== 种群初始化（Hammersley序列） ==========
    % 使用Hammersley序列生成初始种群
    hammersley = hammersley_sequence(1, popsize, D, popsize);% Hammersley序列生成
    % 映射到实际范围（网格单元索引）
    popu = round(hammersley' .* (up - down) + down);
    % 边界处理
    popu = max(min(popu, up), down);
    popu = windfarm_constraint(popu, wf.NA_loc, D, down, up);
    
    % 获取每个维度的上下界（用于变异操作）
    lu = [down * ones(1, D); up * ones(1, D)];
    
    %% ========== 计算初始适应度 ==========
    [popuFitness, power_order, lp_power_accum] = wf_fitness(wf, popu);
    optimal = max(popuFitness);  % 记录当前最佳适应度
    
    % 计算每个网格单元对总功率的贡献
    for i = 1:popsize
        pop_power(i, power_order(i,:)) = pop_power(i, power_order(i,:)) + (lp_power_accum(i,:) / sum(lp_power_accum(i,:)));
    end

    %% ========== CGPSO参数初始化 ==========
    vel = zeros(popsize, D);           % 速度向量
    pBest = popu;                      % 个体历史最佳位置
    pBestFit = popuFitness;             % 个体历史最佳适应度
    pBest_power = power_order;          % 个体最佳功率顺序
    [~, gBestId] = max(pBestFit);       % 全局最佳个体索引
    gBest = pBest(gBestId, :);          % 全局最佳位置
    gBest_fitness = pBestFit(gBestId);  % 全局最佳适应度
    gBest_power = pBest_power(gBestId, :); % 全局最佳功率顺序
    
    % PSO参数
    omega = 0.9;                        % 惯性权重
    c1 = 1.49618;                       % 个体学习因子
    c2 = 1.49618;                       % 社会学习因子
    pm = 0.01;
    
    % 其他参数
    flag = zeros(popsize, 1);            % 标志位数组
    sg = 7;                              % 锦标赛触发代数
    count = 0;                           % 计数器
    pfit = ones(1, 12);                  % 适应度概率数组
    LEP = 25;                            % 长度参数
    success_num = zeros(LEP, 12);        % 成功次数记录
    fail_num = zeros(LEP, 12);           % 失败次数记录
    temp_lep = 1;                        % LEP索引
    
    % HHWOA参数
    DE_iterations = 5;                   % 差分进化迭代次数
    F = 0.5;                             % 差分进化缩放因子
    CR = 0.9;                            % 差分进化交叉概率
    
    %% ========== 主循环 ==========
    for iter = 1:iterations
        st = 0;  % 成功标志
        
        % 两个随机个体的差作为半径
        randPopuList = randperm(popsize);
        randPopuList = setdiff(randPopuList, 1, 'stable');
        indiR1 = pBest(randPopuList(1), :);
        indiR2 = pBest(randPopuList(2), :);
        radius = indiR1 - indiR2;
  
        %% ========== 反向学习初始化 ==========
        OBLPositions = OBL_initialization(popsize, D, down, up, popu);
        
        %% ========== 合并种群 ==========
        TPositions = zeros(2 * popsize, D);
        TPopFitness = zeros(2 * popsize, 1);
        
        for i = 1:popsize
            TPositions(i, :) = popu(i, :);           % 原种群
            TPositions(i + popsize, :) = OBLPositions(i, :); % 反向种群
        end
        
        %% ========== 评估合并种群 ==========
        for i = 1:2 * popsize
            TPopFitness(i) = wf_fitness(wf, TPositions(i, :));
        end
        
        %% ========== 选择最优的popsize个个体 ==========
        [~, Tindex] = sort(TPopFitness, 'descend');
        for newindex = 1:popsize
            popu(newindex, :) = TPositions(Tindex(newindex), :);
            popuFitness(newindex) = TPopFitness(Tindex(newindex));
            [~, power_order(newindex, :), lp_power_accum(newindex, :)] = wf_fitness(wf, popu(newindex, :));
        end
        
        %% ========== 差分进化策略 -> 局部搜索 ==========
        [popu, popuFitness, power_order, lp_power_accum, gBest, gBest_fitness, gBest_power] = ...
            DE_LocalSearch_RPSO(popu, popuFitness, power_order, lp_power_accum, ...
            gBest, gBest_fitness, gBest_power, down, up, wf, D, DE_iterations, F, CR);

        %% ========== 混沌选择策略 -> 全局搜索 ==========
        [gBest, gBest_fitness, pfit, gBest_power, temp_lep] = Chaotic_selection(...
            gBest, gBest_fitness, down, up, Chaos_p, wf, D, pfit, temp_lep, ...
            Strategy, iter, radius, success_num, fail_num, LEP, gBest_power, cls_type, c_r_type);
        
        %% ========== 对每个个体进行更新 ==========
        for i = 1:popsize
            %% ========== 交叉操作（固定的r1和r2） ==========
            offsPbest = zeros(1, D);
            for d = 1:D
                k = randperm(popsize, 1);
                if pBestFit(i) > pBestFit(k)
                    % 如果当前个体优于随机个体，使用加权组合
                    offsPbest(d) = r1 * pBest(i, d) + r2 * gBest(1, d);
                else
                    % 否则直接复制随机个体的位置
                    offsPbest(d) = pBest(k, d);
                end
            end
            
            %% ========== 变异操作 ==========
            for d = 1:D
                 if rand < pm
                    % 随机变异到搜索空间内的任意位置
                    offsPbest(d) = lu(1, d) + rand * (lu(2, d) - lu(1, d));
                end
            end
            
            %% ========== 约束处理 ==========
            offsPbest = windfarm_constraint(offsPbest, wf.NA_loc, D, down, up);
            
            %% ========== 评估新个体 ==========
            [offsPbestFitness, offs_power_order, ~] = wf_fitness(wf, offsPbest);
            
            % 更新最优值
            optimal = max(popuFitness);    
            if offsPbestFitness > optimal
                optimal = offsPbestFitness;
                st = st + 1;  % 成功计数加1
            end           
            
            % 如果新个体更好，更新个体历史最优
            if offsPbestFitness > pBestFit(i)
                pBest(i, :) = offsPbest;
                pBestFit(i) = offsPbestFitness;
                pBest_power(i, :) = offs_power_order;
            end
            
            %% ========== 20%锦标赛 ==========
            if flag(i) == sg
                flag(i) = 0;
                % 随机选择20%的个体进行锦标赛
                competitor = randperm(popsize, max(1, round(0.2 * popsize)));
                [~, winId] = max(pBestFit(competitor));
                % 用优胜者替换当前个体
                pBest(i, :) = pBest(competitor(winId), :);
                pBestFit(i) = pBestFit(competitor(winId));
                pBest_power(i, :) = pBest_power(competitor(winId), :);
            end    
            
            %% ========== 粒子更新（PSO核心公式） ==========
            for d = 1:D
                % 速度更新
                vel(i, d) = omega * vel(i, d) + c1 * rand * pBest(i, d) - c2 * rand * popu(i, d);
                % 位置更新
                popu(i, d) = round(popu(i, d) + vel(i, d));
            end
        end
        
        %% ========== 边界检测和约束处理 ==========
        popu = windfarm_constraint(popu, wf.NA_loc, D, down, up);
        
        %% ========== 评估新种群 ==========
        [popuFitness, power_order, lp_power_accum] = wf_fitness(wf, popu);
        
        % 更新功率贡献矩阵
        for i = 1:popsize
            pop_power(i, power_order(i,:)) = pop_power(i, power_order(i,:)) + (lp_power_accum(i,:) / sum(lp_power_accum(i,:)));
        end
        
        %% ========== 更新最优值统计 ==========
        if max(popuFitness) > optimal
            optimal = max(popuFitness);
        else
            count = count + 1;
        end
        
        %% ========== 更新个体历史最优 ==========
        pos = popuFitness > pBestFit;      % 找出比历史最优更好的个体
        flag(~pos) = flag(~pos) + 1;        % 未更新的个体标志位加1
        pBestFit(pos) = popuFitness(pos);    % 更新适应度
        pBest(pos, :) = popu(pos, :);        % 更新位置
        pBest_power(pos, :) = power_order(pos, :); % 更新功率顺序
        
        % 更新全局最优
        [gBestFit, gBestId] = max(pBestFit);
        if gBestFit > gBest_fitness
            gBest = pBest(gBestId, :);
            gBest_fitness = gBestFit;
            gBest_power = pBest_power(gBestId, :);
        end
        
        %% ========== 记录当前代结果 ==========
        Fbest = gBest_fitness;
        BestChart(iter) = gBest_fitness / wf.power_total;
        BestFitness(iter) = gBest_fitness;
        Bestr1r2(iter, 1) = r1;
        Bestr1r2(iter, 2) = r2;
        
        % 生成布局
        [best_farmlayout, best_farmlayout_NA] = gene_layout_by_indices_one(wf, gBest);
        farmlayout(iter, :) = best_farmlayout;
        farmlayout_NA(iter, :) = best_farmlayout_NA;
        
        %% ========== 输出进度 ==========
        if mod(iter, 50) == 0 || iter == iterations
            fprintf('HPSO-Hammersley (r1=%.2f, r2=%.2f) - NA %d Turbine Num:%d Wind %s run: %d iteration: %d  eta %f fitness %f success=%d\n', ...
                r1, r2, NA_type, tn, wt, t, iter, (Fbest / wf.power_total), Fbest, st);
        end
    end
end

%% ========== Hammersley序列生成函数 ==========
function r = hammersley_sequence(i1, i2, m, n)
    prime = [ ...
        2;   3;   5;   7;  11;  13;  17;  19;  23;  29; ...
       31;  37;  41;  43;  47;  53;  59;  61;  67;  71; ...
       73;  79;  83;  89;  97; 101; 103; 107; 109; 113; ...
      127; 131; 137; 139; 149; 151; 157; 163; 167; 173; ...
      179; 181; 191; 193; 197; 199; 211; 223; 227; 229; ...
      233; 239; 241; 251; 257; 263; 269; 271; 277; 281; ...
      283; 293; 307; 311; 313; 317; 331; 337; 347; 349; ...
      353; 359; 367; 373; 379; 383; 389; 397; 401; 409; ...
      419; 421; 431; 433; 439; 443; 449; 457; 461; 463; ...
      467; 479; 487; 491; 499; 503; 509; 521; 523; 541 ];
    
    if i1 <= i2
        step = 1;
    else
        step = -1;
    end
    
    L = abs(i2 - i1) + 1;
    r = zeros(m, L);
    k = 0;
    
    for i = i1:step:i2
        t(1:m-1,1) = i;
        prime_inv = zeros(m-1, 1);
        for idx = 1:m-1
            prime_inv(idx) = 1.0 / prime(idx);
        end
        k = k + 1;
        r(1, k) = mod(i, n+1) / n;
        
        while any(t ~= 0)
            for j = 1:m-1
                d = mod(t(j), prime(j));
                r(j+1, k) = r(j+1, k) + d * prime_inv(j);
                prime_inv(j) = prime_inv(j) / prime(j);
                t(j) = floor(t(j) / prime(j));
            end
        end
    end
end

%% ========== 反向学习初始化函数 ==========
function OBLPositions = OBL_initialization(N, D, lb, ub, Positions)
    % 生成反向学习位置
    % 反向学习公式：OBL = lb + ub - x
    OBLPositions = zeros(N, D);
    for i = 1:N
        for j = 1:D
            OBLPositions(i, j) = lb + ub - Positions(i, j);
        end
    end
    
    % 边界处理
    OBLPositions = max(min(OBLPositions, ub), lb);
end

%% ========== 差分进化局部搜索函数 ==========
function [popu, popuFitness, power_order, lp_power_accum, gBest, gBest_fitness, gBest_power] = ...
    DE_LocalSearch_RPSO(popu, popuFitness, power_order, lp_power_accum, ...
    gBest, gBest_fitness, gBest_power, lb, ub, wf, D, DE_iterations, F, CR)
    
    popsize = size(popu, 1);
    DEPositions = zeros(popsize, D);
    
    for t = 1:DE_iterations
        for i = 1:popsize
            % 随机选择三个不同的个体
            kkk = randperm(popsize);
            kkk(kkk == i) = [];     % 移除当前个体
            kkk = kkk(1:3);         % 取前三个
            
            jrand = randi(D);       % 确保至少一个维度交叉
            
            for j = 1:D
                if (rand <= CR) || (jrand == j)
                    % DE/rand/1变异策略 V = X1 + F·(X2 - X3)
                    DEPositions(i, j) = round(popu(kkk(1), j) + F * (popu(kkk(2), j) - popu(kkk(3), j)));
                else
                    DEPositions(i, j) = popu(i, j); % 不变异
                end
            end
            
            % 边界处理
            DEPositions(i, :) = max(min(DEPositions(i, :), ub), lb);
            DEPositions(i, :) = windfarm_constraint(DEPositions(i, :), wf.NA_loc, D, lb, ub);
            
            % 评估新个体
            [new_fitness, new_power_order, new_lp_power_accum] = wf_fitness(wf, DEPositions(i, :));
            
            % 贪婪选择
            if new_fitness > popuFitness(i)
                popu(i, :) = DEPositions(i, :);
                popuFitness(i) = new_fitness;
                power_order(i, :) = new_power_order;
                lp_power_accum(i, :) = new_lp_power_accum;
                
                % 更新全局最优
                if new_fitness > gBest_fitness
                    gBest_fitness = new_fitness;
                    gBest = DEPositions(i, :);
                    gBest_power = new_power_order;
                end
            end
        end
    end
end