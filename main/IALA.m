function [Fbest, BestChart, BestFitness, farmlayout, farmlayout_NA] = IALA(popsize, wf, iterations, NA_type, tn, wt, t)    
    % github.com/HYJ-328/Improved-Artificial-Lemming-Algorithm 
    % 初始化记录数组
    BestChart = zeros(iterations, 1);                  % 最佳归一化适应度
    BestFitness = zeros(iterations, 1);                % 最佳实际适应度
    farmlayout = zeros(iterations, wf.rows * wf.cols); % 最佳风电场布局
    farmlayout_NA = zeros(iterations, wf.rows * wf.cols); % 最佳NA的风电场布局
    
    % 问题维度设置
    D = wf.turbine_num;        % 变量维度（风机数量）
    down = 1;                  % 下界
    up = wf.rows * wf.cols;    % 上界（网格总单元数）
    
    %% ========== 种群初始化（Hammersley序列） ==========
    hammersley = hammersley_sequence(1, popsize, D, popsize);% Hammersley序列生成
    % 映射到实际范围
    Positions = round(hammersley' .* (up - down) + down);
    % 边界处理
    Positions = max(min(Positions, up), down);
    Positions = windfarm_constraint(Positions, wf.NA_loc, D, down, up);
    
    %% ========== 计算初始适应度 ==========
    [fitness, ~, ~] = wf_fitness(wf, Positions);
    
    %% ========== 初始化最优解（最大化问题） ==========
    [Leader_score, leader_idx] = max(fitness);  % 最大化问题用max
    Leader_pos = Positions(leader_idx, :);
    Fbest = Leader_score;
    
    % 记录第一代结果
    BestChart(1) = Leader_score / wf.power_total;
    BestFitness(1) = Leader_score;
    [best_farmlayout, best_farmlayout_NA] = gene_layout_by_indices_one(wf, Leader_pos);
    farmlayout(1, :) = best_farmlayout;
    farmlayout_NA(1, :) = best_farmlayout_NA;
    
    %% ========== IALA参数初始化 ==========
    vec_flag = [1, -1];  % 选择移动方向
    
    %% ========== 主循环 ==========
    for Iter = 1:iterations
        theta = 2 * atan(1 - Iter/iterations);% 时变参数theta从π/2线性递减到0，只用在E
        
        RB = randn(popsize, D);% 布朗运动，生成正态分布随机数，只用在第一阶段的探索阶段
        Xnew = zeros(popsize, D);
        
        %% ========== 第一阶段：IALA主更新机制 ==========
        for i = 1:popsize
            F = vec_flag(floor(2 * rand() + 1));% 随机方向标志1或-1
            E = 2 * log(1/rand()) * theta;% 参数E
            
            if E > 1  % 探索阶段
                if rand() < 0.3
                    % 探索策略1：带随机权重的差分变异
                    r1 = 2 * rand(1, D) - 1;% 生成[-1, 1]之间的随机权重
                    RJ = randi(popsize);
                    while RJ == i
                        RJ = randi(popsize);
                    end
                    % 最优个体 + 方向 × 布朗向量 × (随机权重 × (最优个体 - 当前个体) + (1 - 随机权重) × (当前个体 - 随机个体))
                    Xnew(i, :) = round(Leader_pos + F .* RB(i, :) .* ...
                        (r1 .* (Leader_pos - Positions(i, :)) + ...
                        (1 - r1) .* (Positions(i, :) - Positions(RJ, :))));
                else
                    % 探索策略2：向随机个体学习
                    r2 = rand() * (1 + sin(0.5 * Iter));% 在[0, 2]之间随迭代振荡
                    RJ = randi(popsize);
                    while RJ == i
                        RJ = randi(popsize);
                    end
                    % 当前个体 + 方向 × 随机权重 × (最优个体 - 随机个体)
                    Xnew(i, :) = round(Positions(i, :) + F .* r2 * (Leader_pos - Positions(RJ, :)));
                end
                
            else  % 开发阶段
                if rand() < 0.5
                    % 开发策略1：竞争学习策略
                    j = randi(popsize);
                    while j == i
                        j = randi(popsize);
                    end
                    s = rand(1, D);
                    if fitness(j) < fitness(i)% 竞争对手更差，向远离竞争对手的方向运动
                        Xnew(i, :) = round(Positions(i, :) + s .* (Positions(i, :) - Positions(j, :)));
                    else% 竞争对手更好，向靠近竞争对手的方向运动
                        Xnew(i, :) = round(Positions(i, :) + s .* (Positions(j, :) - Positions(i, :)));
                    end
                else
                    % 开发策略2：Levy飞行引导的开发
                    G = 2 * (sign(rand() - 0.5)) * (1 - Iter/iterations);% 自适应步长G
                    Levy_step = Levy(D);% Levy飞行步长
                    Xnew(i, :) = round(Leader_pos + F .* G * Levy_step .* (Leader_pos - Positions(i, :)));
                end
            end
            
            % 边界处理
            Xnew(i, :) = max(min(Xnew(i, :), up), down);
        end
        
        %% ========== 约束处理 ==========
        Xnew = windfarm_constraint(Xnew, wf.NA_loc, D, down, up);
        
        %% ========== 评估和选择 ==========
        for i = 1:popsize
            new_fitness = wf_fitness(wf, Xnew(i, :));
            
            % 贪婪选择
            if new_fitness > fitness(i)
                Positions(i, :) = Xnew(i, :);
                fitness(i) = new_fitness;
            end
            
            % 更新全局最优
            if fitness(i) > Leader_score
                Leader_score = fitness(i);
                Leader_pos = Positions(i, :);
                Fbest = Leader_score;
            end
        end
        
        %% ========== 第二阶段：Lemming防御机制 ==========
        for i = 1:popsize
            % Lemming防御机制
            U1 = rand(1, D) > rand();% 随机二进制掩码
            
            % 随机选择两个不同的个体
            RJ1 = randi(popsize);
            while RJ1 == i
                RJ1 = randi(popsize);
            end
            RJ2 = randi(popsize);
            while RJ2 == i || RJ2 == RJ1
                RJ2 = randi(popsize);
            end
            
            % 平均位置
            y = (Positions(i, :) + Positions(RJ1, :)) / 2;
            
            % 生成新解
            Xnew(i, :) = round((U1) .* Positions(i, :) + ...
                (1 - U1) .* (y + rand() * (Positions(RJ1, :) - Positions(RJ2, :))));
            
            % 边界处理
            Xnew(i, :) = max(min(Xnew(i, :), up), down);
        end
        
        %% ========== 约束处理 ==========
        Xnew = windfarm_constraint(Xnew, wf.NA_loc, D, down, up);
        
        %% ========== 评估和选择 ==========
        for i = 1:popsize
            new_fitness = wf_fitness(wf, Xnew(i, :));
            
            if new_fitness > fitness(i)
                Positions(i, :) = Xnew(i, :);
                fitness(i) = new_fitness;
            end
            
            if fitness(i) > Leader_score
                Leader_score = fitness(i);
                Leader_pos = Positions(i, :);
                Fbest = Leader_score;
            end
        end
        
        %% ========== 记录当前代结果 ==========
        BestChart(Iter) = Leader_score / wf.power_total;
        BestFitness(Iter) = Leader_score;
        [best_farmlayout, best_farmlayout_NA] = gene_layout_by_indices_one(wf, Leader_pos);
        farmlayout(Iter, :) = best_farmlayout;
        farmlayout_NA(Iter, :) = best_farmlayout_NA;
        
        %% ========== 输出进度 ==========
        if mod(Iter, 50) == 0 || Iter == iterations
            fprintf('IALA - NA %d Turbine Num:%d Wind %s run: %d iteration: %d  eta %f fitness %f\n', ...
                NA_type, tn, wt, t, Iter, (Fbest / wf.power_total), Fbest);
        end
    end
    
    fprintf('IALA优化完成! 运行次数:%d, 最佳适应度: %f (归一化: %f)\n', ...
        t, Fbest, BestChart(end));
end

%% ========== Levy飞行函数 ==========
function o = Levy(d)
    beta = 1.5;
    sigma = (gamma(1+beta) * sin(pi*beta/2) / (gamma((1+beta)/2) * beta * 2^((beta-1)/2)))^(1/beta);
    u = randn(1, d) * sigma;
    v = randn(1, d);
    step = u ./ abs(v).^(1/beta);
    o = step;
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