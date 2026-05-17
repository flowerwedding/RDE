function [Fbest, BestChart, BestFitness, farmlayout, farmlayout_NA] = ABCPSO(popsize, wf, iterations, NA_type, tn, wt, t)   
    % 初始化记录数组
    BestChart = zeros(iterations, 1);                  % 最佳归一化适应度
    BestFitness = zeros(iterations, 1);                % 最佳实际适应度
    farmlayout = zeros(iterations, wf.rows * wf.cols); % 最佳风电场布局
    farmlayout_NA = zeros(iterations, wf.rows * wf.cols); % 最佳NA的风电场布局
    
    % 问题维度设置
    D = wf.turbine_num;        % 变量维度（风机数量）
    down = 1;                  % 下界
    up = wf.rows * wf.cols;    % 上界（网格总单元数）
    
    %% ========== 适应度函数 ==========
    CostFunction = @(x) wf_fitness(wf, x);  % 风电场适应度函数（最大化）
    
    %% ========== ABC+PSO参数初始化 ==========
    % PSO参数
    w = 1;                      % 惯性权重
    wdamp = 0.98;               % 惯性权重衰减因子
    c1 = 1.5;                   % 个体学习因子
    c2 = 1.5;                   % 社会学习因子
    
    % ABC参数
    nOnlooker = popsize;        % 观察蜂数量
    L = round(0.6 * D * popsize); % 放弃阈值（蜜源最大未改进次数）
    a = 1;                      % 加速度系数
    
    % 速度边界
    alpha = 0.1;
    VelMax = alpha * (up - down);
    VelMin = -VelMax;
    
    %% ========== 种群结构定义 ==========
    % PSO粒子结构
    empty_particle.Position = [];    % 位置（风机索引）
    empty_particle.Velocity = [];    % 速度
    empty_particle.Cost = [];        % 适应度值
    empty_particle.Sol = [];         % 解结构
    empty_particle.Best.Position = [];
    empty_particle.Best.Cost = [];
    empty_particle.Best.Sol = [];
    
    % ABC蜜蜂结构
    empty_bee.Position = [];
    empty_bee.Cost = [];
    
    %% ========== 种群初始化 ==========
    % 初始化PSO粒子群
    particle = repmat(empty_particle, popsize, 1);
    
    % 初始化ABC种群
    bee_pop = repmat(empty_bee, popsize, 1);
    
    % 初始化全局最优
    GlobalBest.Cost = -inf;  % 最大化问题初始化为负无穷
    GlobalBest.Position = [];
    
    % 初始化放弃计数器（ABC）
    C = zeros(popsize, 1);
    
    %% ========== 初始化循环 ==========
    for i = 1:popsize
        % 初始化位置
        if i == 1
            % 第一个个体使用初始布局（均匀分布）
            [bee_pop(i).Position, ~] = windfarm_init(1, wf.turbine_num, wf);
        else
            % 其他个体随机初始化
            bee_pop(i).Position = windfarm_init(1, wf.turbine_num, wf);
        end
        
        % 评估适应度
        bee_pop(i).Cost = CostFunction(bee_pop(i).Position);
        
        % 更新全局最优
        if bee_pop(i).Cost > GlobalBest.Cost
            GlobalBest.Cost = bee_pop(i).Cost;
            GlobalBest.Position = bee_pop(i).Position;
        end
    end
    
    % 复制到PSO粒子群
    for i = 1:popsize
        particle(i).Position = bee_pop(i).Position;
        particle(i).Velocity = round(rand(1, D) * (VelMax - VelMin) + VelMin);
        particle(i).Cost = bee_pop(i).Cost;
        particle(i).Best.Position = particle(i).Position;
        particle(i).Best.Cost = particle(i).Cost;
    end
    
    % 更新全局最优到PSO
    for i = 1:popsize
        if particle(i).Cost > GlobalBest.Cost
            GlobalBest.Cost = particle(i).Cost;
            GlobalBest.Position = particle(i).Position;
        end
    end
    
    % 记录结果数组
    BestCost = zeros(iterations, 1);
    
    %% ========== 记录第一代结果 ==========
    BestChart(1) = GlobalBest.Cost / wf.power_total;
    BestFitness(1) = GlobalBest.Cost;
    Fbest = GlobalBest.Cost;
    [best_farmlayout, best_farmlayout_NA] = gene_layout_by_indices_one(wf, GlobalBest.Position);
    farmlayout(1, :) = best_farmlayout;
    farmlayout_NA(1, :) = best_farmlayout_NA;
    
    %% ========== ABC+PSO主循环 ==========
    for it = 1:iterations
        %% ========== ABC阶段：雇佣蜂 ==========
        for i = 1:popsize
            % 随机选择另一个不同的个体
            K = [1:i-1, i+1:popsize];
            k = K(randi([1, numel(K)]));
            
            % 计算加速度系数（使用rand替代unifrnd）
            % 原: phi = a * unifrnd(-1, 1, 1, D);
            phi = a * (2 * rand(1, D) - 1);  % 生成[-1,1]之间的均匀随机数
            
            % 生成新位置（雇佣蜂探索）
            new_position = bee_pop(i).Position + round(phi .* (bee_pop(i).Position - bee_pop(k).Position));
            
            % 边界处理和约束
            new_position = max(min(new_position, up), down);
            new_position = windfarm_constraint(new_position, wf.NA_loc, D, down, up);
            
            % 评估新位置
            new_cost = CostFunction(new_position);
            
            % 贪婪选择
            if new_cost > bee_pop(i).Cost
                bee_pop(i).Position = new_position;
                bee_pop(i).Cost = new_cost;
                C(i) = 0;  % 重置放弃计数器
            else
                C(i) = C(i) + 1;
            end
            
            % 更新全局最优
            if bee_pop(i).Cost > GlobalBest.Cost
                GlobalBest.Cost = bee_pop(i).Cost;
                GlobalBest.Position = bee_pop(i).Position;
            end
        end
        
        %% ========== ABC阶段：计算选择概率 ==========
        % 基于适应度计算选择概率（最大化问题）
        F = zeros(popsize, 1);
        MeanCost = mean([bee_pop.Cost]);
        for i = 1:popsize
            % 适应度转换为概率权重
            F(i) = exp(bee_pop(i).Cost / (MeanCost + eps));
        end
        P = F / (sum(F) + eps);
        
        %% ========== ABC阶段：观察蜂 ==========
        for m = 1:nOnlooker
            % 轮盘赌选择蜜源
            i = RouletteWheelSelection(P);
            if isempty(i)
                i = randi(popsize);
            end
            
            % 随机选择另一个不同的个体
            K = [1:i-1, i+1:popsize];
            if isempty(K)
                K = setdiff(1:popsize, i);
            end
            k = K(randi([1, numel(K)]));
            
            % 计算加速度系数（使用rand替代unifrnd）
            phi = a * (2 * rand(1, D) - 1);  % 生成[-1,1]之间的均匀随机数
            
            % 生成新位置（观察蜂开发）
            new_position = bee_pop(i).Position + round(phi .* (bee_pop(i).Position - bee_pop(k).Position));
            
            % 边界处理和约束
            new_position = max(min(new_position, up), down);
            new_position = windfarm_constraint(new_position, wf.NA_loc, D, down, up);
            
            % 评估新位置
            new_cost = CostFunction(new_position);
            
            % 贪婪选择
            if new_cost > bee_pop(i).Cost
                bee_pop(i).Position = new_position;
                bee_pop(i).Cost = new_cost;
                C(i) = 0;
            else
                C(i) = C(i) + 1;
            end
            
            % 更新全局最优
            if bee_pop(i).Cost > GlobalBest.Cost
                GlobalBest.Cost = bee_pop(i).Cost;
                GlobalBest.Position = bee_pop(i).Position;
            end
        end
        
        %% ========== ABC阶段：侦察蜂 ==========
        for i = 1:popsize
            if C(i) >= L
                % 放弃当前蜜源，随机生成新位置
                bee_pop(i).Position = windfarm_init(1, wf.turbine_num, wf);
                bee_pop(i).Cost = CostFunction(bee_pop(i).Position);
                C(i) = 0;
                
                % 更新全局最优
                if bee_pop(i).Cost > GlobalBest.Cost
                    GlobalBest.Cost = bee_pop(i).Cost;
                    GlobalBest.Position = bee_pop(i).Position;
                end
            end
        end
        
        %% ========== PSO阶段：粒子更新 ==========
        % 将ABC最优结果传递到PSO
        for i = 1:popsize
            particle(i).Position = bee_pop(i).Position;
            particle(i).Cost = bee_pop(i).Cost;
        end
        
        % 更新PSO粒子
        for i = 1:popsize
            % 更新速度
            particle(i).Velocity = round(w * particle(i).Velocity + ...
                c1 * rand(1, D) .* (particle(i).Best.Position - particle(i).Position) + ...
                c2 * rand(1, D) .* (GlobalBest.Position - particle(i).Position));
            
            % 速度边界处理
            particle(i).Velocity = max(min(particle(i).Velocity, VelMax), VelMin);
            
            % 更新位置
            particle(i).Position = particle(i).Position + particle(i).Velocity;
            
            % 边界处理和约束
            particle(i).Position = max(min(particle(i).Position, up), down);
            particle(i).Position = windfarm_constraint(particle(i).Position, wf.NA_loc, D, down, up);
            
            % 评估新位置
            new_cost = CostFunction(particle(i).Position);
            
            % 更新个体最优
            if new_cost > particle(i).Best.Cost
                particle(i).Best.Position = particle(i).Position;
                particle(i).Best.Cost = new_cost;
            end
            
            % 更新全局最优
            if new_cost > GlobalBest.Cost
                GlobalBest.Cost = new_cost;
                GlobalBest.Position = particle(i).Position;
            end
        end
        
        %% ========== 将PSO最优结果反馈到ABC ==========
        for i = 1:popsize
            if particle(i).Best.Cost > bee_pop(i).Cost
                bee_pop(i).Position = particle(i).Best.Position;
                bee_pop(i).Cost = particle(i).Best.Cost;
            end
        end
        
        %% ========== 惯性权重衰减 ==========
        w = w * wdamp;
        
        %% ========== 记录当前代结果 ==========
        BestCost(it) = GlobalBest.Cost;
        BestChart(it) = GlobalBest.Cost / wf.power_total;
        BestFitness(it) = GlobalBest.Cost;
        Fbest = GlobalBest.Cost;
        
        [best_farmlayout, best_farmlayout_NA] = gene_layout_by_indices_one(wf, GlobalBest.Position);
        farmlayout(it, :) = best_farmlayout;
        farmlayout_NA(it, :) = best_farmlayout_NA;
        
        %% ========== 输出进度 ==========
        if mod(it, 50) == 0 || it == iterations
            fprintf('ABC+PSO - NA %d Turbine Num:%d Wind %s run: %d iteration: %d  eta %f fitness %f\n', ...
                NA_type, tn, wt, t, it, (Fbest / wf.power_total), Fbest);
        end
    end
    
    fprintf('ABC+PSO优化完成! 运行次数:%d, 最佳适应度: %f (归一化: %f)\n', ...
        t, Fbest, BestChart(end));
end

%% ========== 轮盘赌选择函数 ==========
function i = RouletteWheelSelection(P)
    % 轮盘赌选择（用于观察蜂选择蜜源）
    % P: 选择概率向量
    if isempty(P) || sum(P) == 0
        i = [];
        return;
    end
    r = rand;
    c = cumsum(P);
    i = find(r <= c, 1, 'first');
    if isempty(i)
        i = 1;
    end
end