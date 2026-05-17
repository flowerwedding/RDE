function [Fbest, BestChart, BestFitness, farmlayout, farmlayout_NA] = QL_WOA_GWO(popsize, wf, iterations, NA_type, tn, wt, t)
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
    
    % 计算初始适应度（风电场问题是最大化问题）
    [fitness, ~, ~] = wf_fitness(wf, Positions);
    
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
    
    %% ========== Q-Learning参数初始化 ==========
    N = popsize;
    action_num = 3;  % 三个动作：探索、开发、螺旋更新
    Reward_table = zeros(action_num, action_num, N);
    Q_table = zeros(action_num, action_num, N);
    cur_state = randi(action_num);  % 初始状态随机
    
    % Q-Learning参数
    gamma = 0.5;                     % 折扣因子
    lambda_initial = 0.9;            % 初始学习率
    lambda_final = 0.1;              % 最终学习率
    
    %% ========== 主循环 ==========
    t_iter = 0;  % 迭代计数器
    
    while t_iter < iterations
        %% ========== 边界处理 ==========
        for i = 1:size(Positions, 1)
            % 返回超出边界的搜索个体
            Flag4ub = Positions(i, :) > up;
            Flag4lb = Positions(i, :) < down;
            Positions(i, :) = (Positions(i, :) .* (~(Flag4ub + Flag4lb))) + up .* Flag4ub + down .* Flag4lb;
            
            % 约束处理（去重和NA区域检查）
            Positions(i, :) = windfarm_constraint(Positions(i, :), wf.NA_loc, D, down, up);
        end
        
        %% ========== 计算适应度 ==========
        for i = 1:popsize
            fitness(i) = wf_fitness(wf, Positions(i, :));
            
            % 更新领导者（最大化问题）
            if fitness(i) > Leader_score
                Leader_score = fitness(i);
                Leader_pos = Positions(i, :);
                Fbest = Leader_score;
            end
        end
        
        %% ========== 排序找到Alpha、Beta、Delta ==========
        [fitness_sorted, sort_index] = sort(fitness, 'descend');  % 降序排列
        Alpha_pos = Positions(sort_index(1), :);  % 最优个体
        Beta_pos = Positions(sort_index(min(2, popsize)), :);  % 次优个体
        Delta_pos = Positions(sort_index(min(3, popsize)), :);  % 第三优个体
        
        %% ========== 更新参数 ==========
        a = 2 - t_iter * (2 / iterations);  % a从2线性递减到0
        a2 = -1 + t_iter * ((-1) / iterations);  % a2从-1线性递减到-2
        w2 = 1 * (rand - 0.5);  % 随机权重
        
        %% ========== 更新每个搜索代理的位置 ==========
        for i = 1:popsize
            r1 = rand();
            r2 = rand();
            
            A = 2 * a * r1 - a;  % 式(2.3)
            C = 2 * r2;           % 式(2.4)
            
            b = 1;                 % 螺旋形状常数
            l = (a2 - 1) * rand + 1;  % 螺旋更新参数
            
            p = rand();            % 概率参数
            
            Position = Positions;
            
            %% ========== Q-Learning选择动作 ==========
            % 根据Q表选择最优动作
            [~, action] = max(Q_table(cur_state, :, i));
            
            % 对每个维度进行更新
            for j = 1:D
                if action == 1
                    % 动作1：探索策略（WOA探索）
                    rand_leader_index = randi(popsize);
                    X_rand = Positions(rand_leader_index, :);
                    D_X_rand = abs(C * X_rand(j) - Positions(i, j));
                    Position(i, j) = round(X_rand(j) - A * D_X_rand);
                    
                elseif action == 2
                    % 动作2：开发策略（向领导者靠近）
                    D_Leader = abs(C * Leader_pos(j) - Positions(i, j));
                    Position(i, j) = round(Leader_pos(j) - A * D_Leader);
                    
                else
                    % 动作3：螺旋更新策略（WOA气泡网攻击）
                    distance2Leader = abs(Leader_pos(j) - Positions(i, j));
                    Position(i, j) = round(distance2Leader * exp(b * l) .* cos(l * 2 * pi) + Leader_pos(j));
                end
            end
            
            %% ========== 边界处理 ==========
            Flag4lb = Position(i, :) < down;
            Flag4ub = Position(i, :) > up;
            Position(i, :) = (Position(i, :) .* (~(Flag4ub + Flag4lb))) + up .* Flag4ub + down .* Flag4lb;
            Position(i, :) = windfarm_constraint(Position(i, :), wf.NA_loc, D, down, up);
            
            %% ========== 评估新位置 ==========
            new_fitness = wf_fitness(wf, Position(i, :));
            
            %% ========== GWO变异操作 ==========
            X = zeros(1, D);
            for j = 1:D
                if action == 1
                    % 使用Alpha狼更新
                    r1 = rand();
                    r2 = rand();
                    A1 = 2 * a * r1 - a;
                    C1 = 2 * r2;
                    D_alpha = abs(C1 * Alpha_pos(j) - Positions(i, j));
                    X(j) = round(Alpha_pos(j) .* w2 - A1 * D_alpha);
                    
                elseif action == 2
                    % 使用Beta狼更新
                    r1 = rand();
                    r2 = rand();
                    A2 = 2 * a * r1 - a;
                    C2 = 2 * r2;
                    D_beta = abs(C2 * Beta_pos(j) - Positions(i, j));
                    X(j) = round(Beta_pos(j) .* w2 - A2 * D_beta);
                    
                else
                    % 使用Delta狼更新
                    r1 = rand();
                    r2 = rand();
                    A3 = 2 * a * r1 - a;
                    C3 = 2 * r2;
                    D_delta = abs(C3 * Delta_pos(j) - Positions(i, j));
                    X(j) = round(Delta_pos(j) .* w2 - A3 * D_delta);
                end
            end
            
            %% ========== 边界处理（变异个体） ==========
            Flag4lb = X < down;
            Flag4ub = X > up;
            X = (X .* (~(Flag4ub + Flag4lb))) + up .* Flag4ub + down .* Flag4lb;
            X = windfarm_constraint(X, wf.NA_loc, D, down, up);
            
            %% ========== 评估变异个体 ==========
            mutation_fit = wf_fitness(wf, X);
            
            %% ========== 选择更好的个体 ==========
            if mutation_fit > new_fitness
                new_fitness = mutation_fit;
                Position(i, :) = X;
            end
            
            %% ========== 更新种群 ==========
            if new_fitness > fitness(i)
                Positions(i, :) = Position(i, :);
                fitness(i) = new_fitness;
                Reward = 1;  % 正奖励
            else
                Reward = -1;  % 负奖励
            end
            
            %% ========== Q-Learning更新 ==========
            Reward_table(cur_state, action, i) = Reward;
            r = Reward_table(cur_state, action, i);
            maxQ = max(Q_table(action, :, i));
            
            % 动态学习率（余弦衰减）
            lambda = (lambda_initial + lambda_final) / 2 - ...
                     (lambda_initial - lambda_final) / 2 * cos(pi * (1 - t_iter / iterations));
            
            % Q表更新
            Q_table(cur_state, action, i) = Q_table(cur_state, action, i) + ...
                lambda * (r + gamma * maxQ - Q_table(cur_state, action, i));
            
            % 更新当前状态
            cur_state = action;
            
            %% ========== 更新全局最优 ==========
            if fitness(i) > Leader_score
                Leader_score = fitness(i);
                Leader_pos = Positions(i, :);
                Fbest = Leader_score;
            end
        end
        
        %% ========== 迭代计数和记录 ==========
        t_iter = t_iter + 1;
        
        % 记录当前代结果
        BestChart(t_iter) = Leader_score / wf.power_total;
        BestFitness(t_iter) = Leader_score;
        [best_farmlayout, best_farmlayout_NA] = gene_layout_by_indices_one(wf, Leader_pos);
        farmlayout(t_iter, :) = best_farmlayout;
        farmlayout_NA(t_iter, :) = best_farmlayout_NA;
        
        %% ========== 输出进度 ==========
        if mod(t_iter, 50) == 0 || t_iter == iterations
            fprintf('QL-WOA-GWO - NA %d Turbine Num:%d Wind %s run: %d iteration: %d  eta %f fitness %f\n', ...
                NA_type, tn, wt, t, t_iter, (Leader_score / wf.power_total), Leader_score);
        end
    end
    
    fprintf('QL-WOA-GWO优化完成! 运行次数:%d, 最佳适应度: %f (归一化: %f)\n', ...
        t, Fbest, BestChart(end));
end