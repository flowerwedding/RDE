function [Fbest, BestChart, BestFitness, farmlayout, farmlayout_NA] = HHWOA(popsize, wf, iterations, NA_type, tn, wt, t)
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
    
    %% ========== 主循环 ==========
    for t_iter = 1:iterations
        %% ========== 混沌映射更新p值 ==========
        p = abs(cos(w * acos(p)));  % WOA的随机决策，Chebyshev混沌映射，产生[0,1]之间的混沌序列
        
        %% ========== 反向学习初始化 ==========
        OBLPositions = OBL_initialization(popsize, D, down, up, Positions); % 生成每个个体的反向位置OBL = lb + ub - x
        
        %% ========== 合并种群 ==========
        TPositions = zeros(2 * popsize, D);
        for i = 1:popsize
            TPositions(i, :) = Positions(i, :); % 原种群
            TPositions(i + popsize, :) = OBLPositions(i, :); % 反向种群
        end
        
        %% ========== 评估合并种群 ==========
        for i = 1:2 * popsize
            Tfitness(i) = wf_fitness(wf, TPositions(i, :));
        end
        
        %% ========== 选择最优的popsize个个体进入下一代 ==========
        [~, Tindex] = sort(Tfitness, 'descend');  % 降序排序
        for newindex = 1:popsize
            Positions(newindex, :) = TPositions(Tindex(newindex), :);
            fitness(newindex) = Tfitness(Tindex(newindex));
        end
        
        %% ========== 差分进化局部搜索 ==========
        [Positions, fitness, Leader_score, Leader_pos] = DE_LocalSearch(...
            Positions, fitness, Leader_score, Leader_pos, down, up, wf, D);
        
        %% ========== 更新WOA参数 ==========
        a = 2 - t_iter * (2 / iterations);          % 收缩因子a从2线性递减到0，只用在A
        a2 = -1 + t_iter * ((-1) / iterations);      % 螺旋参数a2从-1线性递减到-2，只用在l
        
        %% ========== WOA位置更新 ==========
        for i = 1:popsize
            r1 = rand();
            r2 = rand();
            
            A = 2 * a * r1 - a; % 系数向量
            C = 2 * r2; % 系数向量
            
            b = 1; % 螺旋常数
            l = (a2 - 1) * rand() + 1; % 螺旋参数
            
            for j = 1:D
                if p < 0.5 % 收缩包围机制
                    if abs(A) >= 1
                        % 探索阶段：随机搜索(全局搜索)
                        rand_leader_index = randi(popsize);% 生成1到popsize的随机数
                        X_rand = Positions(rand_leader_index, :);% 获取随机数对应的个体
                        D_X_rand = abs(C * X_rand(j) - Positions(i, j));% 计算当前个体和随机个体间的距离
                        WOA_Positions(i, j) = round(X_rand(j) - A * D_X_rand);% 向随机个体靠近
                    else
                        % 开发阶段：向最优解靠近(局部开发)
                        D_Leader = abs(C * Leader_pos(j) - Positions(i, j));% 计算当前个体和最优个体间的距离
                        WOA_Positions(i, j) = round(Leader_pos(j) - A * D_Leader);% 向最优个体靠近
                    end
                else
                    % 螺旋更新阶段
                    Distance2Leader = abs(Leader_pos(j) - Positions(i, j));
                    WOA_Positions(i, j) = round(Distance2Leader * exp(b * l) .* cos(l * 2 * pi) + Leader_pos(j));
                end
            end
            
            % 边界处理，第一次确保在[down, up]，第二次确保不在NA区域
            WOA_Positions(i, :) = max(min(WOA_Positions(i, :), up), down);
            WOA_Positions(i, :) = windfarm_constraint(WOA_Positions(i, :), wf.NA_loc, D, down, up);
            
            % 评估新位置
            new_fitness = wf_fitness(wf, WOA_Positions(i, :));
            
            % 贪婪选择：保留更优解
            if new_fitness > fitness(i)
                Positions(i, :) = WOA_Positions(i, :);
                fitness(i) = new_fitness;
                
                if new_fitness > Leader_score% Leader_score当前全局最优适应度
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
            fprintf('HHWOA - NA %d Turbine Num:%d Wind %s run: %d iteration: %d  eta %f fitness %f\n', ...
                NA_type, tn, wt, t, t_iter, (Fbest / wf.power_total), Fbest);
        end
    end
    
    fprintf('HHWOA优化完成! 运行次数:%d, 最佳适应度: %f (归一化: %f)\n', ...
        t, Fbest, BestChart(end));
end

%% ========== 反向学习初始化函数 ==========
function OBLPositions = OBL_initialization(N, D, lb, ub, Positions)
    % 生成反向学习位置
    OBLPositions = zeros(N, D);
    for i = 1:N
        for j = 1:D
            % 反向学习公式：OBL = lb + ub - x
            OBLPositions(i, j) = lb + ub - Positions(i, j);
        end
    end
    
    % 边界处理
    OBLPositions = max(min(OBLPositions, ub), lb);
end

%% ========== 差分进化局部搜索 ==========
function [Positions, fitness, Leader_score, Leader_pos] = DE_LocalSearch(...
    Positions, fitness, Leader_score, Leader_pos, lb, ub, wf, D)
    
    % DE参数
    F = 0.5;                    % 缩放因子
    CR = 0.9;                   % 交叉概率
    DEMax_iter = 5;             % DE最大迭代次数
    
    DEPositions = zeros(size(Positions));
    popsize = size(Positions, 1);
    
    for t = 1:DEMax_iter
        for i = 1:popsize
            % 随机选择三个不同的个体
            kkk = randperm(popsize);% 生成1到popsize的随机排列
            kkk(kkk == i) = [];     % 移除当前个体
            kkk = kkk(1:3);         % 取前三个
            
            jrand = randi(D);     % 确保至少一个维度交叉
            
            for j = 1:D
                if (rand <= CR) || (jrand == j)
                    % DE/rand/1变异策略 V = X1 + F·(X2 - X3)
                    DEPositions(i, j) = round(Positions(kkk(1), j) + F * (Positions(kkk(2), j) - Positions(kkk(3), j)));
                    % DEPositions(i, j) = round(Positions(i, j) + F * (pbX(i, j) - Positions(i, j) + Positions(kkk(1), j) - Positions(kkk(2), j)));
                else
                    DEPositions(i, j) = Positions(i, j); % 不变异
                end
            end
            
            % 边界处理
            DEPositions(i, :) = max(min(DEPositions(i, :), ub), lb);
            DEPositions(i, :) = windfarm_constraint(DEPositions(i, :), wf.NA_loc, D, lb, ub);
            
            % 评估
            new_fitness = wf_fitness(wf, DEPositions(i, :));
            
            % 贪婪选择
            if new_fitness > fitness(i)
                Positions(i, :) = DEPositions(i, :);
                fitness(i) = new_fitness;
                
                if new_fitness > Leader_score
                    Leader_score = new_fitness;
                    Leader_pos = DEPositions(i, :);
                end
            end
        end
    end
end