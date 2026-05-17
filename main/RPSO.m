function [Bestr1r2, Fbest, BestChart, BestFitness, farmlayout, farmlayout_NA] = RPSO(popsize, wf, iterations, NA_type, tn, wt, t, Chaos_p, Strategy, rad_value, cls_type, c_r_type) 
    % 初始化记录数组 RPSO在CGPSO的基础上加了，这个文件把注释掉了，所以是CGPSO算法
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
    
    %% ========== 种群初始化 ==========
    [popu, lu] = windfarm_init(popsize, wf.turbine_num, wf);
    
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
    
    % 其他参数
    flag = zeros(popsize, 1);            % 标志位数组
    sg = 7;                              % 锦标赛触发代数
    pm = 0.01;                           % 变异概率
    count = 0;                           % 计数器
    pfit = ones(1, 12);                  % 适应度概率数组
    LEP = 25;                            % 长度参数
    success_num = zeros(LEP, 12);        % 成功次数记录
    fail_num = zeros(LEP, 12);           % 失败次数记录
    temp_lep = 1;                        % LEP索引
    
    %% ========== r1和r2参数自适应初始化 ==========
    r1 = 0.5;                            % 初始学习率参数1
    r2 = 0.5;                            % 初始学习率参数2
    r1_min = 0.1;                        % r1最小值
    r1_max = 0.9;                        % r1最大值
    r2_min = 0.1;                        % r2最小值
    r2_max = 0.9;                        % r2最大值
    r1_decay = 0.995;                    % r1衰减因子
    r2_decay = 0.995;                    % r2衰减因子
    r1_adapt_rate = 0.01;                 % r1自适应调整率
    r2_adapt_rate = 0.01;                 % r2自适应调整率
    no_improve_count = 0;                 % 连续无改进计数
    prev_best = gBest_fitness;            % 上一代最优值
    
    %% ========== 主循环 ==========
    for iter = 1:iterations
        st = 0;  % 成功标志
        pjd = iter / iterations;  % 迭代进度
        
        %% ========== 搜索机制 ==========
        if Strategy == 3
            % 策略3：两个随机个体的差作为半径
            randPopuList = randperm(popsize);
            randPopuList = setdiff(randPopuList, 1, 'stable');
            indiR1 = pBest(randPopuList(1), :);
            indiR2 = pBest(randPopuList(2), :);
            radius = indiR1 - indiR2;
        else
            % 其他策略：线性递减半径
            radius = rad_value * (1 - iter/iterations);
        end
        
        %% ========== 混沌选择策略 ==========
        [gBest, gBest_fitness, pfit, gBest_power, temp_lep] = Chaotic_selection(...
            gBest, gBest_fitness, down, up, Chaos_p, wf, D, pfit, temp_lep, ...
            Strategy, iter, radius, success_num, fail_num, LEP, gBest_power, cls_type, c_r_type);

        %{
        %% ========== 强化学习更新r1和r2参数 ==========
        [gBestFitMax, ~] = max(pBestFit);
        pytuple = py.ppo.main(py.numpy.array(tn), py.numpy.array(iter2), py.numpy.array(pBest), py.numpy.array(gBest), pBestFit, gBestFitMax, r1, r2, popsize, D);%         gBest = double(gbest1);
        pytuple = cell(pytuple);
        gBest = double(pytuple{1});
        state_current = double(pytuple{2});
        r1 = state_current(1);
        r2 = state_current(2);
        %}
%{
        %% ========== 自适应调整r1和r2 ==========
        % 方法1：基于迭代进度的线性调整
        r1_linear = 0.9 - 0.5 * pjd;  % 从0.9线性递减到0.4
        r2_linear = 0.1 + 0.5 * pjd;  % 从0.1线性递增到0.6
        
        % 方法2：基于种群多样性的自适应
        pop_std = std(pBest);
        diversity = mean(pop_std) / (up - down);
        
        if diversity < 0.1
            % 多样性低，需要增加探索
            r1_diversity = 0.8;
            r2_diversity = 0.2;
        elseif diversity > 0.5
            % 多样性高，可以加强开发
            r1_diversity = 0.4;
            r2_diversity = 0.6;
        else
            % 中等多样性，平衡探索和开发
            r1_diversity = 0.6;
            r2_diversity = 0.4;
        end
        
        % 方法3：基于改进情况的自适应
        if gBest_fitness > prev_best + 1e-6
            % 有改进，保持当前参数
            no_improve_count = 0;
            r1_improve = r1;
            r2_improve = r2;
        else
            % 无改进，增加随机扰动
            no_improve_count = no_improve_count + 1;
            if no_improve_count > 5
                % 连续5代无改进，增加探索
                r1_improve = 0.3 + 0.5 * rand();
                r2_improve = 0.3 + 0.5 * rand();
            else
                r1_improve = r1;
                r2_improve = r2;
            end
        end
        
        % 综合三种方法，加权平均得到最终的r1和r2
        w1 = 0.3;  % 线性调整权重
        w2 = 0.4;  % 多样性调整权重
        w3 = 0.3;  % 改进情况调整权重
        
        r1 = w1 * r1_linear + w2 * r1_diversity + w3 * r1_improve;
        r2 = w1 * r2_linear + w2 * r2_diversity + w3 * r2_improve;
        
        % 确保r1和r2在有效范围内
        r1 = max(r1_min, min(r1_max, r1));
        r2 = max(r2_min, min(r2_max, r2));
        
        % 添加小随机扰动
        r1 = r1 + 0.05 * randn();
        r2 = r2 + 0.05 * randn();
        r1 = max(r1_min, min(r1_max, r1));
        r2 = max(r2_min, min(r2_max, r2));
%}

        r1 = 0.5;
        r2 = 0.5;
        % 记录上一代最优值
        prev_best = gBest_fitness;
        
        %% ========== 对每个个体进行更新 ==========
        for i = 1:popsize
            %% ========== 示例更新：交叉操作 ==========
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
            
            %% ========== 示例更新：变异操作 ==========
            for d = 1:D
                if rand < pm
                    % 随机变异到搜索空间内的任意位置
                    offsPbest(d) = lu(1, d) + rand * (lu(2, d) - lu(1, d));
                end
            end
           
            %% ========== 示例更新：约束处理 ==========
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
            fprintf('CGPSO - NA %d Turbine Num:%d Wind %s run: %d iteration: %d  eta %f fitness %f r1=%.3f r2=%.3f success=%d\n', ...
                NA_type, tn, wt, t, iter, (Fbest / wf.power_total), Fbest, r1, r2, st);
        end
    end
end