function [Fbest,BestChart,BestFitness,farmlayout,farmlayout_NA]=RDE(popsize,wf,iterations,NA_type,tn,wt,t) %种群大小，风电场结构wf，最大迭代次数，禁用区域，风机数量tn，风电场标识wt，运行次数索引t
    BestChart=zeros(iterations,1);%最佳归一化适应度
    BestFitness=zeros(iterations,1);%最佳实际适应度
    farmlayout = zeros(iterations,wf.rows*wf.cols);%最佳风电场布局
    farmlayout_NA = zeros(iterations,wf.rows*wf.cols);%最佳NA的风电场布局
    pop_power = zeros(popsize,wf.cols * wf.rows);%种群中每个个体的功率分布
    
    % 风机数量 变量下界 变量上界(网格总单元数)
    D = wf.turbine_num; down = 1;up = wf.rows *wf.cols;
    
    % 种群初始化
    [popu,lu] = windfarm_init(popsize, wf.turbine_num,wf);
    % 计算适应度 种群评估
    [popuFitness,power_order,lp_power_accum]= wf_fitness(wf,popu);
    optimal = max(popuFitness);%记录当前最佳适应度
    % 计算每个网格单元对总功率的贡献 power_order个体位置索引，lp_power_accum累积功率值 TODO后续没有用到？
    for i = 1:popsize
        pop_power(i,power_order(i,:))  = pop_power(i,power_order(i,:)) + (lp_power_accum(i,:) / sum(lp_power_accum(i,:)));
    end

   %% 局部参数设置
    vel = zeros(popsize, D);%速度向量
    pBest = popu;%个体历史最佳位置
    pBestFit = popuFitness;%个体历史最佳适应度
    pBest_power = power_order;%个体最佳功率顺序
    [~, gBestId] = max(pBestFit);%全局最佳个体索引
    gBest = pBest(gBestId,:);%全局最佳位置
    gBest_fitness = pBestFit(gBestId);%全局最佳适应度
    old_gBest_fitness = gBest_fitness;%旧的全局最佳适应度
    gBest_power = pBest_power(gBestId,:);%全局最佳功率顺序

    omega = 0.7298;%惯性权重
    c = 1.49618;%学习因子
    flag = zeros(popsize,1);%标志位数组
    sg = 7;%参数
    pm = 0.01;%变异概率
    count = 0;%计数器
    pfit = ones(1, 12);%适应度概率数组
    LEP = 25;%长度参数
    success_num =zeros(LEP,12);%成功次数记录
    fail_num = zeros(LEP,12);%失败次数记录
    temp_lep = 1;%LEP索引
    F = 0.5;%缩放因子
    CR = 0.05;%交叉概率 TODO F和CR下面会动态调整
    num_not_update = 0;%未更新计数
    CR = 0.1;%交叉概率

    %% 主循环
    for iter = 1:iterations
        pjd = iter/iterations;%计算迭代进度比例
        p = 0.5;% + 0.3*(pjd);%动态调整参数
%         p = 0.5*(1+pjd);
%         F = F*2^(1 ./ (1 + exp(-iterations/(iterations+1-iter)));
        self_sy = 2^exp(1-iterations/(iterations+1-iter));%自适应缩放因子F
%         p = 0.5 * self_sy;
        F = 0.5 * self_sy;
%         F = 0.5*(1-pjd);
%         F=0.5;

        CR = 0.1*pjd;%0.5;%0.001*self_sy;%动态调整交叉概率


        % 获取前一半
        [AllFitnessSorted,IndexSorted] = sort(popuFitness,'descend');%按适应度降序排序
        N = popsize;%种群大小
        N_half = round(N/2);%种群大小的一半
        index = floor(p*N_half);%计算分组索引 TODO
        gBest = popu(IndexSorted(1),:);
        pBest = popu(IndexSorted(1:10),:);%选择前10个作为个体最佳参考 TODO

        goodPops=popu(IndexSorted(1:N_half),:);%优秀个体
        %goodPopsIdx = IndexSorted(1:N_half); % 优秀个体对应的原始编号
        worstPops=popu(IndexSorted(N_half+1:N),:);%较差个体
        %worstPopsIdx = IndexSorted(N_half+1:N); % 较差个体对应的原始编号
        
        %% 双群重组
        a = randperm(N_half);%优秀个体的随机排列
        b = randperm(N_half);%较差个体的随机排列
        % c = randperm(N_half);
        % d = randperm(N_half);


        gP = goodPops(a,:);%重新组织优秀个体
        % gPIdx = goodPopsIdx(c); % 对应的原始编号
        gp1 = gP(1:index,:);  % 优秀个体的前index个
        % gp1Idx = gPIdx(1:floor(0.95*N_half)); % 对应的原始编号
        gp2 = gP(index+1:N_half,:); % 优秀个体的后N_half-index个
        % gp2Idx = gPIdx(index+1:N_half); % 对应的原始编号

        wP = worstPops(b,:);%重新组织较差个体
        % wPIdx = worstPopsIdx(d); % 对应的原始编号
        wp1 = wP(1:N_half-index, :); % 较差个体的前N_half-index个
        % wp1Idx = wPIdx(1:N_half-floor(0.05*N_half)); % 对应的原始编号
        wp2 = wP(N_half-index+1:N_half, :); % 较差个体的后index个
        % wp2Idx = wPIdx(N_half-index+1:N_half); % 对应的原始编号

        V_pops = [gp1;wp1];
        % V_popsIdx = [gp1Idx; wp1Idx]; % V_pops 对应的原始编号
        
        U = [gp2;wp2];
        % UIdx = [gp2Idx; wp2Idx]; % U 对应的原始编号
            % Get indices for mutation
        %% 变异和交叉
        [r1, r2, r3] = getindex(N_half); % 从1到N_half中随机选择三个索引
             % Implement DE/rand/1 mutation
        % V = V_pops(r1, :) + F * (V_pops(r2, :) - V_pops(r3, :));
%         V = V_pops(r1, :) + F * (V_pops(r2, :)  - U(r3, :));
        % V = 基向量 + 缩放因子 × (向量A - 向量B)
        % 公式是线性组合，说明邻域不是环形的
        V = V_pops(r1, :) + F * (V_pops(r2, :)  - U(r3, :));% 使用DE/rand/1变异策略
%         V = pBest(floor(rand*3)+1, :) + F * (V_pops(r2, :) - U(r3, :));
%           V = V_pops(r1, :) + F * (V_pops(r2, :) - U(r3, :));
%         V = gBest + F * (V_pops(r2, :) - U(r3, :));
            % Check whether the mutant vector violates the boundaries or not
%         [V] = BoundaryDetection(V,lu);
            % Implement binomial crossover
        %% 二项式交叉
        for i = 1:N_half
            j_rand = floor(rand * D) + 1;%确保至少有一个维度交叉
            tt = rand(1, D) < CR;%创建交叉掩码
            tt(1, j_rand) = 1;%确保第j_rand个维度一定会交叉
%             tt = tt .* rand(1, D) * CR;
            t_ = 1 - tt;%互补掩码
            U(i, :) = tt .* V(i, :) + t_ .* pBest(floor(rand*5)+1, :);%V_pops(i, :);%交叉
%             U(i, :) = tt .* U(i, :) + t_ .* V(i, :);
            %X(IndexSorted(i), :) = t .* V(i, :) + t_ .* X(IndexSorted(i), :);
        end
        
       %% 边界检测和处理
        U = windfarm_constraint(U, wf.NA_loc, D,down,up);%确保解在可行域内
        [fit_U,power_order,lp_power_accum] = wf_fitness(wf, U);%计算新个体的适应度
        
        % 精英保留 TODO
        for i = 1:N_half
            if fit_U    (i) > gBest_fitness%更新全局最佳适应度
                gBest_fitness = fit_U(i);
                % num_not_update =  0;
%                 CR = 0.1;
            end
            % popu_num = IndexSorted(N_half+i); % 差个体的编号->替换差个体 TODO
            popu_num = IndexSorted(i); % 优秀个体的编号->替换好个体
            % popu_num = i;
            % popu_num = V_popsIdx(i);  % 优秀种群V_pops个体的编号
            if fit_U(i, :) >= popuFitness(popu_num, :)%新个体更好，替换原有个体，选择
                popu(popu_num, :) = U(i, :);
                popuFitness(popu_num, :) = fit_U(i, :);
            end
        end
        % 
        % % 提取popu中的前一半优秀个体
        % goodPops = popu(IndexSorted(1:N_half), :);
        % goodPopsFitness = popuFitness(IndexSorted(1:N_half));
        % 
        % % 将新的种群U与前一半优秀个体合并
        % combinedPops = [goodPops; U];
        % combinedFitness = [goodPopsFitness; fit_U];
        % 
        % % 对合并后的种群按照适应度进行排序
        % [AllCombinedFitnessSorted, CombinedIndexSorted] = sort(combinedFitness, 'descend');
        % 
        % % 提取排序后的前一半个体作为新的优秀个体
        % newGoodPops = combinedPops(CombinedIndexSorted(1:N_half), :);
        % 
        % % 更新popu的前一半个体
        % popu(IndexSorted(N_half+1:N), :) = newGoodPops;


        % if (old_gBest_fitness == gBest_fitness)
        %     num_not_update = num_not_update+1;
        % else
        %     old_gBest_fitness = gBest_fitness;
        % end
        
        %% 更新最佳值 
        if max(popuFitness) > optimal
            optimal = max(popuFitness);
        else
            count = count +1;%记录未改进的次数
        end
   
        %% Survival
    
        Fbest = gBest_fitness;
        BestChart(iter) = gBest_fitness / wf.power_total;
        BestFitness(iter) = gBest_fitness;
        [best_farmlayout,best_farmlayout_NA]  = gene_layout_by_indices_one(wf,gBest);
        farmlayout(iter,:) = best_farmlayout;
        farmlayout_NA(iter,:) = best_farmlayout_NA;
        if mod(iter, 50) == 0 || iter == iterations
            fprintf('NA %d Turbine Num:%d Wind %s run: %d iteration: %d  eta %f fitness %f CR %f F %f sit %f\n',NA_type,tn,wt,t,iter,(Fbest / wf.power_total),Fbest, CR, F, self_sy)
        end
    end
end

% NA 0 Turbine Num:30 Wind 3speed_12direction run: 1 iteration: 400  
% eta 0.979782 fitness 7872.867616 CR 0.100000 F 0.500000 sit 1.000000