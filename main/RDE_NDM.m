function [Fbest,BestChart,BestFitness,farmlayout,farmlayout_NA]=RDE(popsize,wf,iterations,NA_type,tn,wt,t) %种群大小，风电场结构wf，最大迭代次数，禁用区域，风机数量tn，风电场标识wt，运行次数索引t
    BestChart=zeros(iterations,1);%最佳归一化适应度
    BestFitness=zeros(iterations,1);%最佳实际适应度
    farmlayout = zeros(iterations,wf.rows*wf.cols);%最佳风电场布局
    farmlayout_NA = zeros(iterations,wf.rows*wf.cols);%最佳NA的风电场布局
    
    % 风机数量 变量下界 变量上界(网格总单元数)
    D = wf.turbine_num; down = 1;up = wf.rows *wf.cols;
    
    % 种群初始化
    [popu,lu] = windfarm_init(popsize, wf.turbine_num,wf);
    % 计算适应度 种群评估
    [popuFitness,power_order,lp_power_accum]= wf_fitness(wf,popu);
    optimal = max(popuFitness);%记录当前最佳适应度

    %% 构建环形邻域结构（创建每个个体的邻居列表） TODO
    nsize_k = 5;                 % 邻域半径（可调参数）
    nsize = 2 * nsize_k + 1;     % 邻域大小
    nig = cell(popsize, 1);      % 邻域索引
    
    for i = 1:popsize
        % 添加左邻居
        if i - nsize_k < 1 % 左邻居不足
            nig{i, 1} = [1:i-1, popsize+i-nsize_k:popsize];
        else
            nig{i, 1} = i-nsize_k:i-1;
        end
        % 添加右邻居
        if i + nsize_k > popsize % 右邻居不足
            nig{i, 1} = [i+1:popsize, 1:nsize_k+i-popsize];
        else
            nig{i, 1} = i+1:i+nsize_k;
        end
        % 加上自己
        nig{i, 1} = [nig{i, 1}, i];
    end
    nig = cell2mat(nig);  % 转换为矩阵形式

   %% 局部参数设置
    pBest = popu;%个体历史最佳位置
    pBestFit = popuFitness;%个体历史最佳适应度
    [~, gBestId] = max(pBestFit);%全局最佳个体索引
    gBest = pBest(gBestId,:);%全局最佳位置
    gBest_fitness = pBestFit(gBestId);%全局最佳适应度

    count = 0;%计数器
    F = 0.5;%缩放因子
    CR = 0.05;%交叉概率 TODO F和CR下面会动态调整

    %% 主循环
    for iter = 1:iterations
        pjd = iter/iterations;%计算迭代进度比例
        p = 0.5;% + 0.3*(pjd);%动态调整参数
        self_sy = 2^exp(1-iterations/(iterations+1-iter));%自适应缩放因子F
        F = 0.5 * self_sy;
        CR = 0.1*pjd;%0.5;%0.001*self_sy;%动态调整交叉概率


        % 获取前一半
        [AllFitnessSorted,IndexSorted] = sort(popuFitness,'descend');%按适应度降序排序
        N = popsize;%种群大小
        N_half = round(N/2);%种群大小的一半
        index = floor(p*N_half);%计算分组索引 TODO
        gBest = popu(IndexSorted(1),:);
        pBest = popu(IndexSorted(1:10),:);%选择前10个作为个体最佳参考 TODO

        goodPops=popu(IndexSorted(1:N_half),:);%优秀个体
        goodPopsIdx = IndexSorted(1:N_half); % 优秀个体对应的原始编号
        worstPops=popu(IndexSorted(N_half+1:N),:);%较差个体
        worstPopsIdx = IndexSorted(N_half+1:N); % 较差个体对应的原始编号
        
        a = randperm(N_half);%优秀个体的随机排列
        b = randperm(N_half);%较差个体的随机排列
        c = randperm(N_half);
        d = randperm(N_half);


        gP = goodPops(a,:);%重新组织优秀个体
        gPIdx = goodPopsIdx(c); % 对应的原始编号
        gp1 = gP(1:index,:);  % 优秀个体的前index个
        gp1Idx = gPIdx(1:floor(0.95*N_half)); % 对应的原始编号
        gp2 = gP(index+1:N_half,:); % 优秀个体的后N_half-index个
        gp2Idx = gPIdx(index+1:N_half); % 对应的原始编号

        wP = worstPops(b,:);%重新组织较差个体
        wPIdx = worstPopsIdx(d); % 对应的原始编号
        wp1 = wP(1:N_half-index, :); % 较差个体的前N_half-index个
        wp1Idx = wPIdx(1:N_half-floor(0.05*N_half)); % 对应的原始编号
        wp2 = wP(N_half-index+1:N_half, :); % 较差个体的后index个
        wp2Idx = wPIdx(N_half-index+1:N_half); % 对应的原始编号

        V_pops = [gp1;wp1];
        V_popsIdx = [gp1Idx; wp1Idx]; % V_pops 对应的原始编号
        
        U = [gp2;wp2];
        UIdx = [gp2Idx; wp2Idx]; % U 对应的原始编号
         
        %% 邻域双重变异&划分双种群 TODO
        for i = 1:popsize
            if i <= N_half
                original_idx = V_popsIdx(i);
            else
                original_idx = UIdx(i - N_half);
            end
                        
            % 获取当前个体的邻域
            current_nig = nig(original_idx, :);
            nsize_current = length(current_nig);
            
            % 构建邻域种群（位置+适应度）
            neighbor_positions = zeros(nsize_current, D);
            neighbor_fitness = zeros(nsize_current, 1);
            for j = 1:nsize_current
                neighbor_idx = current_nig(j);
                neighbor_positions(j, :) = popu(neighbor_idx, :);
                neighbor_fitness(j) = popuFitness(neighbor_idx);
            end
            
            % 按适应度降序排序邻居
            [sorted_fitness, sort_idx] = sort(neighbor_fitness, 'descend');
            sorted_neighbors = neighbor_positions(sort_idx, :);
            % 找到邻域最优个体（适应度最高的邻居）
            nbest = sorted_neighbors(1, :);

            % 计算当前个体在邻域中的等级
            current_fitness = popuFitness(original_idx);%按照这个划分优秀个体/差个体感觉不合适
            rank_i = find(sorted_fitness == current_fitness, 1);
            
            % 计算决策空间距离找到最近个体
            current_individual = popu(original_idx, :);
            distances = sqrt(sum((sorted_neighbors(:, 1:D) - repmat(current_individual, size(sorted_neighbors, 1), 1)).^2, 2));
            [sorted_dist, dist_idx] = sort(distances);
            nearest = sorted_neighbors(dist_idx(2), 1:D);  % 最近的个体（排除自己）
            
            % 计算自适应变异因子
            max_rank = nsize_current;
            if max_rank == 0
                max_rank = 1;
            end
            F_i = (rank_i/max_rank) + 0.1 * randn();
            F_i = min(max(F_i, 0.1), 0.9);
            
            % 双重变异策略选择
            if popuFitness(original_idx) >= mean(popuFitness)  % 优秀个体
                % DE/ilbest/2: 利用邻域最优个体进行开发
                if size(sorted_neighbors, 1) >= 5
                    r_indices = randperm(size(sorted_neighbors, 1), 4);
                    if i <= N_half
                        V_pops(i, :) = nbest + F_i * (sorted_neighbors(r_indices(1), 1:D) - sorted_neighbors(r_indices(2), 1:D) + ...
                                             sorted_neighbors(r_indices(3), 1:D) - sorted_neighbors(r_indices(4), 1:D));
                    else
                        U(i - N_half, :) = nbest + F_i * (sorted_neighbors(r_indices(1), 1:D) - sorted_neighbors(r_indices(2), 1:D) + ...
                                             sorted_neighbors(r_indices(3), 1:D) - sorted_neighbors(r_indices(4), 1:D));
                    end
                else
                    % 备用变异策略
                    if i <= N_half
                        V_pops(i, :) = nbest + F_i * (randn(1, D));
                    else
                        U(i - N_half, :) = nbest + F_i * (randn(1, D));
                    end
                end
            else  % 较差个体
                % DE/linrand/2: 利用最近个体进行探索
                if size(sorted_neighbors, 1) >= 5
                    r_indices = randperm(size(sorted_neighbors, 1), 4);
                    if i <= N_half
                        V_pops(i, :) = nearest + F_i * (sorted_neighbors(r_indices(1), 1:D) - sorted_neighbors(r_indices(2), 1:D) + ...
                                               sorted_neighbors(r_indices(3), 1:D) - sorted_neighbors(r_indices(4), 1:D));
                    else
                        U(i - N_half, :) = nearest + F_i * (sorted_neighbors(r_indices(1), 1:D) - sorted_neighbors(r_indices(2), 1:D) + ...
                                               sorted_neighbors(r_indices(3), 1:D) - sorted_neighbors(r_indices(4), 1:D));
                    end
                else
                    % 备用变异策略
                    if i <= N_half
                        V_pops(i, :) = nearest + F_i * (randn(1, D));
                    else
                        U(i - N_half, :) = nearest + F_i * (randn(1, D));
                    end
                end
            end
        end

        %% 变异和交叉
        [r1, r2, r3] = getindex(N_half); % 从1到N_half中随机选择三个索引
        V = V_pops(r1, :) + F * (V_pops(r2, :)  - U(r3, :));

        %% 二项式交叉
        for i = 1:N_half
            j_rand = floor(rand * D) + 1;%确保至少有一个维度交叉
            tt = rand(1, D) < CR;%创建交叉掩码
            tt(1, j_rand) = 1;%确保第j_rand个维度一定会交叉
            t_ = 1 - tt;%互补掩码
            U(i, :) = tt .* V(i, :) + t_ .* pBest(floor(rand*5)+1, :);%V_pops(i, :);%交叉
        end
        
       %% 边界检测和处理
        U = windfarm_constraint(U, wf.NA_loc, D,down,up);%确保解在可行域内
        [fit_U,power_order,lp_power_accum] = wf_fitness(wf, U);%计算新个体的适应度
        
        % 精英保留 TODO
        for i = 1:N_half
            if fit_U(i) > gBest_fitness%更新全局最佳适应度
                gBest_fitness = fit_U(i);
            end
            % popu_num = IndexSorted(N_half+i); % 差个体的编号->替换差个体 TODO
            popu_num = IndexSorted(i); % 优秀个体的编号->替换好个体
            if fit_U(i, :) >= popuFitness(popu_num, :)%新个体更好，替换原有个体，选择
                popu(popu_num, :) = U(i, :);
                popuFitness(popu_num, :) = fit_U(i, :);
            end
        end
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
        fprintf('NA %d Turbine Num:%d Wind %s run: %d iteration: %d  eta %f fitness %f CR %f F %f sit %f\n',NA_type,tn,wt,t,iter,(Fbest / wf.power_total),Fbest, CR, F, self_sy)
    end

end

