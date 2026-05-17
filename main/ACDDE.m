function [gbest, gbestval, fitcount, RecordT, cSTART] = ACDDE(wf, Max_nfe, config_folder, ws_folder, tn, NA_type, run_id)
    % ACDDE算法适配风电场优化问题
    % 输入参数:
    %   wf - 风电场结构
    %   Max_nfe - 最大函数评估次数
    %   config_folder - 结果保存目录
    %   ws_folder - 风场景文件夹名
    %   tn - 风机数量
    %   NA_type - 禁止区域类型
    %   run_id - 运行ID
    
    stm = RandStream('swb2712', 'Seed', sum(100*clock));
    RandStream.setGlobalStream(stm);
    
    cSTART = 0;
    D = wf.turbine_num;  % 变量维度 = 风机数量
    ps_ini = 100;
    ps_min = 5;
    ps = ps_ini;
    Max_nfe = Max_nfe* 50;
    
    % 边界设置
    down = 1;
    up = wf.rows * wf.cols;
    Rmin = repmat(down, 1, D);
    Rmax = repmat(up, 1, D);
    
    % 初始化种群 - 生成整数索引
    pos = zeros(ps, D);
    for i = 1:ps
        pos(i, :) = randperm(up, D);  % 生成不重复的整数索引
    end
    
    % 添加簇标签
    pos = [pos, -1 * ones(ps, 1)];
    
    % 给种群设置counter标签位
    counter = zeros(ps, 1);
    pivot = [2/3 * Max_nfe, 1/3 * ps_ini];
    
    % 初始评估
    pastval = zeros(ps, 1);
    for i = 1:ps
        [pastval(i), ~, ~] = wf_fitness(wf, pos(i, 1:D));
    end
    
    [pastval, indexSel] = sort(pastval, 'descend');  % 风电场问题适应度越大越好
    gbestval = pastval(1);
    gbest = pos(indexSel(1), 1:D);
    fitcount = ps;
    
    % 参数初始化
    memory_size = 6; 
    memory_order = 1;
    memory_MUF = 0.5 * ones(memory_size, 1); 
    memory_MUCr = 0.9 * ones(memory_size, 1);
    memory_MUF_elite = 0.5 * ones(memory_size, 1); 
    memory_MUCr_elite = 0.9 * ones(memory_size, 1);
    Archfactor = 1.05 + abs(D - 30) * 11/400;
    
    pbest_rate = 0.15;
    decayA = [];
    T0 = 1.0;
    A = [];
    A_eval = [];
    gen = 1;
        
    % 计算初始多样性
    Mean1 = mean(pos(:, 1:D));
    DI_ini = sum(sqrt(sum((pos(:, 1:D) - Mean1).^2, 2)));
    RDI = DI_ini;
    t1 = repmat(0.9, ps, 1);
    
    % 聚类设置
    ParameterCase;  % 加载参数配置
    EliteLayering = true;
    case_idx = 0;
    currentParameterCase = eval(['ParameterCase', num2str(case_idx)]);
    
    if fitcount < ceil(pivot(1))
        num_clusters = currentParameterCase(1, 1);
        Kmeans = currentParameterCase(1, 2);
    else
        num_clusters = currentParameterCase(2, 1);
        Kmeans = currentParameterCase(2, 2);
    end
    
    % 详细结果记录
    name = fullfile(config_folder, sprintf('ACDDE_%s_tn%d_NA%d_run%d.dat', ws_folder, tn, NA_type, run_id));
    fout = fopen(name, 'wt');
    fprintf(fout, '%d\t%.15f\t%.15f\n', 1, gbestval, gbestval/wf.power_total);
    
    already_clustered = false;
    
    tic;
    
    %% 主进化循环
    while fitcount < Max_nfe
        
        %% 聚类
        if gen == 1 || (fitcount > 2/3 * Max_nfe && ~already_clustered)
            [cluster_idx, cluster_centers] = my_kmeans(pos(:, 1:end-1), num_clusters, 50);
            pos(:, end) = cluster_idx;
            
            if gen == 1
                gbest_cluster = pos(indexSel(1), end);
            end
            
            if fitcount > 2/3 * Max_nfe
                already_clustered = true;
            end
        end
        
        % 种群大小调整
        if gen > 1
            [~, indexSel] = sort(pastval, 'descend');  % 降序排序
            gbest = pos(indexSel(1), 1:D);
            gbestval = pastval(indexSel(1));
            gbest_cluster = pos(indexSel(1), end);
            
            selected_indices = indexSel(1:ps);
            pos = pos(selected_indices, :);
            pastval = pastval(selected_indices);
            counter = counter(selected_indices);
            Cr = []; F = []; Cr_elite = []; F_elite = [];
        end
        
        %% pbest集合
        pNP = max(round(pbest_rate * ps), 2);
        
        % 在gbest所在的簇中选择适应度最高的邻居
        indices_in_gbest_cluster = find(pos(:, end) == gbest_cluster);
        fitness_values_of_gbest_cluster = pastval(indices_in_gbest_cluster);
        [~, sorted_indices] = sort(fitness_values_of_gbest_cluster, 'descend');
        
        half_length = round(length(sorted_indices) * 0.10);
        selected_indices = indices_in_gbest_cluster(sorted_indices(1:half_length));
        
        next_index = 2;
        while half_length < pNP
            if ~ismember(next_index, selected_indices)
                selected_indices = [selected_indices; next_index];
                next_index = next_index + 1;
                half_length = half_length + 1;
            else
                next_index = next_index + 1;
            end
        end
        
        Restart_val = pastval(pNP);
        Cpnei0 = pos(selected_indices, :);
        lenSel = max(round(pNP * rand(1, ps)), 1);
        Cpnei = Cpnei0(lenSel, :);
        
        %% A集合
        unique_clusters = unique(pos(:, end));
        A = [];
        
        for cluster_label = unique_clusters'
            indices_in_cluster = find(pos(:, end) == cluster_label);
            num_to_select = ceil(Kmeans * length(indices_in_cluster));
            
            if num_to_select > 0 && num_to_select <= length(indices_in_cluster)
                random_selection = indices_in_cluster(randperm(length(indices_in_cluster), num_to_select));
                ADD_A = pos(random_selection, :);
                A = [A; ADD_A];
            end
        end
        
        [~, index] = unique(A, 'rows', 'stable');
        A = A(index, :);
        psExt = size(A, 1);
        
        %% 随机选择个体
        rndBase = randperm(ps)';
        rndSeq1 = ceil(rand(ps, 1) * psExt);
        
        for ii = 1:ps
            while rndBase(ii) == ii
                rndBase(ii) = ceil(rand() * ps);
            end
            while rndSeq1(ii) == ii
                rndSeq1(ii) = ceil(rand() * psExt);
            end
        end
        
        posr = pos(rndBase, :);
        posxr = A(rndSeq1, :);
        
        %% 生成F和Cr参数
        memory_rand_index1 = ceil(memory_size * rand(ps, 1));
        MUF = memory_MUF(memory_rand_index1);
        MUCr = memory_MUCr(memory_rand_index1);
        
        % 生成Cr
        rand1 = rand(ps, 1);
        rand2 = rand(ps, 1);
        Cr = zeros(ps, 1);
        
        for ii = 1:ps
            if rand1(ii) < t1(ii)
                Cr(ii) = MUCr(ii) + 0.1 * randn();
            else
                Cr(ii) = 0.9;
            end
            if MUCr(ii) == -1
                Cr(ii) = 0;
            end
        end
        
        Cr = min(Cr, 1);
        Cr = max(Cr, 0);
        
        % 生成F
        F = zeros(ps, 1);
        for ii = 1:ps
            F(ii, 1) = randCauchy(MUF(ii), 0.1);
        end
        
        label_sf = find(F <= 0);
        while ~isempty(label_sf)
            F(label_sf) = MUF(label_sf) + 0.1 * tan(pi * (rand(length(label_sf), 1) - 0.5));
            label_sf = find(F <= 0);
        end
        
        F = min(F, 1);
        
        % 生成交叉掩码
        label = zeros(ps, D);
        rndVal = rand(ps, D);
        onemat = zeros(ps, D);
        
        for ii = 1:ps
            label(ii, :) = rndVal(ii, :) <= Cr(ii);
            indexJ = ceil(rand() * D);
            onemat(ii, indexJ) = 1;
        end
        
        label = label | onemat;
        
        %% 变异策略 - 修改为整数操作
        copy_pos = pos;
        
        for ii = 1:ps
            % 对每个个体执行变异
            new_indices = copy_pos(ii, 1:D);
            
            % 确定哪些维度需要变异
            mutate_mask = label(ii, :);
            
            if any(mutate_mask)
                % 获取变异参考个体
                ref_indices = Cpnei(ii, 1:D);
                ref_r = posr(ii, 1:D);
                ref_xr = posxr(ii, 1:D);
                
                % 对于需要变异的维度
                for d = find(mutate_mask)
                    % 计算连续的变异值
                    offset = F(ii) * (ref_indices(d) - new_indices(d)) + ...
                            F(ii) * (ref_r(d) - ref_xr(d));
                    
                    % 转换为整数索引
                    new_val = round(new_indices(d) + offset);
                    
                    % 确保在边界内
                    new_val = max(min(new_val, up), down);
                    new_indices(d) = new_val;
                end
                
                % 确保索引唯一性（风电场中风机位置不能重复）
                new_indices = unique_indices(new_indices, up, D);
                
                % 确保没有禁用区域
                new_indices = remove_na_indices(new_indices, wf.NA_loc);
                
                % 如果移除了某些索引，补充新的随机位置
                if length(new_indices) < D
                    new_indices = complete_indices(new_indices, D, up, wf.NA_loc);
                end
                
                pos(ii, 1:D) = new_indices;
            end
        end
        
        % 评估新种群
        posval = zeros(ps, 1);
        for ii = 1:ps
            [posval(ii), ~, ~] = wf_fitness(wf, pos(ii, 1:D));
        end
        
        % 选择操作
        [posval, I] = max([posval, pastval], [], 2);
        better_indices = I == 1;
        worse_indices = I == 2;
        
        pos(worse_indices, :) = copy_pos(worse_indices, :);
        counter(worse_indices) = counter(worse_indices) + 1;
        counter(better_indices) = 0;
        
        % 计算种群多样性
        Meanpos = mean(pos(:, 1:D));
        DI = sqrt(sum(sum((pos(:, 1:D) - Meanpos).^2, 2)));
        RDI1 = DI / DI_ini;
        RDI = [RDI; RDI1];
        fitcount = fitcount + ps;
        
        % 更新A集合
        len = size(A, 1);
        if len > round(Archfactor * ps)
            rndSel = randperm(len)';
            rndSel = rndSel(round(Archfactor * ps) + 1:len);
            A(rndSel, :) = [];
        end
        
        %% 参数自适应更新
        SuccF = F(better_indices);
        SuccCr = Cr(better_indices);
        
        if ~isempty(SuccF)
            num_Succ = length(SuccCr);
            if num_Succ > 0
                % 简化的参数更新策略
                memory_MUF(memory_order) = mean(SuccF);
                if max(SuccCr) == 0 || memory_MUCr(memory_order) == -1
                    memory_MUCr(memory_order) = -1;
                else
                    memory_MUCr(memory_order) = mean(SuccCr);
                end
            end
            
            memory_order = memory_order + 1;
            if memory_order > memory_size
                memory_order = 1;
            end
        end
        
        %% 更新全局最优
        pastval = posval;
        [gbestval, gbestid] = max(pastval);
        gbest = pos(gbestid, 1:D);
        
        % 记录结果
        if mod(fitcount - ps, 100*5*D) >= 100*5*D - ps && mod(fitcount, 100*5*D) < ps
            fprintf(fout, '%d\t%.15f\t%.15f\n', fitcount, gbestval, gbestval/wf.power_total);
        end
        
        %% 重启机制
        V_STD = std(pos(:, 1:D));
        [~, v_label] = sort(V_STD, 'descend');
        v_label = v_label(1:ceil(0.3 * D));
        
        if RDI1 < 0.01
            for i = 1:length(counter)
                if counter(i) >= D && i ~= gbestid && pastval(i) > Restart_val
                    % 部分维度重启
                    new_indices = pos(i, 1:D);
                    ref_indices = Cpnei(i, 1:D);
                    ref_r = posr(i, 1:D);
                    ref_xr = posxr(i, 1:D);
                    
                    for d = v_label
                        offset = F(i) * (ref_indices(d) - new_indices(d)) + ...
                                F(i) * (ref_r(d) - ref_xr(d));
                        new_val = round(new_indices(d) + offset);
                        new_val = max(min(new_val, up), down);
                        new_indices(d) = new_val;
                    end
                    
                    new_indices = unique_indices(new_indices, up, D);
                    new_indices = remove_na_indices(new_indices, wf.NA_loc);
                    
                    if length(new_indices) < D
                        new_indices = complete_indices(new_indices, D, up, wf.NA_loc);
                    end
                    
                    pos(i, 1:D) = new_indices;
                    
                elseif counter(i) >= D + 15 && i ~= gbestid
                    % 完全重启
                    new_indices = zeros(1, D);
                    for d = 1:D
                        new_val = round(gbest(d) + F(i) * (posr(i, d) - posxr(i, d)));
                        new_val = max(min(new_val, up), down);
                        new_indices(d) = new_val;
                    end
                    
                    new_indices = unique_indices(new_indices, up, D);
                    new_indices = remove_na_indices(new_indices, wf.NA_loc);
                    
                    if length(new_indices) < D
                        new_indices = complete_indices(new_indices, D, up, wf.NA_loc);
                    end
                    
                    pos(i, 1:D) = new_indices;
                    [pastval(i), ~, ~] = wf_fitness(wf, pos(i, 1:D));
                    counter(i) = 0;
                    fitcount = fitcount + 1;
                    cSTART = cSTART + 1;
                end
            end
        end
        
        %% 种群递减策略
        if fitcount < 1/3 * Max_nfe
            plan_ps = ceil((pivot(2) - ps_ini) / (pivot(1) - ps_ini)^2 * (fitcount - ps_ini)^2 + ps_ini);
        elseif fitcount < ceil(pivot(1))
            plan_ps = ceil((pivot(2) - ps_ini) / (pivot(1) - ps_ini)^2 * (fitcount - ps_ini)^2 + ps_ini);
        else
            plan_ps = floor((pivot(2) - ps_ini) / (pivot(1) - ps_ini) * (fitcount - Max_nfe) + ps_min);
        end
        
        if ps > plan_ps
            if plan_ps < ps_min
                ps = ps_min;
            else
                ps = plan_ps;
            end
        end
        
        pbest_rate = 0.15 + 0.3 * fitcount / Max_nfe;
        
        

        %% 输出
        if mod(fitcount,250) == 0 || fitcount == Max_nfe
            current_eta = gbestval / wf.power_total;
            fprintf('FES: %d/%d, 适应度: %.2f, eta: %.6f\n', fitcount, Max_nfe, gbestval, current_eta);
        end
    end
    
    RecordT = toc;
    
    % 记录最终结果
    fprintf(fout, '%d\t%.15f\t%.15f\n', fitcount, gbestval, gbestval/wf.power_total);
    fclose(fout);
end

%% 辅助函数
function result = randCauchy(mu, sigma)
    [m, n] = size(mu);
    result = mu + sigma * tan(pi * (rand(m, n) - 0.5));
end

function [cluster_idx, cluster_centers] = my_kmeans(data, k, max_iter)
    if nargin < 3
        max_iter = 50;
    end
    
    [N, D] = size(data);
    
    % 随机初始化聚类中心
    rnd_idx = randperm(N, k);
    cluster_centers = data(rnd_idx, :);
    
    % K-means迭代
    for iter = 1:max_iter
        % 分配每个点到最近的聚类中心
        distances = zeros(N, k);
        for i = 1:k
            diff = bsxfun(@minus, data, cluster_centers(i, :));
            distances(:, i) = sum(diff.^2, 2);
        end
        [~, cluster_idx] = min(distances, [], 2);
        
        % 更新聚类中心
        old_centers = cluster_centers;
        for i = 1:k
            points_in_cluster = data(cluster_idx == i, :);
            if size(points_in_cluster, 1) > 0
                cluster_centers(i, :) = mean(points_in_cluster, 1);
            else
                cluster_centers(i, :) = data(randi(N), :);
            end
        end
        
        % 检查收敛
        if max(abs(cluster_centers(:) - old_centers(:))) < 1e-6
            break;
        end
    end
end

function indices = unique_indices(indices, up, D)
    % 确保索引唯一性
    unique_indices = unique(indices);
    
    if length(unique_indices) < D
        % 如果有重复，用随机值替换重复项
        all_indices = 1:up;
        available = setdiff(all_indices, unique_indices);
        
        % 随机选择补充索引
        if length(available) >= (D - length(unique_indices))
            additional = randperm(length(available), D - length(unique_indices));
            indices = [unique_indices, available(additional)];
        else
            % 如果可用位置不足，使用随机位置（允许重复）
            indices = randperm(up, D);
        end
    end
end

function indices = remove_na_indices(indices, NA_loc)
    % 移除禁用区域的索引
    if ~isempty(NA_loc)
        na_indices = NA_loc(:, 1) * 21 + NA_loc(:, 2) + 1;  % 转换为线性索引
        indices = setdiff(indices, na_indices);
    end
end

function indices = complete_indices(indices, D, up, NA_loc)
    % 补充缺失的索引
    current_count = length(indices);
    
    if current_count < D
        % 生成所有可能的索引
        all_indices = 1:up;
        
        % 移除禁用区域
        if ~isempty(NA_loc)
            na_indices = NA_loc(:, 1) * 21 + NA_loc(:, 2) + 1;
            all_indices = setdiff(all_indices, na_indices);
        end
        
        % 移除已存在的索引
        available = setdiff(all_indices, indices);
        
        % 随机选择补充索引
        if length(available) >= (D - current_count)
            additional = randperm(length(available), D - current_count);
            indices = [indices, available(additional)];
        else
            % 如果可用位置不足，使用随机位置（允许重复）
            additional = randperm(up, D - current_count);
            indices = [indices, additional];
        end
    end
    
    % 确保长度正确
    indices = indices(1:D);
end