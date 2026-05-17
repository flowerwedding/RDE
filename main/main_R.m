clear;
clc;
tic;
format long
%% path setup
problem_path = '../WindFarmOptimization/';
ws_path =  '../WindFarmOptimization/windscenarios';
save_path = '../Results/';
addpath(problem_path)
addpath(ws_path)

%% experiment setup
runTime = 1;
popsize = 50;
max_it  = 400;

%% wind farm parameters
rows = 21;
cols = 21;
cell_width = 77.0 * 3;
%% 3种风机规模
%turbine_num = [30,35,40];
turbine_num = [30];
NA_type_list = 0;

%% 4种风电场场景
%n_speeds = [3,3,4,6];
%n_directions = [12,12,12,12];
%unifrom = [0,1,0,0];
n_speeds = [6];
n_directions = [12];
unifrom = [0];

%% ========== 从RPSO中整合的参数 ==========
% 混沌参数
Chaos_p = GenerateChaos(max_it);  % 生成混沌序列

% 策略参数
Strategy = 3;                      % 策略类型

% 聚类和交叉参数
cls_type = 3;                      % 聚类类型: 1:individual 2d; 2:population 2D; 3:population 1D
c_r_type = 3;                      % 交叉类型: 1:x 2:y 3:x&y
rad_value = 0.01;                   % 半径值

% 记录r1r2参数的数组（RPSO特有）
rr = [];                            % 将在循环中初始化

%% Para
% 并行计算设置
delete(gcp('nocreate'))
mycluster = parcluster('local');
mycluster.NumWorkers = 20;

for rad_ind = 1:length(rad_value)
    total_st = tic;
    rad_value_temp = rad_value(rad_ind);
    
    % 生成算法目录名称（整合了RPSO的参数）
    algorithmDir = sprintf('20260424CGPSO');

    % 创建保存目录和文件
    text_path = sprintf('%s/%s/', save_path, algorithmDir);
    text_file = sprintf('%s/cost_time.txt', text_path);
    if ~exist(text_path, 'dir')
        mkdir(text_path)
    end
    f = fopen(text_file, 'a');
    fprintf(f, datestr(now));
    fprintf(f, '\n');
    fprintf(f, 'Parameters: Strategy=%d, rad=%.2f, cls=%d, cr=%d\n', ...
        Strategy, rad_value_temp, cls_type, c_r_type);

    for wt = 1:length(n_speeds)
        for tn = turbine_num
            for NA_type = NA_type_list
                NA_loc_array = gene_NA_loc(NA_type);
                [wf, ws_folder] = gene_windfram(rows, cols, tn, cell_width, ...
                    NA_loc_array, n_speeds(wt), n_directions(wt), unifrom(wt));
                save('wf.mat', 'wf');
                
                folder = sprintf('%s/%s/%s/tn%d_NA%d', save_path, ...
                    algorithmDir, ws_folder, tn, NA_type);

                if ~exist(folder, 'dir')
                    mkdir(folder)
                end
                
                % 初始化记录数组
                eta = zeros(max_it, runTime);      % 转换效率
                fitness = zeros(max_it, runTime);   % 适应度
                rr = zeros(runTime, max_it * 2);    % r1r2参数记录（RPSO特有）

                farmlayout = zeros(runTime, max_it, wf.cols * wf.rows);
                farmlayout_NA = zeros(runTime, max_it, wf.cols * wf.rows);
                
                fprintf('%s - %s TN %d\n', algorithmDir, ws_folder, tn)
                
                for t = 1:runTime
                    if t == 30
                        fprintf('\n')
                    end
                    st = tic;
                    
                    % 调用RPSO算法，传入所有参数
                    %[Bestr1r2_temp, Fbest, BestChart, BestFitness, farmlayout_temp, farmlayout_NA_temp] = ...
                    %        HPSO2(popsize, wf, max_it, NA_type, tn, ws_folder, t, ...
                    %            Chaos_p, Strategy, rad_value_temp, cls_type, c_r_type);
                    [Bestr1r2_temp, Fbest, BestChart, BestFitness, farmlayout_temp, farmlayout_NA_temp] = ...
                            RPSO(popsize, wf, max_it, NA_type, tn, ws_folder, t, ...
                                Chaos_p, Strategy, rad_value_temp, cls_type, c_r_type);
                    
                    end_t = toc(st);
                    cost_t = seconds(end_t);
                    cost_t.Format = 'hh:mm:ss';
                    
                    % 正确赋值
                    % Bestr1r2_temp 是 [max_it, 2] 的矩阵，需要转换为一行
                    rr(t, :) = reshape(Bestr1r2_temp', 1, []);
                    
                    % BestChart 和 BestFitness 应该是 [max_it, 1] 的向量
                    eta(:, t) = BestChart;
                    fitness(:, t) = BestFitness;
                    
                    % farmlayout_temp 和 farmlayout_NA_temp 应该是 [max_it, wf.rows*wf.cols] 的矩阵
                    farmlayout(t, :, :) = farmlayout_temp;
                    farmlayout_NA(t, :, :) = farmlayout_NA_temp;
                    
                    % 保存结果
                    save_results(reshape(farmlayout(t,:,:), ...
                        size(farmlayout(t,:,:),2), size(farmlayout_NA(t,:,:),3)), ...
                        reshape(farmlayout_NA(t,:,:), ...
                        size(farmlayout_NA(t,:,:),2), size(farmlayout(t,:,:),3)), t, folder)
                    
                    save(sprintf('%s/eta.mat', folder), "eta")
                    save(sprintf('%s/fitness.mat', folder), "fitness")
                    save(sprintf('%s/rr.mat', folder), "rr")
                    
                    fprintf('\n%s - %s TN %d Cost Times %s\n', ...
                        algorithmDir, ws_folder, tn, cost_t)
                    fprintf(f, '%s - %s TN %d Cost Times %s\n', ...
                        algorithmDir, ws_folder, tn, cost_t);
                end
                
                % 输出统计信息
                fprintf("min eta = %f\n", min(eta(end, :)));
                fprintf("max eta = %f\n", max(eta(end, :)));
                fprintf("median eta = %f\n", median(eta(end, :)));
                fprintf("mean eta = %f\n", mean(eta(end, :)));
                fprintf("std eta = %f\n", std(eta(end, :)));
                end
        end
    end
    
    %% 记录总时间
    total_cost = toc(total_st);
    total_cost_t = seconds(total_cost);
    total_cost_t.Format = 'hh:mm:ss';
    fprintf(f, 'Total cost: %s\n', total_cost_t);
    fclose(f);
end