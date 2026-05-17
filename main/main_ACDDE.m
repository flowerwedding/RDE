clear;
clc;
tic;
format long

%% 路径设置
problem_path = '../WindFarmOptimization/';
ws_path = '../WindFarmOptimization/windscenarios';
save_path = '../Results/';
addpath(problem_path)
addpath(ws_path)

%% 实验设置
runs = 3;  % 运行次数
popsize = 50;
max_it = 400;

%% 风电场参数
rows = 21;
cols = 21;
cell_width = 77.0 * 3;

%% 3种风机规模
turbine_num = [30];  % 风力发电机组的规模

NA_type_list = 0;

%% 4种风电场场景
n_speeds = [3, 3, 4, 6];      % 风速数量
n_directions = [12, 12, 12, 12];  % 风向数量
unifrom = [0, 1, 0, 0];       % 是否均匀分布

%% 创建结果目录
subfolder = 'ACDDE_Results';
mkdir(subfolder);
subfolderPath = fullfile(pwd, subfolder);

%% 主循环 - 遍历所有配置
for wt = 1:length(n_speeds)
    for tn = turbine_num
        for NA_type = NA_type_list
            % 生成禁止区域位置
            NA_loc_array = gene_NA_loc(NA_type);
            
            % 生成风电场结构
            [wf, ws_folder] = gene_windfram(rows, cols, tn, cell_width, NA_loc_array, ...
                n_speeds(wt), n_directions(wt), unifrom(wt));
            
            % 保存风电场结构
            save(fullfile(subfolderPath, 'wf.mat'), 'wf');
            
            % 为每个配置创建单独的结果目录
            config_folder = sprintf('%s/%s_tn%d_NA%d', subfolderPath, ws_folder, tn, NA_type);
            if ~exist(config_folder, 'dir')
                mkdir(config_folder)
            end
            
            % 初始化结果矩阵
            fbest = zeros(runs, 1);
            fbest(:) = inf;
            eta_best = zeros(runs, 1);
            Time = zeros(runs, 1);
            all_best_layouts = cell(runs, 1);
            
            % 生成结果文件名
            fname = fullfile(config_folder, sprintf('ACDDE_%s_tn%d_NA%d_results.txt', ws_folder, tn, NA_type));
            f_out = fopen(fname, 'wt');
            
            % 多次运行
            for run_id = 1:runs
                fprintf('Running: %s, Turbines: %d, NA: %d, Run: %d/%d\n', ...
                    ws_folder, tn, NA_type, run_id, runs);
                
                % 调用ACDDE算法
                tic;
                [gbest, gbestval, fitcount, RecordT, ~] = ACDDE(wf, max_it, config_folder, ws_folder, tn, NA_type, run_id);
                Time(run_id) = toc;
                
                % 计算转换效率
                eta = gbestval / wf.power_total;
                
                % 保存结果
                fbest(run_id) = gbestval;
                eta_best(run_id) = eta;
                
                % 生成最佳布局
                [best_farmlayout, best_farmlayout_NA] = gene_layout_by_indices_one(wf, gbest);
                all_best_layouts{run_id} = best_farmlayout;
                
                % 输出到文件
                fprintf(f_out, 'Run %d:\n', run_id);
                fprintf(f_out, 'Best Fitness: %.6f\n', gbestval);
                fprintf(f_out, 'Best Eta: %.6f\n', eta);
                fprintf(f_out, 'Time: %.6f seconds\n', RecordT);
                fprintf(f_out, 'Function Evaluations: %d\n', fitcount);
                fprintf(f_out, 'Best Layout Indices: %s\n\n', num2str(gbest));
                
                % 控制台输出
                fprintf('  Best Fitness: %.6f, Eta: %.6f, Time: %.2f seconds\n', ...
                    gbestval, eta, RecordT);
            end
            
            % 计算统计信息
            f_mean = mean(eta_best);
            f_median = median(eta_best);
            f_std = std(eta_best);
            f_best = max(eta_best);  % 注意：这里eta越大越好
            f_worst = min(eta_best);
            
            % 输出统计结果
            fprintf(f_out, '\n===== Statistical Results =====\n');
            fprintf(f_out, 'Best Eta: %.6f\n', f_best);
            fprintf(f_out, 'Worst Eta: %.6f\n', f_worst);
            fprintf(f_out, 'Mean Eta: %.6f\n', f_mean);
            fprintf(f_out, 'Median Eta: %.6f\n', f_median);
            fprintf(f_out, 'Std Eta: %.6f\n', f_std);
            
            % 控制台输出统计结果
            fprintf('\n===== Final Statistics =====\n');
            fprintf('Configuration: %s, Turbines: %d, NA: %d\n', ws_folder, tn, NA_type);
            fprintf('Best Eta: %.6f\n', f_best);
            fprintf('Worst Eta: %.6f\n', f_worst);
            fprintf('Mean Eta: %.6f\n', f_mean);
            fprintf('Median Eta: %.6f\n', f_median);
            fprintf('Std Eta: %.6f\n', f_std);
            fprintf('Average Time: %.2f seconds\n\n', mean(Time));
            
            fclose(f_out);
            
            % 保存所有运行的最佳布局
            save(fullfile(config_folder, 'best_layouts.mat'), 'all_best_layouts', 'eta_best', 'Time');
        end
    end
end

fprintf('All experiments completed!\n');
toc;