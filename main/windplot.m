%% 从保存的结果中读取并绘制风况场景玫瑰图
clear; clc; close all;

%% 获取桌面保存路径（与之前的可视化保持一致）
if ispc
    desktop_path = [getenv('USERPROFILE'), '\Desktop\'];
else
    desktop_path = [getenv('HOME'), '/Desktop/'];
end

save_folder = [desktop_path, 'WindScenarios_Visualization/'];
if ~exist(save_folder, 'dir')
    mkdir(save_folder);
end

%% 风场景参数配置（与你的代码保持一致）
n_speeds = [3, 3, 4, 6];      % 风速数量
n_directions = [12, 12, 12, 12]; % 风向数量（12个方向）
uniform = [0, 1, 0, 0];       % 是否均匀分布: S1非均匀, S2均匀, S3非均匀, S4非均匀

% 风速值设置（根据论文）
speeds_list = {[4, 8, 12], [4, 8, 12], [4, 8, 12, 16], [4, 7, 10, 13, 16, 19]};

% 风向（12个方向，每个间隔30度）
directions = 0:30:330;
dir_labels_16 = {'N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', ...
                 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'};

% 场景名称
scenario_names = {'S1', 'S2', 'S3', 'S4'};
scenario_titles = {
    '3speed-12direction',
    '3speed-12direction-uniform',
    '4speed-12direction',
    '6speed-12direction'
};

%% 生成各场景的概率分布（模拟 gene_windfram 函数的行为）
fprintf('正在生成四种风场景的概率分布...\n');

for wt = 1:length(n_speeds)
    % 调用风场景生成函数（模拟 gene_windfram）
    [probs, wind_speeds, wind_directions] = generate_wind_scenario(...
        n_speeds(wt), n_directions(wt), uniform(wt), speeds_list{wt});
    
    % 保存概率分布
    scenarios(wt).probs = probs;
    scenarios(wt).speeds = wind_speeds;
    scenarios(wt).directions = wind_directions;
    scenarios(wt).name = scenario_names{wt};
    scenarios(wt).title = scenario_titles{wt};
    
    fprintf('  %s: 生成完成 (%d种风速, %d个风向, %s分布)\n', ...
        scenario_names{wt}, n_speeds(wt), n_directions(wt), ...
        iff(uniform(wt) == 1, '均匀', '非均匀'));
end

%% 绘制风玫瑰图（四合一）
f1 = figure('Position', [100, 100, 1200, 1000], 'Color', 'white');

for wt = 1:4
    subplot(2, 2, wt);
    plot_wind_rose(scenarios(wt).probs, scenarios(wt).speeds, ...
                   scenarios(wt).directions, scenarios(wt).title);
end

sgtitle('四种风电场风况场景 (Wind Scenarios)', 'FontSize', 16, 'FontWeight', 'bold');

% 保存PDF
exportgraphics(f1, [save_folder, 'Figure_WindRose_4in1.pdf'], 'ContentType', 'vector');
fprintf('\n已保存: %s\n', [save_folder, 'Figure_WindRose_4in1.pdf']);

%% 绘制风速-风向联合分布热力图（四合一）
f2 = figure('Position', [100, 100, 1400, 1000], 'Color', 'white');

for wt = 1:4
    subplot(2, 2, wt);
    plot_heatmap(scenarios(wt).probs, scenarios(wt).speeds, ...
                 scenarios(wt).directions, scenarios(wt).title);
end

% 保存PDF
exportgraphics(f2, [save_folder, 'Figure_Heatmap_4in1.pdf'], 'ContentType', 'vector');
fprintf('已保存: %s\n', [save_folder, 'Figure_Heatmap_4in1.pdf']);

%% 保存场景信息文本文件
fid = fopen([save_folder, 'Scenario_Info.txt'], 'w');
fprintf(fid, '========================================================\n');
fprintf(fid, '四种风况场景详细参数\n');
fprintf(fid, '========================================================\n');
fprintf(fid, '\n场景\t风速数量\t风向数量\t分布类型\n');
fprintf(fid, '--------------------------------------------------------\n');

for wt = 1:4
    fprintf(fid, '%s\t%d\t\t%d\t\t%s\n', ...
        scenario_names{wt}, n_speeds(wt), n_directions(wt), ...
        iff(uniform(wt) == 1, '均匀分布', '非均匀分布'));
end

fprintf(fid, '\n风速设置:\n');
fprintf(fid, '  S1/S2: ');
for i = 1:length(speeds_list{1})
    fprintf(fid, '%d m/s%s', speeds_list{1}(i), iff(i<length(speeds_list{1}), ', ', '\n'));
end
fprintf(fid, '  S3:    ');
for i = 1:length(speeds_list{3})
    fprintf(fid, '%d m/s%s', speeds_list{3}(i), iff(i<length(speeds_list{3}), ', ', '\n'));
end
fprintf(fid, '  S4:    ');
for i = 1:length(speeds_list{4})
    fprintf(fid, '%d m/s%s', speeds_list{4}(i), iff(i<length(speeds_list{4}), ', ', '\n'));
end

fclose(fid);
fprintf('\n已保存: Scenario_Info.txt\n');

%% 打印到命令窗口
fprintf('\n========================================================\n');
fprintf('四种风况场景详细参数\n');
fprintf('========================================================\n');
fprintf('\n场景\t风速数量\t风向数量\t分布类型\n');
fprintf('--------------------------------------------------------\n');
for wt = 1:4
    fprintf('%s\t%d\t\t%d\t\t%s\n', ...
        scenario_names{wt}, n_speeds(wt), n_directions(wt), ...
        iff(uniform(wt) == 1, '均匀分布', '非均匀分布'));
end

fprintf('\n所有图片已保存到: %s\n', save_folder);

%% ==================== 子函数定义 ====================

function [probs, speeds, directions] = generate_wind_scenario(n_speeds, n_directions, is_uniform, speeds_list)
    % 模拟 gene_windfram 函数的风场景生成
    % 输入:
    %   n_speeds: 风速数量
    %   n_directions: 风向数量（12）
    %   is_uniform: 是否均匀分布（1:均匀, 0:非均匀）
    %   speeds_list: 风速值列表
    % 输出:
    %   probs: 概率分布矩阵 [n_directions × n_speeds]
    %   speeds: 风速值数组
    %   directions: 风向角度数组
    
    directions = 0:(360/n_directions):(360-360/n_directions);
    speeds = speeds_list;
    n_dirs = length(directions);
    n_spds = length(speeds);
    
    probs = zeros(n_dirs, n_spds);
    
    if is_uniform == 1
        % 均匀分布
        % 风速分布：低风速25%，中风速50%，高风速25%
        for i = 1:n_dirs
            if n_spds == 3
                probs(i, 1) = 0.25 / n_dirs;  % 低风速
                probs(i, 2) = 0.50 / n_dirs;  % 中风速
                probs(i, 3) = 0.25 / n_dirs;  % 高风速
            elseif n_spds == 4
                probs(i, 1) = 0.20 / n_dirs;
                probs(i, 2) = 0.40 / n_dirs;
                probs(i, 3) = 0.30 / n_dirs;
                probs(i, 4) = 0.10 / n_dirs;
            elseif n_spds == 6
                probs(i, 1) = 0.10 / n_dirs;
                probs(i, 2) = 0.20 / n_dirs;
                probs(i, 3) = 0.20 / n_dirs;
                probs(i, 4) = 0.20 / n_dirs;
                probs(i, 5) = 0.15 / n_dirs;
                probs(i, 6) = 0.10 / n_dirs;
            end
        end
    else
        % 非均匀分布：主风方向为西北(315°)
        main_dir = 315;
        
        for i = 1:n_dirs
            % 计算与主风方向的角度差
            angle_diff = min(abs(directions(i) - main_dir), ...
                            360 - abs(directions(i) - main_dir));
            % 高斯分布权重
            base_prob = exp(-angle_diff^2 / (2 * 60^2));
            
            if n_spds == 3
                probs(i, 1) = base_prob * 0.25;
                probs(i, 2) = base_prob * 0.60;
                probs(i, 3) = base_prob * 0.15;
            elseif n_spds == 4
                probs(i, 1) = base_prob * 0.20;
                probs(i, 2) = base_prob * 0.40;
                probs(i, 3) = base_prob * 0.30;
                probs(i, 4) = base_prob * 0.10;
            elseif n_spds == 6
                probs(i, 1) = base_prob * 0.10;
                probs(i, 2) = base_prob * 0.20;
                probs(i, 3) = base_prob * 0.20;
                probs(i, 4) = base_prob * 0.20;
                probs(i, 5) = base_prob * 0.15;
                probs(i, 6) = base_prob * 0.10;
            end
        end
    end
    
    % 归一化
    if sum(probs(:)) > 0
        probs = probs / sum(probs(:));
    end
end

function plot_wind_rose(probs, speeds, directions, title_str)
    % 绘制风玫瑰图
    n_dirs = length(directions);
    n_speeds = length(speeds);
    
    % 创建极坐标轴
    polaraxes;
    hold on;
    
    % 设置颜色映射
    colors = jet(n_speeds);
    
    % 每个风向的扇形宽度（弧度）
    width = 2 * pi / n_dirs;
    
    % 将风向转换为极坐标角度（0°为北，顺时针）
    theta_rad = deg2rad(90 - directions);
    
    % 绘制每个风向的扇形
    for i = 1:n_dirs
        theta_start = theta_rad(i) - width/2;
        
        % 从高风速到底风速叠加绘制
        for j = n_speeds:-1:1
            prob_value = probs(i, j);
            if prob_value > 0
                polarhistogram('BinEdges', [theta_start, theta_start + width], ...
                              'BinCounts', prob_value, ...
                              'FaceColor', colors(j, :), ...
                              'EdgeColor', 'white', ...
                              'LineWidth', 0.5);
            end
        end
    end
    
    % 设置图形属性
    thetalim([0 360]);
    rlim([0 0.8]);
    rticklabels({'20%', '40%', '60%', '80%'});
    
    % 添加方向标签
    theta_ticks = deg2rad(0:22.5:337.5);
    theta_ticklabels = {'N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', ...
                        'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'};
    thetaticks(rad2deg(theta_ticks));
    thetaticklabels(theta_ticklabels);
    
    title(title_str, 'FontSize', 10);
    
    % 添加图例
    legend_handles = gobjects(n_speeds, 1);
    for j = 1:n_speeds
        legend_handles(j) = patch(NaN, NaN, colors(j, :), ...
                                   'DisplayName', sprintf('%.0f m/s', speeds(j)));
    end
    legend(legend_handles, 'Location', 'eastoutside', 'FontSize', 8);
    
    hold off;
end

function plot_heatmap(probs, speeds, directions, title_str)
    % 绘制风速-风向联合分布热力图
    
    % 准备方向标签
    % dir_labels = {'N', '', 'NE', '', 'E', '', 'SE', '', ...
    %              'S', '', 'SW', '', 'W', '', 'NW', ''};
    dir_labels = {'N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', ...
                 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'};
    
    % 绘制热力图
    imagesc(probs' * 100);
    colormap(jet);
    colorbar;
    c = colorbar;
    c.Label.String = 'Probability (%)';
    
    % 设置坐标轴
    set(gca, 'YDir', 'normal');
    
    % X轴：风向
    xticks(1:length(directions));
    xticklabels(dir_labels);
    xtickangle(45);
    
    % Y轴：风速
    yticks(1:length(speeds));
    yticklabels(arrayfun(@(x) sprintf('%.0f m/s', x), speeds, 'UniformOutput', false));
    
    % 添加数值标签
    [n_dirs, n_speeds] = size(probs);
    for i = 1:n_dirs
        for j = 1:n_speeds
            if probs(i, j) > 0.005
                text(i, j, sprintf('%.1f', probs(i, j)*100), ...
                     'HorizontalAlignment', 'center', ...
                     'VerticalAlignment', 'middle', ...
                     'FontSize', 8, 'Color', 'k');
            end
        end
    end
    
    xlabel('Wind Direction', 'FontSize', 10);
    ylabel('Wind Speed', 'FontSize', 10);
    title(title_str, 'FontSize', 10, 'FontWeight', 'bold');
    
    xlim([0.5, n_dirs + 0.5]);
    ylim([0.5, n_speeds + 0.5]);
end

function result = iff(condition, true_val, false_val)
    % 简单的三元运算符替代
    if condition
        result = true_val;
    else
        result = false_val;
    end
end