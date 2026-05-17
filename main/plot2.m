%% 四种算法风电场布局对比 - HHWOA, CGPSO, BDE, RDE
% 对应论文图6的风格，展示28°、128°、144°、183°四个风向
% 顺序: HHWOA | CGPSO | BDE | RDE

clear; clc; close all;

%% 1. 定义算法路径和名称
algorithms = {'HHWOA', 'CGPSO', 'BDE', 'RDE'};
algorithm_names = {'HHWOA', 'CGPSO', 'BDE', 'RDE'};

% 基础路径
base_path = 'D:\MatlabProjects\BDE-WindFarm_code\code\Results\';

% 存储每个算法的布局数据
best_layouts = cell(1, 4);
turbine_positions_all = cell(1, 4);
turbine_grid_pos_all = cell(1, 4);
n_turbines_all = zeros(1, 4);

%% 2. 加载所有算法的数据
for alg_idx = 1:4
    alg_name = algorithms{alg_idx};
    data_path = fullfile(base_path, sprintf('20260424%s', alg_name), ...
                         '6speed_12direction', 'tn30_NA0');
    
    layout_file = fullfile(data_path, 'farmlayout1.mat');
    
    if exist(layout_file, 'file')
        % 加载布局数据
        load(layout_file);
        
        % farmlayout 维度: [runTime, iterations, cells] 或 [iterations, cells]
        if ndims(farmlayout) == 3
            % 三维: [runTime, iterations, cells]
            [n_runs, n_iter, n_cells] = size(farmlayout);
            best_layout = squeeze(farmlayout(end, end, :));
        elseif size(farmlayout, 1) > 100
            % 二维: [iterations, cells]
            best_layout = farmlayout(end, :);
        else
            best_layout = farmlayout;
        end
        
        best_layouts{alg_idx} = best_layout;
        fprintf('已加载 %s 数据\n', alg_name);
    else
        fprintf('警告：未找到 %s 数据\n', alg_name);
        best_layouts{alg_idx} = [];
    end
end

%% 3. 风电场参数设置
rows = 21;                          % 网格行数
cols = 21;                          % 网格列数
cell_width = 77 * 3;                % 单个网格宽度 231米

% 风机参数
rotor_radius = 77;                  % 转子半径 (米)
hub_height = 80;                    % 轮毂高度 (米)
surface_roughness = 0.25;           % 地表粗糙度
entrainment_constant = 0.5 / log(hub_height / surface_roughness);

% 环境风速
ambient_wind_speed = 12;            % m/s

% 四个风向角度
wind_directions = [28, 128, 144, 183];
direction_names = {'\theta=28^\circ', '\theta=128^\circ', '\theta=144^\circ', '\theta=183^\circ'};

%% 4. 提取每个算法的最佳布局和风机位置
for alg_idx = 1:4
    best_layout = best_layouts{alg_idx};
    
    if ~isempty(best_layout)
        % 找出风机位置（大于0的位置）
        turbine_indices = find(best_layout > 0);
        n_turbines_all(alg_idx) = length(turbine_indices);
        
        % 转换为网格坐标
        turbine_grid_pos = zeros(length(turbine_indices), 2);
        turbine_positions = zeros(length(turbine_indices), 2);
        
        for i = 1:length(turbine_indices)
            idx = turbine_indices(i);
            row = ceil(idx / cols);
            col = mod(idx - 1, cols) + 1;
            turbine_grid_pos(i, :) = [col, row];
            turbine_positions(i, :) = [col * cell_width - cell_width/2, ...
                                       row * cell_width - cell_width/2];
        end
        
        turbine_grid_pos_all{alg_idx} = turbine_grid_pos;
        turbine_positions_all{alg_idx} = turbine_positions;
        
        fprintf('%s: 风机数量 = %d\n', algorithms{alg_idx}, length(turbine_indices));
    else
        turbine_positions_all{alg_idx} = [];
        turbine_grid_pos_all{alg_idx} = [];
        fprintf('%s: 无数据\n', algorithms{alg_idx});
    end
end

%% 5. 绘制论文图6风格对比图（4行 x 4列布局）
figure('Position', [50, 50, 1800, 1400]);

% 定义子图位置参数（缩短间隔）
left_margin = 0.06;      % 左边界
bottom_margin = 0.06;    % 下边界
width = 0.205;           % 每个子图的宽度
height = 0.205;          % 每个子图的高度
h_gap = 0.00;            % 水平间隔
v_gap = 0.02;            % 垂直间隔

for alg_idx = 1:4
    for dir_idx = 1:4
        % 计算子图位置
        pos_x = left_margin + (dir_idx - 1) * (width + h_gap);
        pos_y = bottom_margin + (4 - alg_idx) * (height + v_gap);
        
        % 创建子图并设置位置
        subplot('Position', [pos_x, pos_y, width, height]);
        
        wind_angle_deg = wind_directions(dir_idx);
        
        % 获取当前算法的风机位置
        turbine_positions = turbine_positions_all{alg_idx};
        turbine_grid_pos = turbine_grid_pos_all{alg_idx};
        
        if isempty(turbine_positions)
            text(0.5, 0.5, 'No Data', 'HorizontalAlignment', 'center', ...
                'FontSize', 14, 'FontWeight', 'bold');
            axis off;
            continue;
        end
        
        % 计算风速分布
        wind_speed_map = calculate_wind_speed_map(rows, cols, cell_width, ...
            turbine_positions, wind_angle_deg, rotor_radius, entrainment_constant, ...
            ambient_wind_speed);
        
        % 绘制风速热图
        imagesc(1:cols, 1:rows, wind_speed_map);
        set(gca, 'YDir', 'normal');
        
        % 使用jet colormap（红到黄）
        colormap(gca, 'jet');
        
        hold on;
        
        % 绘制风机位置（黑色圆点）
        plot(turbine_grid_pos(:, 1), turbine_grid_pos(:, 2), ...
            'ko', 'MarkerSize', 5, 'MarkerFaceColor', 'k', 'LineWidth', 0.8);
        
        % 添加风向箭头
        arrow_start_x = cols * 0.85;
        arrow_start_y = rows * 0.88;
        arrow_len = rows * 0.1;
        wind_rad = deg2rad(wind_angle_deg);
        arrow_dx = arrow_len * cos(wind_rad);
        arrow_dy = -arrow_len * sin(wind_rad);
        
        quiver(arrow_start_x, arrow_start_y, arrow_dx, arrow_dy, 0, ...
            'Color', 'w', 'LineWidth', 2, 'MaxHeadSize', 0.35);
        
        % 风向文字
        text(arrow_start_x + arrow_dx/2, arrow_start_y + arrow_dy/2, ...
            sprintf('%.0f°', wind_angle_deg), 'Color', 'w', ...
            'FontSize', 9, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
        
        % 图形设置
        axis equal tight;
        xlim([0.5, cols+0.5]);
        ylim([0.5, rows+0.5]);
        
        % 设置坐标轴刻度（显示数字标签）
        set(gca, 'XTick', 1:5:cols);
        set(gca, 'YTick', 1:5:rows);
        set(gca, 'XTickLabel', 1:5:cols);
        set(gca, 'YTickLabel', 1:5:rows);
        set(gca, 'FontSize', 8);
        set(gca, 'TickDir', 'out');
        
        % 设置坐标轴标签颜色为黑色（可见）
        set(gca, 'XColor', 'k');
        set(gca, 'YColor', 'k');
        
        % 设置列标题（第一行）
        if alg_idx == 1
            title(direction_names{dir_idx}, 'FontSize', 13, 'FontWeight', 'bold', 'Interpreter', 'tex', 'Color', 'k');
        end
        
        % 设置算法名称（第一列左侧）
        if dir_idx == 1
            text(-6, rows/2, algorithm_names{alg_idx}, ...
                'FontSize', 12, 'FontWeight', 'bold', ...
                'HorizontalAlignment', 'center', 'Rotation', 90, 'Color', 'k');
        end
        
        % 设置背景色
        set(gca, 'Color', [0.12, 0.12, 0.12]);
        set(gcf, 'Color', 'w');
        
        % 颜色条放在右边
        hcb = colorbar('eastoutside');
        hcb.FontSize = 8;
        hcb.Color = 'k';
        
        % 设置颜色条范围
        caxis([min(wind_speed_map(:)), max(wind_speed_map(:))]);
    end
end

%% 6. 保存主对比图为SVG矢量格式
output_dir = fullfile(base_path, 'Comparison');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% 保存为SVG矢量格式
svg_file = fullfile(output_dir, 'Fig6_Algorithm_Comparison_WindFarm_Layout.svg');
print(gcf, svg_file, '-dsvg', '-r300');
fprintf('主对比图已保存为SVG矢量格式: %s\n', svg_file);

% 同时保存PDF和FIG作为备份
pdf_file = fullfile(output_dir, 'Fig6_Algorithm_Comparison_WindFarm_Layout.pdf');
print(gcf, pdf_file, '-dpdf', '-r300');
fprintf('PDF备份版本已保存: %s\n', pdf_file);

fig_file = fullfile(output_dir, 'Fig6_Algorithm_Comparison_WindFarm_Layout.fig');
saveas(gcf, fig_file);
fprintf('FIG备份版本已保存: %s\n', fig_file);

fprintf('所有文件已保存至: %s\n', output_dir);

%% ==================== 辅助函数 ====================

function wind_speed_map = calculate_wind_speed_map(rows, cols, cell_width, ...
    turbine_positions, wind_angle_deg, rotor_radius, entrainment_constant, ...
    ambient_wind_speed)
    % 计算风速分布矩阵（Jensen尾流模型）
    
    wind_speed_map = ones(rows, cols) * ambient_wind_speed;
    wind_angle_rad = deg2rad(wind_angle_deg);
    wind_dir = [cos(wind_angle_rad), sin(wind_angle_rad)];
    
    if isempty(turbine_positions)
        return;
    end
    
    for i = 1:length(turbine_positions)
        turbine_pos = turbine_positions(i, :);
        
        for row = 1:rows
            for col = 1:cols
                point_x = col * cell_width - cell_width/2;
                point_y = row * cell_width - cell_width/2;
                displacement = [point_x - turbine_pos(1), point_y - turbine_pos(2)];
                
                along_wind = dot(displacement, wind_dir);
                cross_wind = abs(displacement(1)*wind_dir(2) - displacement(2)*wind_dir(1));
                
                if along_wind > 0 && along_wind < 4000
                    wake_radius = rotor_radius + entrainment_constant * along_wind;
                    if cross_wind < wake_radius
                        wake_decay = (rotor_radius / wake_radius)^2;
                        wake_decay = wake_decay * (1 - 1/3);
                        affected_speed = ambient_wind_speed * (1 - wake_decay);
                        wind_speed_map(row, col) = min(wind_speed_map(row, col), affected_speed);
                    end
                end
            end
        end
    end
    
    wind_speed_map = max(wind_speed_map, 0);
end

function [wake_x, wake_y] = draw_wake_polygon(turbine_grid_pos, wind_angle_deg, wake_len_grid)
    % 绘制尾流区域多边形
    
    wind_rad = deg2rad(wind_angle_deg);
    wind_dir = [cos(wind_rad), -sin(wind_rad)];
    
    expansion_angle = deg2rad(12);
    
    start_point = turbine_grid_pos;
    end_point = turbine_grid_pos + wind_dir * wake_len_grid;
    
    if end_point(1) < 1 || end_point(1) > 21 || end_point(2) < 1 || end_point(2) > 21
        wake_x = [];
        wake_y = [];
        return;
    end
    
    half_width_start = 0.5;
    half_width_end = wake_len_grid * tan(expansion_angle);
    
    perp_dir = [-wind_dir(2), wind_dir(1)];
    
    p1 = start_point + perp_dir * half_width_start;
    p2 = start_point - perp_dir * half_width_start;
    p3 = end_point - perp_dir * half_width_end;
    p4 = end_point + perp_dir * half_width_end;
    
    wake_x = [p1(1), p2(1), p3(1), p4(1)];
    wake_y = [p1(2), p2(2), p3(2), p4(2)];
end