function [fitness_val,power_order] = wf_fitness(wf,population)
    population = gene_layout_by_indices(wf,population);
    pop_size = size(population,1);
    power_order = zeros(pop_size, wf.turbine_num);
    fitness_val = zeros(pop_size,1);
    for i =1:pop_size
        xy_position = zeros(2, wf.turbine_num);
        cr_position = zeros(2, wf.turbine_num);
        ind_position = zeros(wf.turbine_num,1);
        ind_pos = 1;
        for ind = 1: wf.rows * wf.cols
            if population(i, ind) == 1
        
                r_i = floor((ind-1) / wf.cols);

                c_i = floor(ind - 1 - r_i * wf.cols);

                cr_position(1, ind_pos) = c_i;
                cr_position(2, ind_pos) = r_i;
                xy_position(1, ind_pos) = c_i * wf.cell_width + wf.cell_width_half;
                xy_position(2, ind_pos) = r_i * wf.cell_width + wf.cell_width_half;
                ind_position(ind_pos) = ind;
                ind_pos  = ind_pos+ 1;
            end
        end
    
        lp_power_accum = zeros(wf.turbine_num,1)  ;
    
        for ind_t  = 1:length(wf.theta)
            for ind_v  = 1 : length(wf.velocity)
                trans_matrix = [[cos(wf.theta(ind_t)),-sin(wf.theta(ind_t))];
                    [sin(wf.theta(ind_t)),cos(wf.theta(ind_t))]];
    
                trans_xy_position = trans_matrix * xy_position;
                speed_deficiency = wake_calculate(trans_xy_position, wf.turbine_num,wf.turbine);
    
                actual_velocity = (1 - speed_deficiency) * wf.velocity(ind_v);
                
                lp_power = layout_power(actual_velocity, wf.turbine_num);
                lp_power = lp_power * wf.f_theta_v(ind_t, ind_v);
                lp_power_accum = lp_power_accum +lp_power;
    
            end
        end
        [~,sorted_index]= sort(lp_power_accum);
        power_order(i, :) = ind_position(sorted_index);
        fitness_val(i) = sum(lp_power_accum);
        
    end
end

function [wake_deficiency] = wake_calculate(trans_xy_position,turbine_num,turbine)
    [~,sorted_index]= sort(-trans_xy_position(2,:));
    
    wake_deficiency = zeros(turbine_num,1);
    wake_deficiency(sorted_index(1)) = 0;
    for i  = 2:turbine_num
        for j  = 1: i
            xdis = abs(trans_xy_position(1, sorted_index(i)) - trans_xy_position(1, sorted_index(j)));
            ydis = abs(trans_xy_position(2, sorted_index(i)) - trans_xy_position(2, sorted_index(j)));
            d = cal_deficiency(xdis, ydis, turbine.rator_radius, turbine.entrainment_const);
            wake_deficiency(sorted_index(i)) = wake_deficiency(sorted_index(i))  +d^ 2;
    
            
        end
        wake_deficiency(sorted_index(i)) = sqrt(wake_deficiency(sorted_index(i)));
    end

end

function d = cal_deficiency( dx, dy, r, ec)
    if dy == 0
        d = 0;
    else
        R = r + ec * dy;
        inter_area = cal_interaction_area(dx, dy, r, R);
        d = 2.0 / 3.0 * (r ^ 2) / (R ^ 2) * inter_area / (pi * r ^ 2);
    
    end
end

function area =cal_interaction_area(dx, dy, r, R)

    if dx >= r + R
        area= 0;
    elseif  dx >= sqrt(R ^ 2 - r ^ 2)
    
        alpha = acos((R ^ 2 + dx ^ 2 - r ^ 2) / (2 * R * dx));
        beta = acos((r ^ 2 + dx ^ 2 - R ^ 2) / (2 * r * dx));
        A1 = alpha * R ^ 2;
        A2 = beta * r ^ 2;
        A3 = R * dx * sin(alpha);
        area= A1 + A2 - A3;
    elseif dx >= R - r
        alpha = acos((R ^ 2 + dx ^ 2 - r ^ 2) / (2 * R * dx));
        beta = pi - acos((r ^ 2 + dx ^ 2 - R ^ 2) / (2 * r * dx));
        A1 = alpha * R ^ 2;
        A2 = beta * r ^ 2;
        A3 = R * dx * sin(alpha);
        area =  pi * r ^ 2 - (A2 + A3 - A1);
    else
        area = pi * r ^ 2;
    end

end

function power = layout_power( velocity, turbine_num)

    power = zeros(turbine_num,1);
    for i =1:turbine_num
        power(i) = P_i_X(velocity(i));
    end
        
end 

%% pix
function re= P_i_X(v)
    if v < 2.0
        re= 0;
    elseif v < 12.8
        re =  0.3 * v ^ 3;
    elseif v < 18
        re = 629.1;
    else
        re = 0;
    end
end

