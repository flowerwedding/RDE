function [wf,ws_folder] = gene_windfram(rows,cols,turbine_num,cell_width,NA_loc,n_speeds,n_directions,unifrom)
format long
    wf.rows = rows;
    wf.cols = cols;
    wf.turbine_num = turbine_num;
    wf.cell_width = cell_width;
    wf.cell_width_half = cell_width *0.5;
    wf.NA_loc = NA_loc;
    
   
    turbine.hub_height = 80.0;
    turbine.rator_diameter = 77.0;
    turbine.surface_roughness = 0.25 * 0.001;
    turbine.rator_radius = turbine.rator_diameter / 2;
    turbine.entrainment_const = 0.5 / log(turbine.hub_height / turbine.surface_roughness);
    

    wf.turbine= turbine;
    
    if unifrom
        ws_folder = sprintf('%dspeed_%ddirection_uniform',n_speeds,n_directions);
    else
        ws_folder = sprintf('%dspeed_%ddirection',n_speeds,n_directions);
    end
    
    ws = load([ws_folder,'.mat']);

    wf.theta = ws.theta;
    wf.velocity = ws.velocity;
    wf.f_theta_v =ws.f_theta_v;
    

    wf = cal_P_rate_total(wf);
end

function wf=init_1_direction_1_N_speed_13(wf)
    wf.theta =  [0];
    wf.velocity = [13.0];
    wf.f_theta_v =[[1]];
end

function wf= init_4_direction_1_speed_13(wf)
    wf.theta =  [0, 3 * pi / 6.0, 6 * pi / 6.0, 9 * pi / 6.0];
    wf.velocity = [13.0];
    wf.f_theta_v =[[0.25]; [0.25]; [0.25]; [0.25]];
end


function wf=init_6_direction_1_speed_13(wf)
    wf.theta = [0, pi / 3.0, 2 * pi / 3.0, 3 * pi / 3.0, 4 * pi / 3.0, 5 * pi / 3.0];
    wf.velocity = [13.0];
    wf.f_theta_v = [[0.2];[0.3]; [0.2]; [0.1]; [0.1]; [0.1]];
end

function wf=init_12_direction_3_speed(wf)
    wf.theta = [0, pi / 6.0, 2 * pi / 6.0, 3 * pi / 6.0, 4 * pi / 6.0, 5 * pi / 6.0,...
        6*pi /6.0, 7*pi /6.0, 8*pi/6.0, 9*pi/6.0, 10*pi/6.0,11*pi/6.0];
    wf.velocity = [13.0,10.0,7.0];
    wf.f_theta_v =  [[0.058333, 0.016667, 0.008333]; 
        [0.058333, 0.016667, 0.008333];
        [0.058333, 0.016667, 0.008333];
        [0.058333, 0.016667, 0.008333];
        [0.058333, 0.016667, 0.008333];
        [0.058333, 0.016667, 0.008333];
        [0.058333, 0.016667, 0.008333];
        [0.058333, 0.016667, 0.008333];
        [0.058333, 0.016667, 0.008333];
        [0.058333, 0.016667, 0.008333];
        [0.058333, 0.016667, 0.008333];
        [0.058333, 0.016667, 0.008333]];
end

function wf = cal_P_rate_total(wf)
    f_p = 0.0;
    for ind_t =1:length(wf.theta)
        for ind_v = 1 :length(wf.velocity)
            f_p = f_p+ wf.f_theta_v(ind_t, ind_v) * P_i_X(wf.velocity(ind_v));
        end
    end
    wf.power_total = wf.turbine_num * f_p;
end



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


