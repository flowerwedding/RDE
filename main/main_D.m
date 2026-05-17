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
runTime = 51;
popsize = 50;
max_it  = 400;

%% wind farm parameters
rows = 28;
cols = 28;
cell_width= 77.0 * 3;
turbine_num = [30,35,40, ];

NA_type_list = 13;

%% wind scenarios
n_speeds = [3,3,4,6];
n_directions = [12,12,12,12];
unifrom = [0,1,0,0];
% 
% turbine_num = [30];
% n_speeds = [3];
% n_directions = [12];
% unifrom = [0];

%% Para
% parpool('threads')
delete(gcp('nocreate'))
mycluster = parcluster('local');
mycluster.NumWorkers = 20;

cls_type = 3; % 1:individual 2d;  2: population 2D; 3:population 1D
c_r_type = 3 ; % 1:x 2:y 3:x&y
rad_value  = (0.01);
for rad_ind = 1:length(rad_value)
    for Strategy = 3
        total_st=tic;
        rad_value_temp = rad_value(rad_ind);
        algorithmDir = sprintf('BDE_NA13_C%d_rad_%.2f_cls_%d_cr_%d',Strategy,rad_value_temp,cls_type,c_r_type);

        text_path = sprintf('%s/%s/',save_path,algorithmDir);
        text_file = sprintf('%s/cost_time.txt',text_path);
        if ~exist(text_path,'dir')
            mkdir(text_path)
        end
        f = fopen(text_file,'a');
        fprintf(f, datestr(now));
        fprintf(f,'\n');

        for wt =+1:length(n_speeds)
            for tn = turbine_num
                for NA_type = NA_type_list
                    NA_loc_array = gene_NA_loc(NA_type);
                    [wf,ws_folder] = gene_windfram(rows,cols,tn,cell_width,NA_loc_array,n_speeds(wt),n_directions(wt),unifrom(wt));
                    save('wf.mat', 'wf');
                    folder = sprintf('%s/%s/%s/tn%d_NA%d',save_path,algorithmDir,ws_folder,tn, NA_type);

                    if ~exist(folder,'dir')
                        mkdir(folder)
                    end

                    eta=zeros(max_it,runTime);
                    fitness = zeros(max_it,runTime);

                    farmlayout= zeros(runTime,max_it,wf.cols*wf.rows);
                    farmlayout_NA= zeros(runTime,max_it,wf.cols*wf.rows);
                    fprintf('%s - %s TN %d\n',algorithmDir,ws_folder,tn)
                    for t=1:runTime
                        if t == 30
                            fprintf('\n')
                        end
                        st=tic;
                        [Fbest,BestChart,BestFitness,farmlayout(t,:,:),farmlayout_NA(t,:,:)]=BDE(popsize,wf,max_it,NA_type,tn,ws_folder,t);
                        end_t = toc(st);
                        cost_t = seconds(end_t);
                        cost_t.Format = 'hh:mm:ss';
                        eta(:,t) = BestChart;
                        fitness(:,t) = BestFitness;
                        save_results(reshape(farmlayout(t,:,:),size(farmlayout(t,:,:),2),size(farmlayout_NA(t,:,:),3)),reshape(farmlayout_NA(t,:,:),size(farmlayout_NA(t,:,:),2),size(farmlayout(t,:,:),3)),t,folder)
                        save(sprintf('%s/eta.mat',folder),"eta")
                        save(sprintf('%s/fitness.mat',folder),"fitness")
                        fprintf('\n%s - %s TN %d Cost Times %s\n',algorithmDir,ws_folder,tn,cost_t)
                        fprintf(f,'%s - %s TN %d Cost Times %s\n',algorithmDir,ws_folder,tn,cost_t);                        
                    end
                end

            end
        end
        total_cost = toc(total_st);
        total_cost_t = seconds(total_cost   );
        total_cost_t.Format = 'hh:mm:ss';
        fprintf(f,'Total cost: %s\n',total_cost_t);
        fclose(f);
    end
end