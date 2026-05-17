function [Bestr1r2, Fbest,BestChart,BestFitness,farmlayout,farmlayout_NA]=CGPSO(popsize,wf,iterations,NA_type,tn,wt,t,Chaos_p,Strategy,rad_value,cls_type,c_r_type) 
    BestChart=zeros(iterations,1);
    BestFitness=zeros(iterations,1);
    Bestr1r2=zeros(iterations,2);
    farmlayout = zeros(iterations,wf.rows*wf.cols);
    farmlayout_NA = zeros(iterations,wf.rows*wf.cols);
    pop_power = zeros(popsize,wf.cols * wf.rows);
    
    D = wf.turbine_num; down = 1;up = wf.rows *wf.cols;
    
    % Population initialization
    [popu,lu] = windfarm_init(popsize, wf.turbine_num,wf);
    % Population evaluation
    [popuFitness,power_order,lp_power_accum]= wf_fitness(wf,popu);
    optimal = max(popuFitness);
    for i = 1:popsize
        pop_power(i,power_order(i,:))  = pop_power(i,power_order(i,:)) + (lp_power_accum(i,:) / sum(lp_power_accum(i,:)));
    end

   %% Local parameters setting
    vel = zeros(popsize, D);
    pBest = popu;
    pBestFit = popuFitness;
    pBest_power = power_order;
    [~, gBestId] = max(pBestFit);
    gBest = pBest(gBestId,:);
    gBest_fitness = pBestFit(gBestId);
    gBest_power = pBest_power(gBestId,:);
%     omega = 0.7298;
    omega = 0.9;
%     c1 = 1.49618;
    c1 = 1.49618;
    c2 = 1.49618;
    flag = zeros(popsize,1);
    sg = 7;
    pm = 0.01;
    count = 0;
    pfit = ones(1, 12);
    LEP = 25;
    success_num =zeros(LEP,12);
    fail_num = zeros(LEP,12);
    temp_lep = 1;
    r1 = 0.5;
    r2 = 0.5;
    for iter = 1:iterations
        st = 0;
        %% Search machine
        if Strategy ==3
            randPopuList = randperm(popsize);
            randPopuList = setdiff(randPopuList,1,'stable');
            indiR1 = pBest(randPopuList(1),:);
            indiR2 = pBest(randPopuList(2),:);
            radius = indiR1 - indiR2;
        else
            radius = rad_value * (1-iter/iterations);
        end
        
        [gBest,gBest_fitness,pfit,gBest_power,temp_lep] = Chaotic_selection(gBest,gBest_fitness,down,up,Chaos_p,wf,D,pfit,temp_lep,Strategy,iter,radius,success_num,fail_num,LEP,gBest_power,cls_type,c_r_type);

        for i = 1:popsize
          %% Exemplar Update: Crossover
            offsPbest = zeros(1,D);
            for d = 1:D
                k = randperm(popsize,1);
                if pBestFit(i) > pBestFit(k)

                    offsPbest(d) = r1 * pBest(i,d) + r2 * gBest(1,d);
                else
                    offsPbest(d) = pBest(k,d);
                end
            end
            
            
            
          %% Exemplar Update: Mutation
            for d = 1:D
                if rand < pm
                    offsPbest(d) = lu(1,d) + rand * (lu(2,d) - lu(1,d));
                end
            end
           
          %% Exemplar Update: Selection
            offsPbest = windfarm_constraint(offsPbest, wf.NA_loc, D,down,up);
            [offsPbestFitness,offs_power_order,~]= wf_fitness(wf,offsPbest);
            optimal = max(popuFitness);    
            if offsPbestFitness > optimal
                optimal = offsPbestFitness;
            end           
            if offsPbestFitness > pBestFit(i)
                pBest(i,:) = offsPbest;
                pBestFit(i) = offsPbestFitness;
                pBest_power(i,:) = offs_power_order;
            end
          %% 20%M tournament
            if flag(i) == sg
                flag(i) = 0;
                competitor = randperm(popsize, 0.2 * popsize);
                [~, winId] = max(pBestFit(competitor));
                pBest(i,:) = pBest(competitor(winId),:);
                pBestFit(i) = pBestFit(competitor(winId));
                pBest_power(i,:) = pBest_power(competitor(winId),:);
            end    
          %% Particle Update
            for d = 1:D
                vel(i,d) =  omega * vel(i,d) + c1 * rand * pBest(i,d) - c2 * rand *popu(i,d);
                popu(i,d) = popu(i,d) + vel(i,d);
            end
        end
       %% Boundary detection
        popu = windfarm_constraint(popu, wf.NA_loc, D,down,up);
        [popuFitness,power_order,lp_power_accum] = wf_fitness(wf,popu);
        
        for i = 1:popsize
            pop_power(i,power_order(i,:))  = pop_power(i,power_order(i,:)) + (lp_power_accum(i,:) / sum(lp_power_accum(i,:)));
        end
        
        %% Evaluatition
        if max(popuFitness) > optimal
            optimal = max(popuFitness);
        else
            count = count +1;
        end
        
        %% Update pBest
        pos = popuFitness > pBestFit;
        flag(~pos) = flag(~pos) + 1;
        pBestFit(pos) = popuFitness(pos);
        pBest(pos,:) = popu(pos,:);
        [gBestFit, gBestId] = max(pBestFit);
        if gBestFit > gBest_fitness
            gBest = pBest(gBestId,:);
            gBest_fitness =gBestFit;
            gBest_power = pBest_power(gBestId,:);
        end
        %% Survival
        [gBestFitMax, gBestId] = max(pBestFit);
        pytuple = py.ppo.main(py.numpy.array(tn), py.numpy.array(iter), py.numpy.array(pBest), py.numpy.array(gBest), pBestFit, gBestFitMax, r1, r2, popsize, D);%         gBest = double(gbest1);
        pytuple = cell(pytuple);
        gBest = double(pytuple{1});
        state_current = double(pytuple{2});
        r1 = state_current(1);
        r2 = state_current(2);
        pBest(gBestId,:) = gBest;
        [pBest1,power_order,lp_power_accum] = wf_fitness(wf,pBest);
        [gBestFit, gBestId] = max(pBest1);
        if gBestFit > gBest_fitness
            gBest = pBest(gBestId,:);
            gBest_fitness =gBestFit;
            gBest_power = pBest_power(gBestId,:);
        end        
        
        Fbest = gBest_fitness;
        BestChart(iter) = gBest_fitness / wf.power_total;
        BestFitness(iter) = gBest_fitness;
        Bestr1r2(iter, 1) = r1;
        Bestr1r2(iter, 2) = r2;
        [best_farmlayout,best_farmlayout_NA]  = gene_layout_by_indices_one(wf,gBest);
        farmlayout(iter,:) = best_farmlayout;
        farmlayout_NA(iter,:) = best_farmlayout_NA;
        fprintf('NA %d Turbine Num:%d Wind %s run: %d iteration: %d  eta %f fitness %f rl_success %d\n',NA_type,tn,wt,t,iter,(Fbest / wf.power_total),Fbest,st)
    end
end