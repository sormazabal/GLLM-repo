%% loading StepMiner/ANOVA results
load(strcat('../',projectId,'/',bioMarker,'/',bioMarker,'_name_pool.mat'))
name_pool = cellstr(name_pool');  % genes pass StepMiner/ANOVA test
load(strcat('../',projectId,'/',bioMarker,'/',bioMarker,'_bind_info.mat'))
load(strcat('../',projectId,'/',bioMarker,'/',bioMarker,'_nonstemness.mat'))

bind_info = bind;
anova_exprs = double(ANOVA_profile_nonstemness);

% normalization
gene_mean = mean(anova_exprs,2);
gene_var = var(anova_exprs,0,2);
anova_exprs = (anova_exprs - gene_mean) ./ sqrt(gene_var);

%% build interaction network and trim the network with AIC/t-test
extra_reg_num = 0;  % not equal to 0 with basal level present
reg_ability = nan(size(bind_info,1),size(bind_info,2)+extra_reg_num);
reg_ability_AIC = nan(size(bind_info,1),size(bind_info,2)+extra_reg_num);
reg_ability_pvalue = nan(size(bind_info,1),size(bind_info,2)+extra_reg_num);
Pvalue = nan(size(bind_info,1),size(bind_info,2)+extra_reg_num);
cut_TFs_AIC = nan(size(bind_info,1),size(bind_info,2)+extra_reg_num);
reg_TFs_AIC = nan(size(bind_info,1),size(bind_info,2)+extra_reg_num);
cut_TFs_pvalue = nan(size(bind_info,1),size(bind_info,2)+extra_reg_num);
reg_TFs_pvalue = nan(size(bind_info,1),size(bind_info,2)+extra_reg_num);

threshold = 0.05;
reg_num_pvalue = [];

for i = 1:size(bind_info,1)  % for all genes
    if sum(bind_info(i,:)) > 0  % if there are interactions
        % LMMSE problem
        bind_gene = find(bind_info(i,:));  % index of connected genes
        X = anova_exprs(i,:)';
        phi = [];
        for j = 1:length(bind_gene)
            %phi = cat(1,phi,anova_exprs(bind_gene(j),1:tt));
            phi = [phi;anova_exprs(bind_gene(j),:)];
        end
        phi = phi';
        cons = [zeros(1,length(bind_gene))];
        options = optimset('LargeScale','off');
        theta=lsqlin(phi,X,cons,0,[],[],[],[],[],options);
        for j = 1:length(bind_gene)
            reg_ability(i,bind_gene(j)) = theta(j);
        end

        % AIC 
        if (size(bind_gene,2) > 1)
            % trim with AIC if more than one connections
            [theta_AIC,phi_AIC,reg_id_AIC,cut_id_AIC,AIC_value] = ...
                AICfun_ForStepRegression_constrained(X,phi,theta,bind_gene);
        else
            % no connections trimmed
            theta_AIC = theta;
            phi_AIC = phi;
            reg_id_AIC = bind_gene;
            cut_id_AIC = [];
            sigma_aic = (X - phi*theta)'*(X - phi*theta) / length(X);
            AIC_value = log(sigma_aic) + 2*length(theta) / length(X);
        end
        reg_TFs_AIC(i,1:length(reg_id_AIC)) = reg_id_AIC;
        cut_TFs_AIC(i,1:length(cut_id_AIC)) = cut_id_AIC;
        reg_AIC = nan(1,size(bind_info,2)+extra_reg_num);
        for j = 1:length(reg_id_AIC)
            reg_AIC(reg_id_AIC(j)) = theta_AIC(j);
        end        
        reg_ability_AIC(i,:) = reg_AIC;
        
        % Student's t-test
        if (size(theta_AIC,1) > 1)
            [theta_pvalue,phi_pvalue,reg_id_pvalue,cut_id_pvalue,PPvalue] = ...
                pvalue_selection_constrained_gene(X,phi_AIC,theta_AIC,reg_id_AIC,extra_reg_num,threshold);
        else
            theta_pvalue = theta_AIC;
            phi_pvalue = phi_AIC;
            reg_id_pvalue = reg_id_AIC;
            cut_id_pvalue = cut_id_AIC;
            PPvalue = ones(size(theta_AIC));
        end
        reg_TFs_pvalue(i,1:length(reg_id_pvalue)) = reg_id_pvalue;
        cut_TFs_pvalue(i,1:length(cut_id_pvalue)) = cut_id_pvalue;
        reg_pvalue = nan(1,size(bind_info,2)+extra_reg_num);
        pvalue_pvalue = nan(1,size(bind_info,2)+extra_reg_num);
        % pvalue_pvalue is the p-value for each connecion
        for p = 1:length(reg_id_pvalue)
            reg_pvalue(reg_id_pvalue(p)) = theta_pvalue(p);
            pvalue_pvalue(reg_id_pvalue(p)) = PPvalue(p);
        end
        reg_ability_pvalue(i,:) = reg_pvalue;
        Pvalue(i,:) = pvalue_pvalue;        
        reg_num_pvalue(i) = length(reg_id_pvalue);  % #connections left
        clear X phi theta cons reg_AIC reg_pvalue phi_AIC phi_pvalue theta_AIC theta_pvalue pvalue_pvalue PPvalue
        clc;   
    else
        reg_num_pvalue(i) = 0;  % no connections
        clc;   
    end
end

clear i j k p reg_id_AIC cut_id_AIC reg_id_pvalue cut_id_pvalue AIC_value

Final_reg_ability = reg_ability_pvalue;
Final_Pvalue = Pvalue;

%% retrieve trimmed reg_ability
disp('saving network')

p = 1-isnan(Final_reg_ability(:,1:size(Final_reg_ability,2)-extra_reg_num));

% p is 0 iff Final_reg_ability is nan (no connection)
for i = 1:size(p,1)
    for j = 1:size(p,2)
        if p(i,j) == 0  % is nan == no connection
            Final_reg_ability_0(i,j) = 0;
        else  % is NOT nan == with connection
            Final_reg_ability_0(i,j) = Final_reg_ability(i,j);
        end
    end
end
clear p i j;

Final_reg_ability_0 = Final_reg_ability_0';
% final reg_ability A<->B = (A->B + B->A) / 2.0
Final_reg_ability_1 = (Final_reg_ability_0 + Final_reg_ability_0')./2;
Final_reg_ability_1 = triu(Final_reg_ability_1,1);

% retrieve corresponding p-values
Final_Pvalue_0 = Final_Pvalue;           
[x,y,z] = find(Final_reg_ability_1);
for m = 1:length(x)
    pv(m) = Final_Pvalue_0(x(m),y(m));
end
pv = pv';

Final_TF_gene_pair = [x,y,z,pv];
Finalgenename = name_pool(Final_TF_gene_pair(:,2));
Finaltfname = name_pool(Final_TF_gene_pair(:,1));

cd(strcat('../',projectId,'/',bioMarker))
save 'Final_TF_gene_pair_nonstem.mat' 'Final_TF_gene_pair' '-v7.3'
save 'Finalgenename_nonstem.mat' 'Finalgenename' '-v7.3'
save 'Finaltfname_nonstem.mat' 'Finaltfname' '-v7.3'

Final_reg_ability_nonstem = Final_reg_ability_0;
save 'Final_reg_ability_nonstem.mat' 'Final_reg_ability_nonstem' '-v7.3'

clear Final_reg_ability_nonstem
clear x y z pv 
clear Final_reg_ability_0 Final_reg_ability_1 Final_Pvalue_0