function [theta_pvalue,phi_pvalue,reg_id_pvalue,cut_id_pvalue,PPvalue] = ...
    pvalue_selection_constrained_gene(X,phi,theta,ori_reg_id,extra_reg_num,threshold)
% p-value selection by p-value < threshold

% theta=pinv(phi'*phi)*phi'*X;
% cons=[zeros(1,size(phi,2)-extra_reg_num),0,-1];
options=optimset('LargeScale','off');
% theta=lsqlin(phi,X,cons,0,[],[],[],[],[],options);
reg_id_pvalue=ori_reg_id;
cut_id_pvalue=[];
phi_pvalue=phi;

if isempty(ori_reg_id) == 0
    for x=1:length(ori_reg_id)
        for m=1:length(reg_id_pvalue)
            PPvalue(m)=pvaluefun_t(phi_pvalue,X,theta,m); % p-values for regulation ability
        end
        
         %PPvalue(length(reg_id_pvalue)+1)=pvaluefun_t(phi_pvalue,X,theta,length(reg_id_pvalue)+1); % p-value for lambda
         %PPvalue(length(reg_id_pvalue)+2)=pvaluefun_t(phi_pvalue,X,theta,length(reg_id_pvalue)+2); % p-value for k  
         
%         Pvalue_adj=Pvalue.*(length(reg_id_pvalue)+extra_reg_num);
        cut=find(PPvalue(1:length(reg_id_pvalue))>(threshold/length(reg_id_pvalue)));
        if isempty(cut)==0   % some regulator cut by p-value
            cut_id_pvalue=cat(2,cut_id_pvalue,reg_id_pvalue(cut));
            phi_pvalue(:,cut)=[];
            reg_id_pvalue(cut)=[];
            clear cut
        else
            theta_pvalue=theta;
            break
        end
%     theta_pvalue=pinv(phi_pvalue'*phi_pvalue)*phi_pvalue'*X;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if (isempty(phi_pvalue)==0)
        %cons=[zeros(1,size(phi_pvalue,2)-extra_reg_num),0,-1];
        cons=[zeros(1,size(phi_pvalue,2)-extra_reg_num)];
        theta_pvalue=lsqlin(phi_pvalue,X,cons,0,[],[],[],[],[],options);
    else
        theta_pvalue=[];
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    if isempty(reg_id_pvalue)==0
        theta=theta_pvalue;
        clear cons PPvalue
    else
        theta_pvalue=theta_pvalue;
        phi_pvalue=phi_pvalue;
        reg_id_pvalue=reg_id_pvalue;
        cut_id_pvalue=sort(cut_id_pvalue);
        %PPvalue(1)=pvaluefun_t(phi_pvalue,X,theta_pvalue,1); % p-value for lambda
        %PPvalue(2)=pvaluefun_t(phi_pvalue,X,theta_pvalue,2); % p-value for k
        break
    end
    end
    %theta_pvalue=theta_pvalue;
    %phi_pvalue=phi_pvalue;
    %reg_id_pvalue=reg_id_pvalue;
    cut_id_pvalue=sort(cut_id_pvalue);
    %PPvalue=PPvalue;
else
    phi_pvalue=phi;
    reg_id_pvalue=ori_reg_id;
    cut_id_pvalue=[];
%     theta_pvalue=pinv(phi_pvalue'*phi_pvalue)*phi_pvalue'*X;
%     cons=[zeros(1,size(phi_pvalue,2)-extra_reg_num),0,-1];
%     theta_pvalue=lsqlin(phi_pvalue,X,cons,0,[],[],[],[],[],options);
    theta_pvalue=theta;
    %PPvalue(1)=pvaluefun_t(phi_pvalue,X,theta_pvalue,1); % p-value for lambda
    %PPvalue(2)=pvaluefun_t(phi_pvalue,X,theta_pvalue,2); % p-value for k
%     Pvalue_adj=Pvalue.*extra_reg_num;
end