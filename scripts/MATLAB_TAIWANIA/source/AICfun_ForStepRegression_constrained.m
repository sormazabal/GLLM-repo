function [theta_AIC,phi_AIC,reg_id_AIC,cut_id_AIC,AIC_value] = ...
    AICfun_ForStepRegression_constrained(Y,phi,theta,ori_reg_id)
% Akaike Information Criterion Forward selection method.
% given Y, phi, and the regulator IDs before AIC
% calculate AIC and output theta_AIC, regulator IDs after AIC, regulator IDs cutting by AIC, and AIC value

cut_id_AIC=ori_reg_id;
reg_id_AIC=[];
ori_phi=phi;

extra_num=size(phi,2)-length(ori_reg_id);
phi_extra=phi(:,(size(phi,2)-extra_num+1):(size(phi,2)));
theta_extra=theta((size(phi,2)-extra_num+1):(size(phi,2)));
sigma_2=(Y-phi_extra*theta_extra)'*(Y-phi_extra*theta_extra)/length(Y);
AIC_ori=log(sigma_2)+2*length(theta_extra)/length(Y);
options=optimset('LargeScale','off');

for x=1:(length(ori_reg_id))^2
    for z=1:length(cut_id_AIC)
        phi_modi=cat(2,phi(:,z),phi_extra);
        phi_modi_sav(:,:,z)=phi_modi;
%         theta_modi=pinv(phi_modi'*phi_modi)*phi_modi'*Y;
        %cons=[zeros(1,size(phi_modi,2)-extra_num),0,-1];
        cons=[zeros(1,size(phi_modi,2)-extra_num)];
        theta_modi=lsqlin(phi_modi,Y,cons,0,[],[],[],[],[],options);
        sigma_2_modi=(Y-phi_modi*theta_modi)'*(Y-phi_modi*theta_modi)/length(Y);
        AIC_modi(z)=log(sigma_2_modi)+2*length(theta_modi)/length(Y);
        clear phi_modi theta_modi sigma_2_modi cons
    end
    if (isempty(cut_id_AIC)==0) && (min(AIC_modi) < AIC_ori) 
        [min_value min_index]=min(AIC_modi);
        phi_extra=phi_modi_sav(:,:,min_index);
        AIC_ori=min_value; 
        reg_id_AIC=cat(2,cut_id_AIC(min_index),reg_id_AIC);
        cut_id_AIC(min_index)=[];    
        phi(:,min_index)=[];
        
        %cons2=[zeros(1,size(phi_extra,2)-extra_num),0,-1];
        cons2=[zeros(1,size(phi_extra,2)-extra_num)];
        theta_extra=lsqlin(phi_extra,Y,cons2,0,[],[],[],[],[],options);
        if (size(reg_id_AIC,2)>1)
            [theta_extra2,phi_extra2,reg_id_AIC2,cut_id_AIC2,AIC_value2]=AICfun_backward_constrained(Y,phi_extra,theta_extra,reg_id_AIC);
        else
            theta_extra2=theta_extra;
            phi_extra2=phi_extra;
            reg_id_AIC2=reg_id_AIC;
            cut_id_AIC2=[];
            AIC_value2=AIC_ori;
        end
        if (isempty(cut_id_AIC2)==1)
            reg_id_AIC=reg_id_AIC;
            cut_id_AIC=cut_id_AIC;
            phi_extra=phi_extra;
        else
            for k=1:length(cut_id_AIC2)
                index(k)=find(reg_id_AIC==cut_id_AIC2(k));
            end
            phi=cat(2,phi_extra(:,index),phi);
            cut_id_AIC=cat(2,reg_id_AIC(index),cut_id_AIC);
            reg_id_AIC=reg_id_AIC2;
            phi_extra=phi_extra2;
            AIC_ori=AIC_value2;
        end        
        clear phi_modi_sav AIC_modi cons2 index theta_extra2 phi_extra2 reg_id_AIC2 cut_id_AIC2 AIC_value2
    elseif (isempty(cut_id_AIC)==0) && (min(AIC_modi) >= AIC_ori) 
        break        
    else
        break              
    end      
end
AIC_value=AIC_ori;
cut_id_AIC=cut_id_AIC;
reg_id_AIC=sort(reg_id_AIC);
if isempty(cut_id_AIC)==0
    phi_AIC=ori_phi;
    for q=1:length(cut_id_AIC)
        judge(q)=find(ori_reg_id==cut_id_AIC(q));
    end
    phi_AIC(:,judge)=[];
else
    phi_AIC=ori_phi;
end
% theta_AIC=pinv(phi_AIC'*phi_AIC)*phi_AIC'*Y;
%cons=[zeros(1,size(phi_AIC,2)-extra_num),0,-1];
cons=[zeros(1,size(phi_AIC,2)-extra_num)];
if isempty(phi_AIC)~=1
    theta_AIC=lsqlin(phi_AIC,Y,cons,0,[],[],[],[],[],options);
else
    theta_AIC=[];
end