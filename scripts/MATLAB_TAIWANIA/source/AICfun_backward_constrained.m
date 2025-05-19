function [theta_AIC,phi_AIC,reg_id_AIC,cut_id_AIC,AIC_value]=AICfun_backward_constrained(Y,phi,theta,ori_reg_id)
% Akaike Information Criterion Backward elimination method.
% given Y, phi, theta and the regulator IDs before AIC
% calculate AIC and output theta_AIC, regulator IDs after AIC, regulator IDs cutting by AIC, and AIC value

extra_num=size(phi,2)-length(ori_reg_id);
sigma_2=(Y-phi*theta)'*(Y-phi*theta)/length(Y);
AIC_ori=log(sigma_2)+2*length(theta)/length(Y);
reg_id_AIC=ori_reg_id;
cut_id_AIC=[];
options=optimset('LargeScale','off');

for x=1:length(ori_reg_id)
    for z=1:length(reg_id_AIC)
        phi_modi=phi;
        phi_modi(:,z)=[];
        phi_modi_sav(:,:,z)=phi_modi;
         %theta_modi=pinv(phi_modi'*phi_modi)*phi_modi'*Y;
        %cons=[zeros(1,size(phi_modi,2)-extra_num),0,-1];
        cons=[zeros(1,size(phi_modi,2)-extra_num)];
        theta_modi=lsqlin(phi_modi,Y,cons,0,[],[],[],[],[],options);
        sigma_2_modi=(Y-phi_modi*theta_modi)'*(Y-phi_modi*theta_modi)/length(Y);
        AIC_modi(z)=log(sigma_2_modi)+2*length(theta_modi)/length(Y);
        clear phi_modi theta_modi sigma_2_modi cons
    end
    if (isempty(reg_id_AIC)==0) && (min(AIC_modi) < AIC_ori) 
        [min_value min_index]=min(AIC_modi);
        phi=phi_modi_sav(:,:,min_index);
        AIC_ori=min_value;
        cut_id_AIC=cat(2,cut_id_AIC,reg_id_AIC(min_index));
        reg_id_AIC(min_index)=[];
        clear phi_modi_sav AIC_modi
    elseif (isempty(reg_id_AIC)==0) && (min(AIC_modi) >= AIC_ori) 
        break
    else
        break
    end        
end
AIC_value=AIC_ori;
cut_id_AIC=sort(cut_id_AIC);
reg_id_AIC=reg_id_AIC;
phi_AIC=phi;
% theta_AIC=pinv(phi_AIC'*phi_AIC)*phi_AIC'*Y;
%cons=[zeros(1,size(phi_AIC,2)-extra_num),0,-1];
cons=[zeros(1,size(phi_AIC,2)-extra_num)];
theta_AIC=lsqlin(phi_AIC,Y,cons,0,[],[],[],[],[],options);
