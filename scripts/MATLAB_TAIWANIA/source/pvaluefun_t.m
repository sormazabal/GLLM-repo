function p_value=pvaluefun_t(phi,X,theta,q)
 
n=size(phi,2);   % number of parameters
m=size(phi,1);   % number of time points

dof=m-n-1; % degree of freedom
SSE=(X-phi*theta)'*(X-phi*theta);
s=sqrt(SSE/dof);

phiphi_inv=pinv(phi'*phi);
c=phiphi_inv(q,q);

real_value=theta(q);

cutvalue=abs(real_value/(s*sqrt(c)));

p_value=2*(1-tcdf(cutvalue,dof));

if p_value==0
    p_value=1e-16;
else p_value=p_value;
end



