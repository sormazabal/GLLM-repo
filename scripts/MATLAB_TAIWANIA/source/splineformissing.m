function data_expre_c=splineformissing(data_expre,time)

nan=isnan(data_expre);
nansum=sum(nan,2);
  
for a=1:size(data_expre,1)
    z=1;
    if nansum(a)>0
        tt=zeros(1,(size(data_expre,2)-nansum(a)));
        gene0=zeros(1,(size(data_expre,2)-nansum(a)));
        for b=1:size(data_expre,2)
            if nan(a,b)==0
                tt(z)=time(b);
                gene0(z)=data_expre(a,b);
                z=z+1;
            else
            end
        end
        gene(a,:)=spline(tt,gene0,time);
        clear tt gene0;
    else
        gene(a,:)=data_expre(a,:);
    end
end

data_expre_c=gene;

