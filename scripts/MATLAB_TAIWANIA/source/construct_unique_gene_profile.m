% % a=[];
% % for i=1:length(gn)
% %     display(i)
% %     x=strcmp(gn(i),gene_name);
% %     s=p(x);
% %     m=find(s==max(s));
% %     gp=gene_profile(x,:);
% %     gp=gp(m,:);
% %     a=[a;gp];
% % end
% 
% b=[];
% bn={};
% for i=1:length(geneName)
%     i
%     x=strcmp(gn,geneName(i));
%     b=[b;a(x,:)];
%     bn=[bn;gn(x)];
% end
% 
% c=[];
% cn={};
% for i=1:length(tfName)
%     i
%     x=strcmp(gn,geneName(i));
%     c=[c;a(x,:)];
%     cn=[cn;gn(x)];
% end
bind1=[];
tf_bind={};
for i=1:length(cn)
    i
    x=strcmp(tfName,cn(i));
    bind1=[bind1 ;bind(x,:)];
    tf_bind=[tf_bind tfName(x)];
end

bind2=[];
gene_bind={};
for i=1:length(bn)
    i
    x=strcmp(geneName,bn(i));
    bind2=[bind2 bind1(:,x)];
    gene_bind=[gene_bind geneName(x)];
end
