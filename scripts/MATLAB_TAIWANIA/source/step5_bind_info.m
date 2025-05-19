
% current directory: /m2_aae/utils/
%% read tf and gene association
x=textread('../data/raw/colon/human_PPI.txt','%s');
tt=length(x)/2;
binr = reshape(x,2,tt)'; 

%% doing ANOVA
bioMarker = 'PROM1';
name_pool_dir = strcat('../data/colon/resplit/ANOVA/os/',bioMarker,'_name_pool.mat');
load(name_pool_dir)
name_pool = name_pool';
name_pool = cellstr(name_pool);  % converting char array to cell

%% choose the candidate

bind_tf_Name=unique(binr(:,1));
bind_gene_Name=unique(binr(:,2));

bn={};
tgf_gene_prof=[];
cn={};
tgf_tf_prof=[];
leng=length(bind_gene_Name);

%% Bind info
bn=name_pool;
cn=name_pool;
 
bind=zeros(length(cn),length(bn));
leng=length(cn);

%%
for i=1:length(cn)
	%clc;
    display(['construct tf to gene table for ' sprintf('%3.1f%% in %d',roundn(i/leng*100,-1),leng)]);    
    x=strcmpi(cn(i),binr(:,1));
    s=binr(x,2);
    
    g=zeros(length(bn),1);
    for j=1:length(s)
        g=g+strcmpi(bn,s(j));  
    end
    bind(i,:)=g'; 
end
    
bind=bind';

save_dir = strcat('../data/colon/resplit/PPI/os/', bioMarker);
mkdir(save_dir)
cd(save_dir)
save 'bind_info.mat' 'bind'
cd('/mnt/299887b6-5a53-4144-9d50-b01beaec83be/m2_aae/utils')
