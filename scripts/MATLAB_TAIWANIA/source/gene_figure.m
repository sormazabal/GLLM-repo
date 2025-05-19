genename='MYC';

choose_pool=1;
time=[0.5 1 2 4 8 16 24 48 72 96];

    x=strcmp(c_name,genename);
    a=[];
    a1=[];
    a2=[];
    a3=[];
    gene_profile=c_pool(x,:);
a(:,1)=mean(gene_profile(:,1:3),2);
a(:,2)=mean(gene_profile(:,4:6),2);
a(:,3)=mean(gene_profile(:,7:9),2);
a(:,4)=mean(gene_profile(:,10:12),2);
a(:,5)=mean(gene_profile(:,13:15),2);
a(:,6)=mean(gene_profile(:,16:18),2);
a(:,7)=mean(gene_profile(:,19:21),2);
a(:,8)=mean(gene_profile(:,22:24),2);
a(:,9)=mean(gene_profile(:,25:27),2);
a(:,10)=mean(gene_profile(:,28:30),2);
%     figure
%     plot(sum(a1,1));
    
a1=sum(a,1);
a=[];
   x=strcmp(m_name,genename);
   gene_profile=m_pool(x,:);
a(:,1)=mean(gene_profile(:,1:3),2);
a(:,2)=mean(gene_profile(:,4:6),2);
a(:,3)=mean(gene_profile(:,7:9),2);
a(:,4)=mean(gene_profile(:,10:12),2);
a(:,5)=mean(gene_profile(:,13:15),2);
a(:,6)=mean(gene_profile(:,16:18),2);
a(:,7)=mean(gene_profile(:,19:21),2);
a(:,8)=mean(gene_profile(:,22:24),2);
a(:,9)=mean(gene_profile(:,25:27),2);
a(:,10)=mean(gene_profile(:,28:30),2);
    
%     figure
%     plot(time,a2(a,1)); 
a2=sum(a,1);
a3=a2./a1;
plot(a3);

