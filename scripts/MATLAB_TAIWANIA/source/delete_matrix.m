function [matrix_final_0,p]=delete_matrix(matrix) % transform a(i,j)=a(j,i)
    

    matrix=matrix-diag(diag(matrix));
    matrix=trans(matrix);
    matrix=triu(matrix);
    p=[];

    for i=1:size(matrix,2)
        if sum(matrix(i,:))~=0 || sum(matrix(:,i))~=0
            p=[p;i];
        end
    end
    
    matrix_final=[];
    matrix_final_0=[];
    
    for j=1:size(p,1)
        matrix_final=[matrix_final;matrix(p(j),:)];
    end
    
    for k=1:size(p,1)
       matrix_final_0=[matrix_final_0,matrix_final(:,p(k))];
    end
    
    matrix_final_0=trans(matrix_final_0);
    
end

%a=[0 0 0 0 0;0 1 0 2 0;0 0 0 0 0;0 1 1 0 0; 0 0 0 0 0]