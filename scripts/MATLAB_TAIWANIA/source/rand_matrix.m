function matrix=rand_matrix(matrix) % transform a(i,j)=a(j,i)

    matrix=triu(matrix);

    for i=1:size(matrix,2)
        
        x=[]; 
%       for j=(i):size(matrix,2)      
        for j=(i+1):size(matrix,2)
            x=cat(2,x,matrix(i,j));
        end
        
        x_rand=x(randperm(numel(x)));
        
%       for k=(i):size(matrix,2)
        for k=(i+1):size(matrix,2)
             matrix(i,k)=x_rand(k-i);
%            matrix(i,k)=x_rand(size(matrix,2)-k+1);
        end
        
        clear x x_rand
    end
    
    matrix=trans(matrix);
    
end

%a= [1 2 33 4;0 2 3 4; 9 9 0 1; 0 0 2 3]