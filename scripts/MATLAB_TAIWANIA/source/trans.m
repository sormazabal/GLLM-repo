function matrix=trans(matrix) % transform a(i,j)=a(j,i)

    for i=1:size(matrix,2)
        for j=i:size(matrix,2)
            if matrix(i,j)~=matrix(j,i)
                x=max(abs(matrix(i,j)),abs(matrix(j,i)));
                x1=max(matrix(i,j),matrix(j,i));
                if x==x1
                    matrix(i,j)=x;
                    matrix(j,i)=x;
                else
                    matrix(i,j)=-x;
                    matrix(j,i)=-x;
                end
            end
        end
    end

end