function alpha = adj_coeffs_to_stack_imgs(A, imalpha)

% This is the adjoint operator of coeffs_to_stack_imgs.m
% This is also twice the inverse operation of coeffs_to_stack_imgs (when
% restricting to the image of this function).

% Different sizes:
lQ = length(A.Q);
M = A.M;
N = A.N;
% Extremas of the grid:
egM = floor((M-1)/2);
egN = floor((N-1)/2);


% initialization:
alpha = zeros(A.n,1);
% index of the first coeff of the bandwidth j:
indj = 1;

% for each bandwidth:
for j=1:lQ
    k = A.k(j);
    ncbj = A.NCB(j);
    % number of coefficient on each axis:
    nbcjM = (1+2*floor(egM/k));
    nbcjN = (1+2*floor(egN/k));
    
    % copy the image of the bandwidth:
    Temp = imalpha(:,:,j);
    
    % remove outer lines if the dimensions are even:
    if(mod(M,2)==0)
        Temp = Temp(2:end,:);
    end
    if(mod(N,2)==0)
        Temp = Temp(:,2:end);
    end
    if(k>1)
        % Size with only inner interlaced zeros:
        wzM = nbcjM + (k-1)*(nbcjM-1);
        wzN = nbcjN + (k-1)*(nbcjN-1);
        % Remove outter lines (of zeros):
        if(2*egM+1-wzM > 0)
           nl =  (2*egM+1-wzM)/2;
           Temp = Temp((nl+1):(end-nl),:);
        end
        if(2*egN+1-wzN > 0)
           nr =  (2*egN+1-wzN)/2;
           Temp = Temp(:,(nr+1):(end-nr));
        end
        % Remove inner interlaced lines (of zeros):
        Temp2 = zeros(wzM, nbcjN);
        for i = 1:nbcjN
            Temp2(:,i) = Temp(:, 1+k*(i-1));
        end
        Temp = zeros(nbcjM, nbcjN);
        for i = 1:nbcjM
            Temp(i,:) = Temp2(1+k*(i-1),:);
        end
    end
    % sum of Temp and its symmetric part: (adjoint of the symmetrization)
    Temp = Temp + rot90(Temp,2);
    % copy the coefficient of the bandwidth in alpha:
    alpha(indj:(indj+ncbj-1)) = Temp(1:ncbj);    
    % increase indj for next bandwidth:
    indj = indj + ncbj;
end









end





