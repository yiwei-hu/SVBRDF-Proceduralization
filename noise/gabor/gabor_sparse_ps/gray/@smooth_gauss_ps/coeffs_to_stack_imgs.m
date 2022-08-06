function imalpha = coeffs_to_stack_imgs(A, alpha)

% Function to obtain the stack of images from a set of coefficients alpha
% for the @smoth_gauss_pw class.
% The operation consists in doing a symmetry and interlacing zeros lines and
% rows when k(j) > 1.

% WARNING: there is a lot of memory reallocation, it might be faster in
% using for loops.

% Different sizes:
lQ = length(A.Q);
M = A.M;
N = A.N;
% Extremas of the grid:
egM = floor((M-1)/2);
egN = floor((N-1)/2);


% initialization:
imalpha = zeros(M,N,lQ);
% index of the first coeff of the bandwidth j:
indj = 1;

% for each bandwidth:
for j=1:lQ
    k = A.k(j);
    ncbj = A.NCB(j);
    % number of coefficient on each axis:
    nbcjM = (1+2*floor(egM/k));
    nbcjN = (1+2*floor(egN/k));
    Temp = zeros(nbcjM, nbcjN);
    % fill the coefficients of the left part:
    Temp(1:ncbj) = alpha(indj:(indj+ncbj-1));
    % complete by symmetry:
    Temp = Temp + rot90(Temp,2);
    % Insert the lines of zeros:
    if(k>1)
        % New size with interlaced zeros:
        wzM = nbcjM + (k-1)*(nbcjM-1);
        wzN = nbcjN + (k-1)*(nbcjN-1);
        Temp2 = zeros(wzM, nbcjN);
        for i = 1:nbcjM
            Temp2(1+k*(i-1),:) = Temp(i,:);
        end
        Temp = zeros(wzM, wzN);
        for i = 1:nbcjN
            Temp(:, 1+k*(i-1)) = Temp2(:,i);
        end
        % add lines of zeros on each side to have the wished size (2*egM+1) x
        % (2*egN+1)
        if(2*egM+1-wzM > 0)
           nl =  (2*egM+1-wzM)/2;
           Temp = [zeros(nl,size(Temp,2)) ; Temp ; zeros(nl,size(Temp,2))];
        end
        if(2*egN+1-wzN > 0)
           nr =  (2*egN+1-wzN)/2;
           Temp = [zeros(size(Temp,1), nr) , Temp , zeros(size(Temp,1), nr)];
        end
    end
    % add lines of zeros if the dimensions are even:
    if(mod(M,2)==0)
        Temp = [ zeros(1,size(Temp,2)) ; Temp];
    end
    if(mod(N,2)==0)
        Temp = [zeros(size(Temp,1),1), Temp];
    end
    
    % copy the final image in imalpha
    imalpha(:,:,j) = Temp;
    
    % increase indj for next bandwidth:
    indj = indj + ncbj;
end

end
