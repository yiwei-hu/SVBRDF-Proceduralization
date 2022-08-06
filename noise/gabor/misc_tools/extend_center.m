function Y = extend_center(X, s)

% Y = extract_center(X, s) returns the matrix which is equal to X in its
% center and to zero on its border.


% If M is even, the index of the center is M/2+1 (index of 0 in Fourier)
% If M is odd, the index of the center is (M+1)/2
% In both cases it is equal to floor(M/2)+1.



if (length(s)~=2)
    error('[extract_center] the lenght of the size array s must be 2');
end
if (sum((size(X) > s)) ~= 0)
    error('[extract_center]: the size s is smaller than the size of X');
end


K = size(X,1);
L = size(X,2);
pk = floor(K/2);
pl = floor(L/2);

M = s(1);
N = s(2);
cm = floor(M/2) + 1;
cn = floor(N/2) + 1;

Y = zeros(M,N);

Y( (cm-pk):(cm-pk+K-1) , (cn-pl):(cn-pl+L-1) ) = X;




end


