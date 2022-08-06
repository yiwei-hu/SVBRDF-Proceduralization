function Y = extract_center(X, s)

% Y = extract_center(X, s) returns the matrix of size s = [M,N] corresponding
% of the center of the input matrix X.


% If M is even, the index of the center is M/2+1 (index of 0 in Fourier)
% If M is odd, the index of the center is (M+1)/2
% In both cases it is equal to floor(M/2)+1.

if (length(s)~=2)
    error('[extract_center] the lenght of the size array s must be 2');
end
if (sum((size(X) < s)) ~= 0)
    error('[extract_center]: the size s is larger than the size of X');
end

K = size(X,1);
L = size(X,2);
M = s(1);
N = s(2);
pm = floor(M/2);
pn = floor(N/2);

ck = floor(K/2) + 1;
cl = floor(L/2) + 1;

Y = X( (ck-pm):(ck-pm+M-1) , (cl-pn):(cl-pn+N-1) );

end
















