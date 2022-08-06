function res = smooth_gauss_ps(M, N, Q, eps)

% Function to create an element of the class @smooth_gauss_ps
% Input:
%  - M, N: size of the represented power spectrum
%  - Q: array of bandwidth indices (should be between 2 and
%  log2(max(M,N))
%  - eps: value for the incoherence between basis vectors, should be
%  around 0.05.

% List of attributes of the class @smooth_gauss_ps:
% - adjoint: Boolean for A or A' (used by the times function)
% - M and N: size of the corresponding image
% - Q: set of bandwidths
% - k: array of size length(Q) for the size of the grid of each bandwidth
% (if k=1, all frequencies, if k=2 1 freq. over 4...)
% - GDFT: three dimensional array of size 2*M x 2*N x length(Q) to store
% the DFT of the discrete Gaussian function of each bandwidth.
% - m and n: m x n: size of the abstract matrix A
% - NCB an array specifying the number of coefficients in each bandwidth
% (which depends on M, N and k(j) for each bandwidth q(j)).

% adjoint:
res.adjoint = 0;

% M, N: size of the image:
res.M = M;
res.N = N;

% Set of bandwidths Q:
res.Q = Q;

% Determination of the array k and computation of the Gaussian DFT.
lQ = length(Q);
k = ones(1, lQ);
GDFT = zeros(2*M, 2*N, lQ);

% compute the grid for the Gaussian
x = 1/M*((1:2*M)-(M+1));
y = 1/N*((1:2*N)-(N+1));
[Y X] = meshgrid(y,x);
for j = 1:length(Q)
    t = 0;    % translation vector
    dp = 1;   % dot product
    % image of the Gaussian with discrete l2 normalization:
    q = Q(j);
    sig = 2^(-q)/(2*sqrt(2*log(2)));
    G = exp(- (X.*X + Y.*Y)/(2*sig^2));
    G = G/norm(G(:), 2);
    % determination of k(j)
    while (dp > 1-eps)
        t = t+1;
        tGx = circshift(G,[t,0]);
        tGy = circshift(G,[0,t]);
        dp = max(dot(G(:),tGx(:)), dot(G(:),tGy(:)));
    end
    k(j) = t;
    % Computation of GDFT(:,:,j)
    G = fftshift(G);
    GDFT(:,:,j) = fft2(G);
end

res.k = k;
res.GDFT = GDFT;

% computation of the number of coefficients per bandwidth:
NCB = zeros(1, lQ);
for j=1:lQ
    NCB(j) = number_coeff_bandwidth(M,N,k(j));
end
res.NCB = NCB;

% computation of the number of the size of A
m = M*N;
n = sum(NCB);

res.m = m;
res.n = n;


res = class(res, 'smooth_gauss_ps');

end