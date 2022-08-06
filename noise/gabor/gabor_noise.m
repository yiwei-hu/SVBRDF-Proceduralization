function psd = gabor_noise(h, oolratio)
% Add necessary paths:
addpath('fourier_tools/');
addpath('gabor_sparse_ps/gray/');
addpath('misc_tools/');
addpath('my_FISTA/');

if nargin == 1
    oolratio = 8; % [8 16 32 64 128 256 512 1024]
end

% mean of h:
mu = mean(h(:));

% Size of h: 
[M, N] = size(h);

% Compute "clean" discrete power spectrum of h:
dfth = periodic_dft_for_gabor(h);
s = abs(dfth).^2;
s = s(:);
phase = angle(dfth);

% Define the linear operator A:
Q = 2:(floor(log2(sqrt(M^2+(N^2)))));
lQ = length(Q);
eps = 0.05;
A  = smooth_gauss_ps(M,N,Q,eps); % A
At = A';                     % transpose of A
m = M*N;
n = get_n_smooth_gauss_ps(A);
lambda_max = find_lambdamax_nnbpdn(At, s);
L = get_lipschitz_constant(A, At, m, n, 1); % this may take some time.

disp(['Running NNBPDN with lratio = 1/', num2str(oolratio)]);

% run my_nnbpdn_fista
lambda  = (1/oolratio)*lambda_max; % regularization parameter
rel_tol = 0.02; % relative target duality gap
alpha0 = zeros(n,1); % initialize as zeros
alpha = my_nnbpdn_fista(A, At, m, n, s, lambda, alpha0, L, 1, rel_tol);
psd = A*alpha;
psd = reshape(psd, M, N);

end








