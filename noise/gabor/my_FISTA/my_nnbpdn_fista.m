function x = my_nnbpdn_fista(A, At, m, n, b, lambda, x0, L, quiet, reltol)

% Solve the Non-Negative Basis Pursuit Denoising problem with the FISTA
% algorithm (with constant stepsize).
% The algorithm ended when the relative duality gap is smaller than reltol.
%
% References:
%  - For FISTA:
% 1) A. Beck and M. Teboulle
% Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems
% SIAM J. Imaging Sciences, Vol. 2 (2009), 183 -- 202
% http://dx.doi.org/10.1137/080716542
%  - For the stopping criterion:
% 2)  S.-J. Kim, K. Koh, M. Lustig, S. Boyd, and D. Gorinevsky.
% An interior-point method for large-scale l1 -regularized least squares.
% IEEE J. Selected Topics in Signal Processing, 2007.
% http://dx.doi.org/10.1109/JSTSP.2007.910971

% Initialization of y and t
x = x0;
y = x;
t = 1;
nbiter = 0;
maxG = 0; % for nu = 0, G(nu) = 0
[Fx, maxG, rel_gap] = evaluate_duality_gap(A, At, b, lambda, x, maxG);
if(quiet~=1) % first display:
    disp(sprintf('%5s %15s %15s %15s', 'iter','rel_gap','Fx','maxG'));
    disp(sprintf('%4d %15.5e %15.5e %15.5e', nbiter, rel_gap, Fx, maxG));
end


while(rel_gap > reltol)
    % FISTA iteration:
    Ay = A*y;
    gradfy = At*(2*(Ay-b));
    pLy = non_neg_soft_threshold(y-1/L*gradfy, lambda/L);
    xold = x;
    x = pLy;
    told = t;
    t = (1+sqrt(1+4*t^2))/2;
    y = x + (told-1)/t*(x-xold);
    
    % Evaluate duality gap:
    [Fx maxG rel_gap] = evaluate_duality_gap(A, At, b, lambda, x, maxG);
    
    nbiter = nbiter + 1;
    % Display on screen:
    if(quiet ~= 1)
        disp(sprintf('%4d %15.5e %15.5e %15.5e', nbiter, rel_gap, Fx, maxG));
    end
end

end


function [Fx, newmaxG, rel_gap] = evaluate_duality_gap(A, At, b, lambda, x, maxG)

% Evaluate F(x):
z = A*x-b;
Fx = sum(z.^2) + lambda*sum(x); % x <= 0

% Compute nu the associated dual variable and evaluate G(nu)
nu = 2*z;
rho = max(max(At*(-nu),0));
if (rho > lambda)
    nu = lambda/rho*nu;
end
Gnu = -0.25*sum(nu.^2) - nu'*b;
newmaxG = max(Gnu, maxG);

% Compute the relative duality gap:
rel_gap = (Fx-maxG)/maxG;

end


function out = non_neg_soft_threshold(x, tau)
    out = max(x - tau, 0);
end





