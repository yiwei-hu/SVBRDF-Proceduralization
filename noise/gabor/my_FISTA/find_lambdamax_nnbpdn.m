function lambda_max = find_lambdamax_nnbpdn(At, b)
%
% find_lambdamax_nnbpdn gives the maximal value of lambda above which
% x=0 is an optimal solution for the NNBPDN problem.
%


lambda_max = 2*max(max(At*b,0));

end


% Note: this is the same function as find_lambdamax_l1_ls_nonneg of the
% l1_ls matlab code:
% http://www.stanford.edu/~boyd/l1_ls/
