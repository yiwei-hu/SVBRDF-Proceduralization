function  res = ctranspose(A)

% ctranspose function for the class @smooth_gauss_ps. This is used by the
% times function.

A.adjoint = xor(A.adjoint, 1);
res = A;