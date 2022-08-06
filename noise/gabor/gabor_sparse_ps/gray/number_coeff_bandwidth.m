function out = number_coeff_bandwidth(M,N,k)

% compute the number of coefficients for a bandwidth with grid number k.


% Extremas of the grid:
egM = floor((M-1)/2);
egN = floor((N-1)/2);

out = 1/2*( (1+2*floor(egM/k))*(1+2*floor(egN/k)) - 1 );

end




