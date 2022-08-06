function out = mean_pw_visualization(mmax, A)


% out = mean_pw_visualization(mmax, A) computes the average power spectrum
% of the series of images of A and apply a logarithmic tonemap with mmax as
% a max value.
% INPUT:
% - mmax: max value for the logarithmic tonemap
% - A: a family of grayscale images. A has size MxNxK where K is the number
% of images.
% OUTPUT:
% - out: the log tonemaped image (values larger than 1 are thresholded to
% 1, so that out has only values between 0 and 1).


% size of A:
M = size(A,1);
N = size(A,2);
K = size(A,3);

% compute mean power spectrum:
out = zeros(M,N);
for k=1:K
    out = out + (abs(periodic_dft_for_gabor(A(:,:,k)))).^2;
end
out = 1/K*out;

% logarithmic tonemap:
out = 1/log(1+mmax)*log(1+out);

% threshold values larger than 1:
out = min(out, 1);



end
















