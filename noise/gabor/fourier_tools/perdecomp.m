function [P, S]=perdecomp(U)
% P = PERDECOMP(U) computes Lionel Moisan's periodic component of U.
% [P, S] = PERDECOMP(U) computes Lionel Moisan's periodic plus smooth
% decomposition of U = P+S.
% Works for gray-valued and RGB color images. 
% Author : Bruno Galerne (2009/03/17)

% For the resolution of Poisson's equation Laplacian(U) = f see the book
% "Numerical Recipes: the Art of Scientific Computing"

% Image resolution:
[M, N, nc] = size(U);

% Compute LIU the interior Laplacian of U:
zc = zeros(M, 1, nc); % nul column
zr = zeros(1, N, nc); % nul row

ILU = [(U(1:(M-1),:,:) -U(2:M,:,:)) ; zr] + [zr ; (U(2:M,:,:) - U(1:(M-1),:,:))] +...
    [(U(:,1:(N-1),:) - U(:,2:N,:)), zc] + [zc , (U(:,2:N,:) - U(:,1:(N-1),:))];

% Fourier transform:
F = fft2(ILU);
clear ILU zc zr;

% Resolution in Fourier domain:
cx = 2*cos(2*pi/M*(0:(M-1)));
cy = 2*cos(2*pi/N*(0:(N-1)));
[CY, CX] = meshgrid(cy, cx);
C = 4*ones(M,N) - CX - CY;
clear CX CY cx cy;
C(1,1) = 1;
C = repmat(C, [1, 1, nc]);
F = F./C;
clear C;
% (O,O) frequency: we impose the same mean as U.
F(1,1,:) = sum(sum(U));

% Inverse Fourier transform:
P = real(ifft2(F));
clear F;
S = U-P;
end

% References :
%  @article {Moisan_periodic_plus_smooth_decomposition_2011,
%  author = {L. Moisan},
%  title = {Periodic Plus Smooth Image Decomposition},
%  journal = {J. Math. Imag. Vis.},
%  publisher = {Springer Netherlands},
%  pages = {161--179},
%  volume = {39},
%  issue = {2},
%  doi = {10.1007/s10851-010-0227-1},
%  year = {2011}
%  }

%  @book{Press_numerical_recipes_2007,
%  author = {W. H. Press and S. A. Teukolsky and W. T. Vetterling and B. P. Flannery},
%  title = {Numerical Recipes 3rd Edition: The Art of Scientific Computing},
%  year = {2007},
%  isbn = {0521880688, 9780521880688},
%  publisher = {Cambridge University Press},
%  address = {New York, NY, USA},
%  }




















