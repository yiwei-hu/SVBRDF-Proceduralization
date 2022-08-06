function mp = periodic_dft_for_gabor(u)

% mp = periodic_dft_for_gabor(u) computes the DFT of the
% periodic component of u, and sets the highest frequencies to zero so as
% to satisfy a Shannon condition.
% The obtained DFT is centered (fftshift).

% Image size:
M = size(u,1);
N = size(u,2);

% Computes the interior Laplacian of u:
zc = zeros(M, 1); % null column
zr = zeros(1, N); % null row
Lu = [(u(1:(M-1),:) -u(2:M,:)) ; zr] +...
    [zr ; (u(2:M,:) - u(1:(M-1),:))] +...
    [(u(:,1:(N-1)) - u(:,2:N)) , zc] +...
    [zc , (u(:,2:N) - u(:,1:(N-1)))];

% Fourier transform :
mp = fft2(Lu);

% Inversion of the periodic Laplacian in Fourier domain:
cx = 2*cos(2*pi/M*(0:(M-1)));
cy = 2*cos(2*pi/N*(0:(N-1)));
[CY, CX] = meshgrid(cy, cx);
C = 4*ones(M,N) - CX - CY;
clear CX CY cx cy;
C(1,1) = 1;

mp = mp./C;
clear C;

% Special frequencies:

% null mean:
mp(1,1) = 0;

% Shannon condition :
mp = fftshift(mp);
if (mod(M,2) == 0) % if M is even we set the frequencies (-M/2,l) to 0.
    mp(1,:) = 0;
end
if (mod(N,2) == 0) % if N is even we set the frequencies (k,-N/2) to 0.
    mp(:,1) = 0;
end



