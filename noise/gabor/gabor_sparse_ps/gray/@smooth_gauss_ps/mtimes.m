function out = mtimes(A,x)

% Multiplication for the @smooth_gauss_pw class.

% addpath('..');

if A.adjoint == 0 %A*x
    if (length(x)~=A.n)
        error('[mtimes for smooth_gauss_ps] Size problem for forward multiplication');
    end
    M = A.M;
    N = A.N;
    lQ = length(A.Q);
    imaplha = coeffs_to_stack_imgs(A, x);
    S = zeros(2*M,2*N);
    % sum of the DFT of the convolution:
    for j=1:lQ
        S = S + (fft2(extend_center(imaplha(:,:,j), 2*[M N]))).*(A.GDFT(:,:,j));
    end
    % inverse FFT and extract_center:
    out = extract_center(ifft2(S), [M,N]);
    out = out(:);
    
else %At*y
    if (length(x)~= A.m)
        error('[mtimes for smooth_gauss_ps] Size problem for adjoint multiplication');
    end
    M = A.M;
    N = A.N;
    lQ = length(A.Q);
    imalpha = zeros(M,N,lQ);
    dfty = fft2(extend_center(reshape(x,[M N]), 2*[M N]));
    for j=1:lQ
        imalpha(:,:,j) = extract_center(ifft2(dfty.*(A.GDFT(:,:,j))), [M,N]);
    end
    out = adj_coeffs_to_stack_imgs(A, imalpha);
end