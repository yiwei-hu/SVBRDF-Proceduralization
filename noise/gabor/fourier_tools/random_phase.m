function rp = random_phase(M,N)

% rp = random_phase(M,N) compute the realization of a random phase of size (M,N).
% The frequencies are assumed to be placed with the fftshift(fft2()) convention, that is the frequency zero is at the center.
% M and N are supposed to be even.
% A random phase is a random antisymmetric image the elements of which are random angles drawn in (-pi,pi].
% Bruno Galerne

rp = zeros(M,N);

% Draw the random phase in the complex half-plane:
for m = 1:(M/2+1)
    for n=1:N
        % case 1 : (m,n) is its own symmetric: one draws random between 0 or pi.
        if( ((m==1)||(m==M/2+1)) && ((n==1)||(n==N/2+1)) )
            % for the frequence zero, we set rp=0 to keep the mean.
            if( (m==1) && (n==1) )
                rp(m,n) = 0;
            else
                rp(m,n) = pi*(rand(1) > 0.5);
            end
        % case 2 : the symmetric point of (m,n) is in the same half-plane
        % and its phase is already drawn.
        elseif( ((m==1)||(m==M/2+1)) && (n>N/2+1) )
            ns = N-n+2;
            rp(m,n) = -rp(m,ns);
        % generic case: one draws a uniform angle.
        else
            rp(m,n) = -pi + 2*pi*rand(1);
        end
    end
end


% completion of rp by symmetry.
for m = (M/2+2):M
    for n=1:N
        % computation of the coordinates of the symmetric point.
        ms = M-m+2;
        if(n==1)
            ns = 1;
        else
            ns = N-n+2;
        end
        rp(m,n) = -rp(ms,ns);
    end
end


% shift :
rp = fftshift(rp);

%  % Reference:
%  @article{Galerne_Gousseau_Morel_random_phase_textures_2011,
%  author = {B. Galerne and Y. Gousseau and J.-M. Morel},
%  title = {Random Phase Textures: Theory and Synthesis},
%  journal = {IEEE Trans. Image Process.},
%  year = {2011},
%  volume = {20},
%  number = {1},
%  pages = {257 -- 267}, 
%  doi = {10.1109/TIP.2010.2052822}
%  }




