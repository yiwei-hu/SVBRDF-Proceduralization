function write_gnbe_file_v2(filename, M, N, mu, Q, beta, phase, comments)


% Function to write a gnbe file (sum of Gaussians)
% set of Gaussians with their bandwidth, weight, and frequency index, and
% phase.
% Comments should already contain the symbol # at the beginning of each
% line and a linebreak at the end of the comments. (for now we don't use
% any comments...).


if(nargin < 7)
    error('There is not enough input arguments');
end

% Open file
fileid = fopen(filename, 'w');

% First line: filetype
fprintf(fileid, 'GNBE');

% version number
fprintf(fileid, '\n2');

% Write comments
if (nargin == 8)
    fprintf(fileid, comments);
end

% width and height of the image
fprintf(fileid, ['\n', num2str(N),' ',num2str(M)]);
% Be careful with matlab convention which is different from the common one.
% Mean of the image
fprintf(fileid, ['\n',num2str(mu)]);


% number of Gaussians:
ng = sum(sum(beta>0))/2;
disp(['Number of Gaussians for ', filename , ': ',num2str(sum(ng))]);

% number of octaves:
noctaves = sum(ng > 0);
fprintf(fileid, ['\n',num2str(noctaves)]);
disp(['Number of bandwidths for ', filename , ': ',num2str(noctaves)]);

% display sparsity:
sparsity = sum(ng)/(M*N);
disp(['Sparsity for ', filename , ': ',num2str(sparsity, '%e')]);


% For each bandwidth:
for i=1:length(Q)
    q = Q(i);
    if(ng(i)>0)
        % Bandwidth index and number of Gaussians:
        fprintf(fileid, ['\n',num2str(q)]);
        fprintf(fileid, ['\n',num2str(ng(i))]);
        % Sequential description of each Gaussians:
        for n = 1:N
            for m = 1:M
                if(beta(m,n,i)>0)
                   k = m - M/2 - 1;
                   l = n - M/2 -1;
                   if((l>0)||((l==0)&&(k<0)))
                       fprintf(fileid, ['\n', num2str(l),' ',...
                           num2str(-k),' ',num2str(2*beta(m,n,i), '%e'),' ',...
                           num2str(phase(m, n), '%e')]);
                       
                       
                   end
                end
            end
        end
    end
end

% Close file:
fclose(fileid);

end
