function mean_pw_visualization_io(mmax, outputname, varargin)

% mean_pw_visualization_io is the input/output function for calling
% mean_pw_visualization from a script:
% the syntax is:
% mean_pw_visualization_io(mmax, 'outputname.png', 'imagename1.png', 'imagename2.png',...)


if(isempty(varargin))
    error('there must be at least one image');
end

% check on mmax:
if (mmax<=0)
    error('mmax should be positive');
end

% number of images:
K = size(varargin,2);

% read each image and check that they have the same size:
% first image:
%  im = double(imread(varargin{1})); % AL
im = double(rgb2gray(imread(varargin{1}))); % AL
M = size(im,1);
N = size(im,2);
A = zeros(M,N,K);
A(:,:,1) = im;
% other images:
if (K>1)
    for k = 2:K
        %im = double(imread(varargin{k}));  % AL
        im = double(rgb2gray(imread(varargin{k})));
        % check size:
        if( (M ~= size(im,1)) || N ~= size(im,2))
            error('All images must have the same size');
        end
        A(:,:,k) = im;
    end
end

% call mean_pw_visualization

%out = mean_pw_visualization(mmax, A); % AL
out = mean_pw_visualization(mmax * M * N * M * N, A); % AL

% write out:
imwrite(uint8(255*out), outputname);

exit;

end
