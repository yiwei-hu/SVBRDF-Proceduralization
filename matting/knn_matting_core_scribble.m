function [alpha, alpha_binary] = knn_matting_core(im, user_inputs)
% load KNN lib
run('vlfeat-0.9.20/toolbox/vl_setup');
% parameters
lambda = 100;
level = 1;
h = 3;

%% compute laplacian
[m, n, d] = size(im);
n_feat = [10; 2];
[x, y] = ind2sub([m n], 1:m*n);
features = [reshape(im, m*n, d)'; [x; y] / sqrt(m*m+n*n)*level + rand(2, m*n)*1e-6]; % RGB + XY
idx = 0;

for i = 1 : size(n_feat, 1)
    ind = vl_kdtreequery(vl_kdtreebuild(features), features, features, 'NUMNEIGHBORS', n_feat(i), 'MAXNUMCOMPARISONS', n_feat(i)*2);
    index1 = reshape(repmat(uint32(1:m*n), n_feat(i), 1), [], 1);
    index2 = reshape(ind, [], 1);
    knn_pairs(idx + 1 : idx + m*n*n_feat(i), :) = [min(index1, index2) max(index1, index2)];
    features(d+1:d+2, :) = features(d+1:d+2, :) / 100; %reduce the weights of XY
    idx = idx + m*n*n_feat(i);
end

value = max(1 - sum(abs(features(:, knn_pairs(:, 1)) - features(:, knn_pairs(:, 2)))) / (d+2), 0);
A = sparse(double(knn_pairs(:, 1)), double(knn_pairs(:, 2)), value, m*n, m*n);
A = A + A';
D = spdiags(sum(A, 2), 0, n*m, n*m);

%% compute m
n_layers = size(user_inputs, 1);
map = squeeze(sum(user_inputs, 1));

%figure;
%subplot(1, 2, 1);imagesc(map);
%subplot(1, 2, 2);imagesc(im);

alpha = zeros(m * n, n_layers);
M = D - A + lambda*spdiags(map(:), 0, m*n, m*n);
L = ichol(M);

for i = 1:n_layers
    v = squeeze(user_inputs(i, :, :));
    alpha(:, i) = pcg(M, lambda*v(:), 1e-10, 2000, L, L');
end

alpha = min(max(reshape(alpha, m, n, n_layers), 0), 1);
alpha_binary = imbinarize(alpha);
end