function [alpha, alpha_binary] = knn_matting_spectra_core_masked(im, mask, spectra, user_inputs, spectra_weight)
% load KNN lib
run('vlfeat-0.9.20/toolbox/vl_setup');
% parameters
lambda = 100;
level = 1;
h = 3;

%% compute laplacian
[m, n, dm] = size(im);
[ms, ns, ds] = size(spectra);
d = dm + ds;
assert(m == ms && n == ns);

n_feat = [10; 2];
[x, y] = ind2sub([m n], 1:m*n);
features = [reshape(im, m*n, dm)'; reshape(spectra, m*n, ds)'*spectra_weight; [x; y] / sqrt(m*m+n*n)*level + rand(2, m*n)*1e-6];
% RGB; spectra * spectra_coef; XY*level

mask = reshape(mask, m*n, 1);
features = features(:, mask);
idx = 0;
S = sum(mask);

for i = 1 : size(n_feat, 1)
    ind = vl_kdtreequery(vl_kdtreebuild(features), features, features, 'NUMNEIGHBORS', n_feat(i), 'MAXNUMCOMPARISONS', n_feat(i)*2);
    index1 = reshape(repmat(uint32(1:S), n_feat(i), 1), [], 1);
    index2 = reshape(ind, [], 1);
    knn_pairs(idx + 1 : idx + S*n_feat(i), :) = [min(index1, index2) max(index1, index2)];
    features(d + 1:d + 2, :) = features(d + 1:d + 2, :) / 100; %reduce the weights of XY
    idx = idx + S*n_feat(i);
end

value = max(1 - sum(abs(features(:, knn_pairs(:, 1)) - features(:, knn_pairs(:, 2)))) / (d + 2), 0);
A = sparse(double(knn_pairs(:, 1)), double(knn_pairs(:, 2)), value, S, S);
A = A + A';
D = spdiags(sum(A, 2), 0, S, S);

%% compute m
n_layers = size(user_inputs, 1);
map = squeeze(sum(user_inputs, 1));
map = reshape(map, m*n, 1);
map = map(mask);

%figure;
%subplot(1, 2, 1);imagesc(map);
%subplot(1, 2, 2);imagesc(im);
%pause(10);

alpha = zeros(m*n, n_layers);
M = D - A + lambda*spdiags(map, 0, S, S);
L = ichol(M);

for i = 1:n_layers
    v = squeeze(user_inputs(i, :, :));
    v = reshape(v, m*n, 1);
    v = v(mask);
    masked_alpha = pcg(M, lambda*v, 1e-10, 2000, L, L');
    alpha(mask, i) = masked_alpha;
end

alpha = min(max(reshape(alpha, m, n, n_layers), 0), 1);
alpha_binary = imbinarize(alpha);
end