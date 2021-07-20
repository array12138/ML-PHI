function [S, D] = get_Kconstraints(data, labels, K_neigh)
% Input: data (n x d), labels (n x 1), N is the number of training pair
% Output: S(n x n), D(n x n)

% 1 compute the same_label matrix
[N, ~] = size(data);
assert(length(labels) == N);
[lablist, ~, labels] = unique(labels);
K = length(lablist);
label_matrix = false(N, K);
label_matrix(sub2ind(size(label_matrix), (1:length(labels))', labels)) = true;
same_label = logical(double(label_matrix) * double(label_matrix'));
    
%2 Select target neighbors
sum_X = sum(data .^ 2, 2);
[n,~]=size(data);
DD = bsxfun(@plus, sum_X, bsxfun(@plus, sum_X', -2 * (data * data')));
disi_DD = DD;
DD(~same_label) = Inf; DD(1:N + 1:end) = Inf;
[~, targets_ind] = sort(DD, 2, 'ascend');
targets_ind = targets_ind(:,1:K_neigh);

disi_DD(same_label) = Inf; 
[~, disi_ind] = sort(disi_DD, 2, 'ascend');
targets_disi = disi_ind(:,1:K_neigh);


D = zeros(n);
S = zeros(n);
for i=1:n
    for j=1:K_neigh
        S(i,targets_ind(j))=1;
        D(i,targets_disi(j))=1;
    end
end
end