function [S, D] = get_training_data(Examples, Labels, N)
% Input: Examples (n x d), Labels (n), N is the number of training pair
% Output: S(n x n), D(n x n)
    [n] = size(Examples,1);
    indexs = randperm(n^2, N);
    S = zeros(n,n);
    S = sparse(S);
    D = zeros(n,n);
    D = sparse(D);
    for i=1: length(indexs)
        u = indexs(i)/n;
        v = mod(indexs(i), n);
        p = floor(u) + (v~=0);
        q = v + 1;
        if Labels(p) == Labels(q)
            if p<q
                S(p,q) = 1;
            else
                S(q,p) = 1;
            end
        else
            if p<q
                D(p,q) = 1;
            else
                D(q,p) = 1;
            end
        end
    end
end