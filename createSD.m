function [X_S,D,S] = createSD( X,Y )
%Create Similarity, Disimilarity and X S matrix of Xings paper
[n,d]=size(X);
X_S = zeros(d);
D = zeros(n);
S = zeros(n);
k_d= 1; k_s=1;
for i=1:(n-1)
    for j=i+1:n
        if (Y(i) == Y(j)) %% They are similar
            S(i,j)=1;
            k_s=k_s+1;
            x_i = X(i,:);
            x_j = X(j,:);
            X_S = X_S + (x_i-x_j)'*(x_i-x_j);
        else    %% Dissimilar
                % Col Diss = L
                % D(:,k) = (X(:,i) ? X(:,j));
            D(i,j)=1; %% Dissimilarity pairs
            k_d=k_d+1;
        end
    end
end
end