function [ M,weight] = sample_weight_metriclearning(data,label,parameter)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

MaxIter = 100;
[n_sam,n_fea] = size(data);
 [S, D] = get_Kconstraints(data, label, parameter.K);
% [D,S] = createSD(data,label);
Id_matrix = eye(n_fea,n_fea);
M = Id_matrix;
weight =  ones(n_sam,1);
DD_matrix = zeros(n_sam,n_sam);
for i = 1:n_sam%这个地方错了 
    DD_matrix(i,i) = sqrt(sum(data(i,:).^2));
end
a_vec = sum(DD_matrix,2);

for iter = 1:MaxIter
    % Update M
    M_old = M;
    weight_old = weight;
    [XD,XS] = compute_XD_XS(data,S,D,weight);
    M = compute_AjingB(inv(XS  + parameter.gamma * Id_matrix), XD +  parameter.gamma * Id_matrix,parameter.t);
    
    [QD,QS] = compute_QD_QS(data,S,D,M);
    H = QD + QS +  parameter.mu * DD_matrix;
    weight = pinv(H) * parameter.mu * parameter.theta * a_vec;

    Dsld = trace(M) + trace(inv(M))-2*n_fea;
    obj1 = parameter.gamma * Dsld ;
    obj2 =  0.5 * weight' * (QS+QD)*weight;
    obj3 = parameter.mu * (0.5 * weight' * DD_matrix * weight - parameter.theta * weight'* a_vec);
    
    obj = obj1 + obj2 + obj3;
    disp(['obj = ',num2str(obj), ';obj1= ',num2str(obj1),'; obj2=',num2str(obj2),'; obj3=',num2str(obj3)]);

    if iter ==1
        obj_max = obj;
    else
%         disp(['obj = ',num2str(obj),'; obj_max=',num2str(obj_max)]);
        if abs(obj-obj_max)/max(obj_max,1)<=1e-5
            disp(['abs(obj-obj_max)/max(obj_max,1) = ',num2str(abs(obj-obj_max)/max(obj_max,1))]);
            break;
        end
        if obj>obj_max
            M = M_old;
            weight = weight_old;
            disp(['obj>obj_max','; obj=',num2str(obj),'; obj_max=',num2str(obj_max)]);
            break;
        else
            obj_max = obj;
        end
    end
end
end
function  matrix = compute_AjingB(A,B,t)
    A = A^0.5;
    A_ = A^(-0.5);
    matrix = A*(A_*B*A_)^t*A;
end
function [XD,XS] = compute_XD_XS(data,S,D,weight)
% compute \sum_{{i,j}\in S}(w_ix_i-w_jx_j)(w_ix_i-w_jx_j)^T
% compute \sum_{{i,j}\in D}(w_ix_i-w_jx_j)(w_ix_i-w_jx_j)^T
[n_fea] = size(data,2);
XD = zeros(n_fea,n_fea);
XS = zeros(n_fea,n_fea);
[X_Si,X_Sj] = find(S==1);
for i = 1:length(X_Si)
     one_vec = weight(X_Si(i)) * data(X_Si(i),:)- weight(X_Sj(i)) * data(X_Sj(i),:);
     XS = XS + one_vec' * one_vec;
end
[X_Di,X_Dj] = find(D==1);
for i = 1:length(X_Di)
     one_vec = weight(X_Di(i)) * data(X_Di(i),:)- weight(X_Dj(i)) * data(X_Dj(i),:);
     XD = XD + one_vec' * one_vec;
end
end

function [QD,QS] = compute_QD_QS(data,S,D,M)
[n_sam,~] = size(data);
S = S + S';
D = D + D';
QD_matrix = zeros(n_sam,n_sam);
QS_matrix = zeros(n_sam,n_sam);
D_diag = sum(D);
L_D = diag(D_diag) - D;
S_diag = sum(S);
L_S = diag(S_diag) - S;
M_inv = pinv(M);
for i = 1:n_sam
     one = data(i,:);
     QS_matrix(i,i) = one * M * one';
     QD_matrix(i,i) = one * M_inv * one';
end
[X_Si,X_Sj] = find(S == 1);
for i = 1:length(X_Si)
    QS_matrix(X_Si(i),X_Sj(i)) = data(X_Si(i),:) * M * data(X_Sj(i),:)';
end
[X_Di,X_Dj] = find(D == 1);
for i = 1:length(X_Di)
    QD_matrix(X_Di(i),X_Dj(i)) = data(X_Di(i),:) * M_inv * data(X_Dj(i),:)';
end
QD = QD_matrix.*L_D;
QS = QS_matrix.*L_S;
end