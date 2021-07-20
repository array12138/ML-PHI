function [ M,weight] = sample_weight_metriclearning(data,label,parameter)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

MaxIter = 100;
[n_sam,n_fea] = size(data);
 [S, D] = get_Kconstraints(data, label, parameter.K);
% [D,S] = createSD(data,label);
Id_matrix = eye(n_fea,n_fea);
M = Id_matrix;
weight = ones(n_sam,1);
weight0 = weight;
DD_matrix = zeros(n_sam,n_sam);
for i = 1:n_sam
    DD_matrix(i,i) = sum(sqrt(data(i,:).^2));
end

for iter = 1:MaxIter
    % Update M
    M_old = M;
    weight_old = weight;
    [XD,XS] = compute_XD_XS(data,S,D,weight);
    M = compute_AjingB(inv(XS  + parameter.gamma * Id_matrix), XD +  parameter.gamma * Id_matrix,parameter.t);
    
    [QD,QS] = compute_QD_QS(data,S,D,M);
    H = QD + QS +  2 * parameter.mu * DD_matrix;
    H = (H + H')/2;
    f = 2*parameter.mu * DD_matrix * weight0;
    AA = zeros(n_sam + 1,n_sam+1);
    AA(1:n_sam,1:n_sam) = H;
    AA(1:n_sam,n_sam+1) = ones(n_sam,1);
    AA(n_sam+1,1:n_sam) = ones(1,n_sam);
    AA(n_sam+1,n_sam+1) = 0;
    bb = zeros(n_sam+1,1);
    bb(1:n_sam,1) = f;
    bb(n_sam+1,1) = n_sam;
    tic;
    solve = pinv(AA)*bb;
    toc;
%     solve = Gauss(AA,bb);
    weight = solve(1:n_sam);
    
%     options = optimoptions(@quadprog,'Algorithm','active-set');
% %     disp('add the shang jie');
%     [weight,~,~,~,~] = quadprog(H,f,[],[],ones(1,n_sam),n_sam,zeros(n_sam,1),[],weight0,options);
% %     weight = 2* mu * inv(H) * weight0;
% %     plot(weight);

    Dsld = trace(M) + trace(inv(M))-2*n_fea;
    obj1 = parameter.gamma * Dsld ;
    obj2 =  0.5 * weight' * (QS+QD)*weight;
    obj3 = parameter.mu * (weight-weight0)'*(weight-weight0);
    
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
[n_sam,n_fea] = size(data);
XD = zeros(n_fea,n_fea);
XS = zeros(n_fea,n_fea);
for i = 1:n_sam-1
    for j = i+1:n_sam
        one_vec = weight(i) * data(i,:)- weight(j) * data(j,:);
        if S(i,j) == 1
            XS = XS + one_vec' * one_vec;
        end
        if D(i,j) == 1
            XD = XD + one_vec' * one_vec;
        end
    end
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
for i = 1:n_sam
    one = data(i,:);
    for j = 1:n_sam
        two = data(j,:);
        if i==j
            QS_matrix(i,j) = one * M * two';
            QD_matrix(i,j) = one * inv(M) * two';
        end
        if S(i,j) == 1
            QS_matrix(i,j) = one * M * two';
        end
        if D(i,j) == 1
            QD_matrix(i,j) = one * inv(M) * two';
        end
    end
end
QD = QD_matrix.*L_D;
QS = QS_matrix.*L_S;
end
function value = vec(x)
    value = x(:);
end