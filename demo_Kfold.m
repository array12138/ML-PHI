clc;clear;
%1 Read dataset
filename = 'dataset.mat'; % nxd matrix,n: sample num, d: dim num
load(filename); %dataset name: data, label:true label
%% Indexing of data
index_data =4;
%%
filename = dataset{index_data,3};
disp(filename);
data = dataset{index_data,1};
label = dataset{index_data,2};
[n_sam] = length(label);
% 2 Product n_trials trials
n_trials = 20;
disp(['n_trials = ',num2str(n_trials)]);
i_trials = 1;
t_value = [0.1:0.2:0.9]; % t value;
mu_value = [1e-2,1e-1,1,1e1,1e2]; % \mu value
k_value = [1,2,3];
parameter.gamma = 1e-1;
% parameter.K = 3;
alltrials_result = zeros(n_trials,4);
while i_trials <= n_trials
    ntrain = floor(n_sam*0.7);
    train_index = randperm(n_sam,ntrain);
    test_index = (1:n_sam)';
    test_index(train_index(:)) = [];
    % 3 Extraction of data

    train_data = data(train_index(:),:);
    [train_data,mu,sigma] = zscore(train_data);
       % Prevent data from appearing exactly the same
    index = find(sigma==0);
    mu(index(:)) = 1;
    sigma(index(:)) = 1;
    train_label = label(train_index(:));
    test_data = data(test_index(:),:);
    for i = 1:size(test_data,1)
        test_data(i,:) = (test_data(i,:) - mu)./sigma;
    end
    test_label = label(test_index(:));
    para_metrix = zeros(length(t_value)*length(mu_value)*length(k_value),4);
    parm_iter = 1;
    %4.1 Perform parameter t selection
    for t_index = 1:length(t_value)
        parameter.t = t_value(t_index);    
        %5.1 Perform the selection of the parameter mu value
        for mu_index = 1:length(mu_value)
            parameter.mu = mu_value(mu_index);
            para_metrix(parm_iter,2) = t_value(t_index);
            para_metrix(parm_iter,3) = mu_value(mu_index);   
            for k = 1: length(k_value)
                parameter.K = k_value(k);
                para_metrix(parm_iter,4) =  k_value(k);
                try
                [ M,weight] = sample_weight_metriclearning(train_data,train_label,parameter);
                preds = KNN(train_data,train_label, M, 3, test_data);
                index = find((test_label-preds)==0);
                accuracy = length(index)/size(test_data,1);
                para_metrix(parm_iter,1)  = accuracy;
                catch
                    para_metrix(parm_iter,1)  = 0;
                end
                parm_iter = parm_iter + 1;
            end%k
        end%beta
     end%mu
     best_index = find(para_metrix(:,1) == max(para_metrix(:,1)));
     alltrials_result(i_trials,1) = para_metrix(best_index(1),1);
     alltrials_result(i_trials,2) = para_metrix(best_index(1),2);
     alltrials_result(i_trials,3) = para_metrix(best_index(1),3);
     alltrials_result(i_trials,4) = para_metrix(best_index(1),4);
     i_trials = i_trials  + 1;
end
save([filename,'_all','.mat'],'alltrials_result');
result = [mean(alltrials_result(:,1)),std(alltrials_result(:,1))];
save([filename,'_index','.mat'],'result');



