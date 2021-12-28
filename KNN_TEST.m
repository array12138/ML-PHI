clc;clear;
%1 Read dataset
filename = 'dataset.mat'; % nxd matrix,n: sample num, d: dim num
load(filename); %dataset name: data, label:true label
filename = 'Ionoshpere';
%% Indexing of data
index_data =18;
%%

disp(filename);
data = dataset{index_data,1};
label = dataset{index_data,2};
[n_sam] = length(label);
% 2 Product n_trials trials
n_trials = 20;
disp(['n_trials = ',num2str(n_trials)]);
i_trials = 1;
t_value = [1]; % t value;
mu_value = [1]; % \mu value
k_value = [1];
theta = [0.1,1,10];
parameter.gamma = 1e-1;
% parameter.K = 3;
alltrials_result = zeros(n_trials,1);
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
   
    M = eye(size(train_data,2));
    preds = KNN(train_data,train_label, M, 3, test_data);
    index = find((test_label-preds)==0);
    accuracy = length(index)/size(test_data,1);
    alltrials_result(i_trials,1) = accuracy;
    i_trials = i_trials  + 1; 
end


% alltrials_result
% save([filename,'_all','.mat'],'alltrials_result');
% result = [mean(alltrials_result(:,1)),std(alltrials_result(:,1))]
% save([filename,'_index','.mat'],'result');



