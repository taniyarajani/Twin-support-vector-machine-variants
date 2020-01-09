clear all;
clc;
%classifier_name={'TBSVM','Sparse_Pin_SVM','IPTWSVM'};
classifier_name={'TBSVM','Sparse_Pin_SVM','Improved_Sparse_Pin_TSVM'}; % Here variable Sparse_Pin_SVM is actually Sparse_Pin_TSVM, only for coding convenience.
data_path='C:\Users\iiiti513\Dropbox\Codes\TWSVM\TWSVM\Code\Proposed_SPTSVM_trial_only_C\Mudasir_Nonlinear_ISPTSVM\Taniya_Code_sent\small_scale\'; %put slash at last THIS IS DATASET PATH
% addpath(genpath(cd));
Folder_info=dir(data_path); % List The Datasets
%rng('default');
kern.type='rbf';

%Paths for Saving the Individual and Net Results
path1='C:\Users\iiiti513\Dropbox\Codes\TWSVM\TWSVM\Code\Proposed_SPTSVM_trial_only_C\Mudasir_Nonlinear_ISPTSVM\Taniya_Code_sent\Results\';  % THIS IS WHERE RESULTS ARE SAVED
addpath(genpath('C:\Users\iiiti513\Dropbox\Codes\TWSVM\TWSVM\Code\Proposed_SPTSVM_trial_only_C\Mudasir_Nonlinear_ISPTSVM\Taniya_Code_sent'));
% temp_net_file_name=classifier_name(1);
% for i=2:length(classifier_name)
%     temp_net_file_name=strcat(temp_net_file_name,'_',classifier_name(i));
% end
temp_net_file_name='Taniya_code';
path1=strcat(path1,temp_net_file_name,'\');

net_result_file=char(strcat(path1,kern.type,'_kernel\Net_Result\'));
if ~exist(net_result_file, 'dir')
    mkdir(net_result_file);
end
%addpath(net_result_file);
f_name_all_Net = strcat(net_result_file,'Net_Result.txt');
file1=fopen(f_name_all_Net,'a+');
fclose(file1);

%Saving Individual Result
Individual_result_file=char(strcat(path1,kern.type,'_kernel\Individual_Result\'));
if ~exist(Individual_result_file,'dir')
    mkdir(Individual_result_file);
end
%addpath(Individual_result_file);

for i=1:length(classifier_name)
    classifier=char(classifier_name{i});
    f_name_all = strcat(Individual_result_file,classifier,'.txt');
    file4=fopen(f_name_all,'a+');
    fclose(file4);
end

%Eleiminating data_done
%    table_data=readtable('E:\Taniya_Results\Results_Non_Linear_New\Taniya_code\rbf_kernel\Net_Result\Net_Result.txt');
%   t=extractfield(Folder_info,'name');
%    t=t';
%   data2=table_data.Var1;
%  zz=setdiff(t,data2);
zz_temp=importdata('C:\Users\iiiti513\Dropbox\Codes\TWSVM\TWSVM\Code\Proposed_SPTSVM_trial_only_C\Mudasir_Nonlinear_ISPTSVM\Taniya_Code_sent\datasets.txt');
for ii=1:7
    % for ii=38:38
    %Dataset_name=Folder_info(ii).name
    Dataset_name=char(zz_temp(ii))
    % Reading Dataset
    read_path=strcat(data_path,Dataset_name);
    
    Dataset_path=strcat(read_path,'.txt');
    %     if ~exist(Dataset_path,'file')
    %         continue;
    %     end
    if exist(Dataset_path,'file')
        Data_struct=importdata(Dataset_path);
        Data=Data_struct;  %Data Matrix with first column as in index and final as the class label
        % Separate Labels and Data
        dataX=Data(:,1:end-1);
        dataY=Data(:,end);
        labels=zeros(length(dataY),1);
        unq_labels=unique(dataY);
        %if multiclass data skip
        if length(unq_labels)>2
            continue;
        end
        
        % assign +1 and -1 labels
        labels(dataY==unq_labels(1))=1;
        labels(dataY~=unq_labels(1))=-1;
        
        dataY=labels;
        %     if min(dataY)==0
        %         dataY=dataY+1;
        %     end
        % Divide the Data into Training and Testing Data
        %dataA = cumsum(ones(20,3));  % some test data
        p = .7 ;     % proportion of rows to select for training
        N = size(dataX,1);  % total number of rows
        tf = false(N,1);    % create logical index vector
        tf(1:round(p*N)) = true;
        tf = tf(randperm(N)) ;  % randomise order
        dataTrainingX = dataX(tf,:); % Data
        dataTrainingY = dataY(tf,:);% Labels
        dataTestingX = dataX(~tf,:);% Data
        dataTestingY = dataY(~tf,:);% Labels
    else
        train_Dataset_path=strcat(read_path,Dataset_name,'_train_R.dat');
        test_Dataset_path=strcat(read_path,Dataset_name,'_test_R.dat');
        if ~exist(train_Dataset_path,'file')
            continue;
        end
        train_Data_struct=importdata(train_Dataset_path);
        train_Data=train_Data_struct;  %Data Matrix with first column as in index and final as the class label
        test_Data_struct=importdata(test_Dataset_path);
        test_Data=test_Data_struct;  %Data Matrix with first column as in index and final as the class label
        dataTrainingX=train_Data(:,1:end-1);
        dataTrainingY=train_Data(:,end);
        dataTestingX=test_Data(:,1:end-1);
        dataTestingY=test_Data(:,end);
        
        labels=zeros(length(dataTrainingY),1);
        unq_labels=unique(dataTrainingY);
        % if multiclass data skip
        if length(unq_labels)>2
            continue;
        end
        % assign +1 and -1 labels
        labels(dataTrainingY==unq_labels(1))=1;
        labels(dataTrainingY~=unq_labels(1))=-1;
        dataTrainingY=labels;
        
        test_labels=zeros(length(dataTestingY),1);
        % assign +1 and -1 labels
        test_labels(dataTestingY==unq_labels(1))=1;
        test_labels(dataTestingY~=unq_labels(1))=-1;
        dataTestingY=test_labels;
    end
    % take the lables also
    [mm,~]=size(dataTrainingX);
%     if mm>600
%     continue;
%     end
    TBSVM_acc=[];
    Sparse_Pin_SVM_acc=[];
    Improved_Sparse_Pin_TSVM_acc=[];
    % All Parameters
    epsilon = [0; 0.05; 0.1];			%Here epsilon = epsilon1 (for subproblem 1) = epsilon2 (for subproblem 2)
    %epsilon = [0; 0.05];
    tau = [0.01; 0.1; 0.2; 0.5; 1.0];					%Here tau = tau1 (for subproblem 1) = tau2 (for subproblem 2)
    %tau =[0.5];
    cc1=[10^0,10^-5,10^-4,10^-3,10^-2,10^-1,10^1,10^2,10^3,10^4,10^5]; % 15 values
    %cc1=[10^-7,10^-6];
    %c1 = power(10, -7);									%Here c1 (for subproblem 1) = c2 (for subproblem 2)
    cc3=[10^0,10^-5,10^-4,10^-3,10^-2,10^-1,10^1,10^2,10^3,10^4,10^5]; % 15 values
    %cc3=[10^-7,10^-6];
    %c3 = power(10, -7);									%Here c3 (for subproblem 1) = c4 (for subproblem 2)
    %gamma = power(10, -8);
    gamma_value=[10^0,10^-5,10^-4,10^-3,10^-2,10^-1,10^1,10^2,10^3,10^4,10^5]; % 15 values
    %gamma_value=[10^0,10^-7];
    p=10; % 10 fold cross validation
    % Declare Variables
    % classifier_name={'TBSVM','Sparse_Pin_SVM','ISPTWSVM'};
    for i=1:length(classifier_name)
        var_name=char(strcat('overall_',classifier_name(i),'=[]')); % creating the variable names for storing the overall matrix
        eval(var_name);
    end
    % All Functions
    
    for k = 1:length(cc1) 								%Iterate on 13 values of c1
        %c1 = c1*10;
        for q = 1: length(cc3)      %Iterate on 13 values of c3
            %c3 = c3*10;
            for i = 1: length(epsilon)
                for j = 1: length(tau)
                    for gg=1:length(gamma_value)
                        gamma=gamma_value(gg);
                        fprintf('running till here 1')
                        %10-fold cross validation
                        
                        FunPara=struct('f_name','','c1',cc1(k),'c3',cc3(q),'eplsion','','kerfPara',struct('type',kern.type,'pars',gamma));
                        fprintf('running till here 2')
                        cvFolds = crossvalind('Kfold', size(dataTrainingX,1), p);   %# get indices of 10-fold CV of "groups" observation
                        %cp = classperf(groups);                      %# init performance tracker
                        %classifier_name={'TBSVM','Sparse_Pin_SVM','ISPTWSVM'};
                        onef_acc_TBSVM=[];
                        onef_acc_Sparse_Pin_SVM=[];
                        onef_acc_Improved_Sparse_Pin_TSVM=[];
                        
                        onef_train_time_TBSVM=[];
                        onef_train_time_Sparse_Pin_SVM=[];
                        onef_train_time_Improved_Sparse_Pin_TSVM=[];
                        for va = 1:p                                  %# for each fold
                            testIdx = (cvFolds == va);                %# get indices of test instances
                            trainIdx = ~testIdx;
                            % Training Set
                            trainX=dataTrainingX(trainIdx,:);
                            trainY=dataTrainingY(trainIdx,:);
                            % Testing Set
                            testX=dataTrainingX(testIdx,:);
                            testY=dataTrainingY(testIdx,:);
                            % Separate the The two class in training data
                            X1_Train=trainX(trainY==1,:); X2_Train=trainX(trainY~=1,:);  X_Test=testX; Y_Test=testY;
                            Y_Train=[trainY(trainY==1,:);trainY(trainY~=1,:)]; % Y_Train=[Labels_of_A;Labels_of_B]
                            
                            % TBSVM
                            if i==1&&j==1
                              [accuracy0,train_tym] = TWSVM(X_Test, Y_Test,X1_Train, X2_Train,FunPara);
                              onef_acc_TBSVM=[onef_acc_TBSVM,accuracy0];
                            end
                            
                            %[accuracy1, non_zero_dual_variables, training_time, lambda] = Sparse_Pin_SVM(X1_Train, X2_Train, Y_Train, X_Test, Y_Test, cc1(k), epsilon(i), tau(j));
                            %[accuracy2, non_zero_dual_variables, training_time, lambda] = Sparse_Pin_SVM_Kernel(X1_Train, X2_Train, Y_Train, X_Test, Y_Test, cc1(k), gamma, epsilon(i), tau(j));
                            if q==1
                                [accuracy2, non_zero_dual_variables, training_time, lambda] = Sparse_Pin_TSVM_Kernel(X1_Train, X2_Train, X_Test, Y_Test, cc1(k), gamma, epsilon(i), tau(j));
                                onef_acc_Sparse_Pin_SVM=[onef_acc_Sparse_Pin_SVM,accuracy2];
                            end
                            %Improved ISPTWSVM
                            %[accuracy3, non_zero_dual_variables, training_time, lambda] = IPTWSVM(X1_Train, X2_Train, X_Test, Y_Test, cc1(k),cc3(q), tau(j));
                            
                            %[accuracy3, non_zero_dual_variables, training_time, lambda] = IPTWSVM_Kernel(X1_Train, X2_Train, X_Test, Y_Test, cc1(k),cc3(q),gamma,tau(j));
                            [accuracy3, non_zero_dual_variables, training_time, lambda] = Improved_Sparse_Pin_TSVM_Kernel(X1_Train, X2_Train, X_Test, Y_Test, cc1(k), gamma, epsilon(i), tau(j),cc3(q));
                            onef_acc_Improved_Sparse_Pin_TSVM=[onef_acc_Improved_Sparse_Pin_TSVM,accuracy3];
                            
                        end% 10 fold validation
                        %classifier_name={'TBSVM','Sparse_Pin_SVM','ISPTWSVM'};
                        %temp1=[mean(onef_acc_TBSVM),cc1(k),cc3(q),gamma];
                        %overall_TBSVM=[overall_TBSVM;temp1];
                        temp2=[mean(onef_acc_Sparse_Pin_SVM),cc1(k),epsilon(i),tau(j),gamma];
                        overall_Sparse_Pin_SVM=[overall_Sparse_Pin_SVM;temp2];
                        temp3=[mean(onef_acc_Improved_Sparse_Pin_TSVM),cc1(k),cc3(q),epsilon(i),tau(j),gamma];
                        overall_Improved_Sparse_Pin_TSVM=[overall_Improved_Sparse_Pin_TSVM;temp3];
                    end % Gamma_Value
                end
            end
        end
    end %for K i.e. C1
    
    %     overall_TBSVM=cell2mat(overall_TBSVM);
    %     overall_Sparse_Pin_SVM=cell2mat(overall_Sparse_Pin_SVM);
    %     overall_ISPTWSVM=cell2mat(overall_ISPTWSVM);
    % TBSVM
         non_nan_indx = not(isnan(overall_TBSVM(:,1)));
         overall_TBSVM=overall_TBSVM(non_nan_indx,:);
         f_handle_file_TBSVM=strcat(Individual_result_file,'TBSVM','.txt');
         zz=save_to_file(f_handle_file_TBSVM,Dataset_name,overall_TBSVM);
    % Sparse_Pin_SVM
    non_nan_indx = not(isnan(overall_Sparse_Pin_SVM(:,1)));
    overall_Sparse_Pin_SVM=overall_Sparse_Pin_SVM(non_nan_indx,:);
    f_handle_file_Sparse_Pin_SVM=strcat(Individual_result_file,'Sparse_Pin_TSVM','.txt');
    zz=save_to_file(f_handle_file_Sparse_Pin_SVM,Dataset_name,overall_Sparse_Pin_SVM);
    %
    non_nan_indx = not(isnan(overall_Improved_Sparse_Pin_TSVM(:,1)));
    overall_Improved_Sparse_Pin_TSVM=overall_Improved_Sparse_Pin_TSVM(non_nan_indx,:);
    f_handle_file_ISPTWSVM=strcat(Individual_result_file,'Improved_Sparse_Pin_TSVM','.txt');
    zz=save_to_file(f_handle_file_ISPTWSVM,Dataset_name,overall_Improved_Sparse_Pin_TSVM);
    % Run here for optimal parameters
    % Separate the The two class in training data
    trainX=dataTrainingX;
    trainY=dataTrainingY;
    testX=dataTestingX;
    testY=dataTestingY;
    
    X1_Train=trainX(trainY==1,:); X2_Train=trainX(trainY~=1,:);  X_Test=testX; Y_Test=testY;
    Y_Train=[trainY(trainY==1,:);trainY(trainY~=1,:)]; % Y_Train=[Labels_of_A;Labels_of_B]
    
    %      TBSVM
         DataTrain.A=X1_Train;
         DataTrain.B=X2_Train;
         [max_TBSVM,indxTBSVM]=max(overall_TBSVM(:,1));
         TBSVM_param=overall_TBSVM(indxTBSVM,:);
         FunPara1.c1=TBSVM_param(2);
         FunPara1.c3=TBSVM_param(3);
         FunPara1.kerfPara.type=kern.type;
         FunPara1.kerfPara.pars=TBSVM_param(end);
         [accuracy_TBSVM,training_time_TBSVM] = TWSVM(X_Test, Y_Test,X1_Train, X2_Train,FunPara1);
    
    % Sparse_Pin_SVM
    [max_Sparse_Pin_SVM,indxSparse_Pin_SVM]=max(overall_Sparse_Pin_SVM(:,1));
    Sparse_Pin_SVM_param=overall_Sparse_Pin_SVM(indxSparse_Pin_SVM,:);
    Sparse_Pin_SVM_c1=Sparse_Pin_SVM_param(2); Sparse_Pin_SVM_epsilon=Sparse_Pin_SVM_param(3); Sparse_Pin_SVM_tau=Sparse_Pin_SVM_param(4);Sparse_Pin_SVM_gamma=Sparse_Pin_SVM_param(end);
    %[accuracy_Sparse_Pin_SVM, non_zero_dual_variables_Sparse_Pin_SVM, training_time_Sparse_Pin_SVM, lambda] = Sparse_Pin_SVM(X1_Train, X2_Train, Y_Train, X_Test, Y_Test, Sparse_Pin_SVM_c1, Sparse_Pin_SVM_epsilon, Sparse_Pin_SVM_tau);
    %[accuracy_Sparse_Pin_SVM, non_zero_dual_variables_Sparse_Pin_SVM, training_time_Sparse_Pin_SVM, lambda] = Sparse_Pin_SVM_Kernel(X1_Train, X2_Train, Y_Train, X_Test, Y_Test, Sparse_Pin_SVM_c1, Sparse_Pin_SVM_gamma, Sparse_Pin_SVM_epsilon, Sparse_Pin_SVM_tau);
    [accuracy_Sparse_Pin_SVM, non_zero_dual_variables_Sparse_Pin_SVM, training_time_Sparse_Pin_SVM, lambda] =Sparse_Pin_TSVM_Kernel(X1_Train, X2_Train, X_Test, Y_Test, Sparse_Pin_SVM_c1, Sparse_Pin_SVM_gamma, Sparse_Pin_SVM_epsilon, Sparse_Pin_SVM_tau);
    % Improved_Sparse_Pin_TSVM
    [max_Improved_Sparse_Pin_TSVM,indxImproved_Sparse_Pin_TSVM]=max(overall_Improved_Sparse_Pin_TSVM(:,1));
    Improved_Sparse_Pin_TSVM_param=overall_Improved_Sparse_Pin_TSVM(indxImproved_Sparse_Pin_TSVM,:);
    Improved_Sparse_Pin_TSVM_c1=Improved_Sparse_Pin_TSVM_param(2); Improved_Sparse_Pin_TSVM_c3=Improved_Sparse_Pin_TSVM_param(3);Improved_Sparse_Pin_TSVM_epsilon=Improved_Sparse_Pin_TSVM_param(4); Improved_Sparse_Pin_TSVM_gamma=Improved_Sparse_Pin_TSVM_param(end); Improved_Sparse_Pin_TSVM_tau=Improved_Sparse_Pin_TSVM_param(5);
    %[accuracy_ISPTWSVM, non_zero_dual_variables_ISPTWSVM, training_time_ISPTWSVM, lambda] = IPTWSVM(X1_Train, X2_Train, X_Test, Y_Test, ISPTWSVM_c1, ISPTWSVM_c3, ISPTWSVM_epsilon, ISPTWSVM_tau);
    %[accuracy_Improved_Sparse_Pin_TSVM, non_zero_dual_variables_Improved_Sparse_Pin_TSVM, training_time_Improved_Sparse_Pin_TSVM, lambda2] = IPTWSVM_Kernel(X1_Train, X2_Train, X_Test, Y_Test, ISPTWSVM_c1, ISPTWSVM_c3, ISPTWSVM_gamma, ISPTWSVM_tau);
    [accuracy_Improved_Sparse_Pin_TSVM, non_zero_dual_variables_Improved_Sparse_Pin_TSVM, training_time_Improved_Sparse_Pin_TSVM, lambda2] = Improved_Sparse_Pin_TSVM_Kernel(X1_Train, X2_Train, X_Test, Y_Test, Improved_Sparse_Pin_TSVM_c1, Improved_Sparse_Pin_TSVM_gamma, Improved_Sparse_Pin_TSVM_epsilon, Improved_Sparse_Pin_TSVM_tau,Improved_Sparse_Pin_TSVM_c3);
    % Net Result File
    max_all=[];
    for i=1:length(classifier_name)
        c_name=char(strcat('accuracy_',classifier_name(i)));
        max_all=[max_all,eval(c_name)];
    end
    zz=save_to_file(f_name_all_Net,Dataset_name,max_all);
    % Net Result File with Parameters
    indiv_path=strcat(net_result_file,'Net_Result_parameters\');
    if ~exist(indiv_path,'dir')
        mkdir(indiv_path);
    end
         f_name_all_Net_parameters = strcat(indiv_path,'TBSVM.txt');
         zz=save_to_file(f_name_all_Net_parameters,Dataset_name,[accuracy_TBSVM,overall_TBSVM(indxTBSVM,2:end),training_time_TBSVM]);
    f_name_all_Net_parameters = strcat(indiv_path,'Sparse_Pin_TSVM.txt');
    zz=save_to_file(f_name_all_Net_parameters,Dataset_name,[accuracy_Sparse_Pin_SVM,overall_Sparse_Pin_SVM(indxSparse_Pin_SVM,2:end),training_time_Sparse_Pin_SVM]);
    f_name_all_Net_parameters = strcat(indiv_path,'Improved_Sparse_Pin_TSVM.txt');
    zz=save_to_file(f_name_all_Net_parameters,Dataset_name,[accuracy_Improved_Sparse_Pin_TSVM,overall_Improved_Sparse_Pin_TSVM(indxImproved_Sparse_Pin_TSVM,2:end),training_time_Improved_Sparse_Pin_TSVM]);
    
end % For all Datasets