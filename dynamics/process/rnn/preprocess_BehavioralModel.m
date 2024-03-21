%convert RNN data to input data for behavioral model and save to server

addpath(genpath('~/projects/dynamics'));
addpath(genpath('~/projects/constantinoplelab/Analysis/BehavioralModel'));

%% specifics about the kfolds and stuff

ptest = 0.0; %proportion of held-out test data. use all data
nfolds = 1; %number of cross-val folds
simtype = 'local';

%% decide which system to use, and what your naming conventions are

switch simtype
    
    case 'greene' %take hpc files and save on hpc
        rawdatadir = '/scratch/dh148/dynamics/results/rnn/reinforce/kindergarten_sup/';
        savedir =  '/scratch/dh148/dynamics/data/BehavioralModel/Data/';
    case 'local' %take local files and save them locally
        rawdatadir = '~/projects/dynamics/results/rnn/reinforce/kindergarten_sup/';
        savedir = '~/projects/dynamics/data/BehavioralModel/Data/'; 
    case 'lab' %take local files and save them on lab server
        rawdatadir = '~/projects/dynamics/results/rnn/reinforce/kindergarten_sup/';
        savedir = '/Volumes/server/BehavioralModel/Data/'; 
end

savedir_crossval = fullfile(savedir,'CrossVal/');
savedir_NFold = fullfile(savedir,'NFold/');

numvec = [2,4,5,6,7,13]; %numbering of your networks

%convert raw file names to behavioral-model-friendly ones
%dataname_fun = @(idx) strcat('rnn_curric_',num2str(idx),'_block_0.mat'); %name of raw files
%ratname_fun = @(idx) strcat('rnnCurric',num2str(idx),'');

dataname_fun = @(idx) strcat('rnn_curric_',num2str(idx),'_block_0_freeze.mat'); %name of raw files
ratname_fun = @(idx) strcat('rnnCurric',num2str(idx),'freeze');


%% load, reformat, put in correct subfolders

ratList = cell(numel(numvec),1);

for j = 1:numel(numvec)
    %load data from equivalent of ratTrial file
    ratList{j} = ratname_fun(numvec(j));
    disp(ratList{j})
    A = load(strcat(rawdatadir,dataname_fun(numvec(j))));
    
    %some extra pre-processing
    A.wait_time(A.vios==1)= nan;    
    %overwrite a bug about loading catch
    A.catch = A.catch_;
    A = rmfield(A,'catch_');
    
    %save to 
    save(strcat(savedir,'ratTrial_',ratList{j},'.mat'),'A')   
end


%% create test and cross-validation data. equiv to generateCrossValidationSets

disp('making cross-validated data sets')

for j = 1:numel(ratList)
    ratname = ratList{j};
    disp(ratname)

    generateRatTrial_CrossVal(ratname, ptest, savedir, savedir_crossval);
    generateRatTrial_NFold(ratname, nfolds, savedir_crossval, savedir_NFold);
    
end


