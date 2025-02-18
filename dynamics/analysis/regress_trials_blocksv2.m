function [b_wait, b_iti, dwt, diti, b_iti_rew] =...
    regress_trials_blocksv2(A, tback)

%%Convert blocks into interpretable regressors
A.block(A.block==3) = 0.5;
A.block(A.block==2) = 3;
A.block(A.block==1) = 2;
A.block(A.block==0.5) = 1;
A.wait_time(A.optout==0) = nan;

%convert rewards to ordinal (log) scale
[A.reward, ~] = convertreward(A.reward);

x = A.reward;%.*A.hits;

%initialize design matrix
X = nan(length(x), tback+2); %current trial, previous trials, block

%add reward on current trial to design matrix
X(:,1) = x;

%populate previous rewards in design matrix
for k = 2:tback+1
    X(:,k) = [zeros(k-1,1); x(1:end-k+1)];
end

%add blocks to design  matrix
X(:,tback+2) = A.block;

%z-score wait times and ITIs to get rid of constant term.
Y = (A.wait_time-mean(A.wait_time, 'omitnan'))./...
    std(A.wait_time, 'omitnan');
Y2 = (A.ITI-mean(A.ITI, 'omitnan'))./std(A.ITI, 'omitnan');

intr = (X(:,1)-mean(X(:,1), 'omitnan')).*(X(:,3)-mean(X(:,3), 'omitnan'));

%regression
[b_wait, ~] = regress(Y,[X(:,1), X(:,3)]);
b_wait = [b_wait(1); nan; b_wait(2)];
[b_iti, ~] = regress(Y2,X(:,2)); %omit current reward for itis.
b_iti = [nan; b_iti; nan];

[b_iti_rew, ~] = regress(Y2,X(:,1));

low = mean(A.wait_time(find(A.block==1 & A.reward==3)), 'omitnan');
high = mean(A.wait_time(find(A.block==3 & A.reward==3)), 'omitnan');

dwt = high/low;

low = mean(A.ITI(find(A.block==1)), 'omitnan');
high = mean(A.ITI(find(A.block==3)), 'omitnan');

diti = high/low;