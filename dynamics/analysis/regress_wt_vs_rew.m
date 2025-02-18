function [b, tau, er] = regress_wt_vs_rew(A, nback, plotarg, cond)
%regress waittime against rewards, a la Wang/Uchida 2013.
%also fits exponential. Adapted from regress_latency_vs_rew
%inputs: A struct. If you give it the S struct, will convert it to A struct.
%nback specifies number of previous trials to regress.
%avm 11/22/21
% avm 23 Nov 21 - edit to only use rewards and wait times from mixed blocks

if nargin == 2
    plotarg = false;
    cond = true(size(A.optout));
elseif nargin == 3
    cond = true(size(A.optout));
end

if isfield(A, 'pd') %if it's an S struct.
    [A, ~, ~] = parse_data_from_mysql(A);
end

% A.block == 1
%z-score wait times - only for catch trials .
usethese = A.optout==1 & A.catch==1 & A.block==1 & cond;

WT = A.wait_time(usethese);
WT = (WT - mean(WT, 'omitnan'))./(std(WT, 'omitnan'));

R = log2(double(A.reward));
R(A.block~=1) = nan;

%Create design matrix. This includes post-violation trials.
x = nan(length(R), nback+1);
for j = 1:nback
    x(:,j) = [zeros(j-1,1); R(1:end-(j-1))];
end
x(:,end) = ones(length(R),1); %constant term.
X = x(usethese,:);

[b, bint] = regress(WT,X); %do regression.
er = bint(:,2)-bint(:,1); %calculate confidence intervals.

%this is just to get the fmincon code to fit better.
% values were really small after z-scoring.
% b = b.*10;

% Find the first non-significant regression coefficient (first one with
% 0 in 95% CI) then zero out all coefficients including and after that

if nback > 0
    i = find(arrayfun(@(ii) bint(ii,1) < 0 & bint(ii,2) > 0, 1:nback+1), 1);
    b(i:end-1) = 0;
    
    [~, bestfit, ~] = fit_exp_decay(1:nback, 10*b(1:nback)', 10,...
        [-10 0], [10 10]); %fit exponential
    f = @(x, bestfit) (bestfit(1)*exp(-x./bestfit(2)));
    newx = 2:.1:nback;
    newy = f(newx, bestfit);
    
    if all(b(2:end-1)==0)
        tau = 0;
    else
        tau = bestfit(2);
    end
else
    tau = 0;
end


if plotarg
    %makeplot.
    figure;
    line([0 nback+1], [0 0], 'Color', [0 0 0]);
    plot(1:nback, b(1:nback), '.k', 'MarkerSize', 10); hold on
    errorbar(1:nback, b(1:nback), er(1:nback),...
        'Color', [0 0 0], 'LineStyle', 'none');
    plot(newx, newy, 'r');
    set(gca, 'TickDir', 'out'); box off;
    xlabel('N trials back');
    ylabel('Negative Regression Coefficient');
    title(strcat(['WT Tau = ', num2str(tau)]));
end
end

