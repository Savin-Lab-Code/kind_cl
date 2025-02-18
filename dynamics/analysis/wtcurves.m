function [hi, lo, mix, p] = wtcurves(A, varargin)
%block 1 is test block.
%block 2 = high adaptation block.
%block 3 = low adaptation block.

A.reward = convertreward(A.reward);
rvec = 1:5;

hi.wt = nan(1, length(rvec));
hi.er = nan(1, length(rvec));

lo.wt = nan(1, length(rvec));
lo.er = nan(1, length(rvec));

mix.wt = nan(1, length(rvec));
mix.er = nan(1, length(rvec));

% A.wait_time(A.wait_time>=A.wt_thresh) = nan;
if ~isempty(varargin)
    usethese =  A.optout & A.catch & varargin{1};
else
    usethese = A.optout & A.catch;
end

wt = A.wait_time(usethese);

rew = A.reward(usethese);
blk = A.block(usethese);
optout = A.optout(usethese);
iscatch = A.catch(usethese);

p = ranksum(wt(rew==3 & blk==2), wt(rew==3 & blk==3),...
    tail='left');

for j = 1:length(rvec)
    
    ix_mi = find(rew==rvec(j) & blk==1 & optout==1 & iscatch==1);
    mix.wt(1,j) = mean(wt(ix_mi), 'omitnan');
    mix.er(1,j) = std(wt(ix_mi), 'omitnan')./sqrt(length(ix_mi));
    
    ix_hi = find(rew==rvec(j) & blk==2 & optout==1 & iscatch==1);
    hi.wt(1,j) = mean(wt(ix_hi), 'omitnan');
    hi.er(1,j) = std(wt(ix_hi), 'omitnan')./sqrt(length(ix_hi));
    
    ix_lo = find(rew==rvec(j) & blk==3 & optout==1 & iscatch==1);
    lo.wt(1,j) = mean(wt(ix_lo), 'omitnan');
    lo.er(1,j) = std(wt(ix_lo), 'omitnan')./sqrt(length(ix_lo));
    
end
