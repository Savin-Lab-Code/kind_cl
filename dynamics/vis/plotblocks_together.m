function plotblocks_together(A,name, num, use_eb, varargin)
   %updated form to plot on same plot 
  %plot wait times against reward volumes for mixed, high, and low blocks.
  %input: A struct. If you give it the S struct, will convert it to A struct.
    %calls blocks.m
    %cmc 10/19/21.
    
    if isfield(A, 'pd') %if it's an S struct.
       [A, ~, ~] = parse_data_from_mysql(A);
    end
    
    if isempty(varargin)
         [hi, lo, mix] = blocks(A);
    else
         [hi, lo, mix] = blocks(A, varargin);
    end
    
    %because people like to see the p-value for their data
    x1 = find(A.reward==20 & A.catch==1 & A.block==2);
    x2 = find(A.reward==20 & A.catch==1 & A.block==3);
    if numel(x1) > 1 & numel(x2) > 1
        [pval, ~] = ranksum(A.wait_time(x1), A.wait_time(x2));
    else
        pval = nan;
    end
    

figure(num);
x = [1:5];
if use_eb
    shadedErrorBar(x, mix.wt, mix.er, 'lineprops', '-k'); hold on
else
    plot(x,mix.wt,'k','linewidth',2); hold on
    scatter(x,mix.wt,80,'k','linewidth',2)
end
set(gca, 'TickDir', 'out'); box off;
xlim([0 6]);
ylabel('Mean Wait Time (s)');
set(gca, 'xTick', x);
set(gca, 'XTickLabels', {'5'; '10'; '20'; '40'; '80'});
    
if use_eb
    shadedErrorBar(x, hi.wt, hi.er, 'lineprops', '-r'); hold on
    shadedErrorBar(x, lo.wt, lo.er, 'lineprops', '-b');
else
    plot(x,hi.wt,'r','linewidth',2);hold on
    plot(x,lo.wt,'b','linewidth',2);
    scatter(x,hi.wt,80,'r','linewidth',2);
    scatter(x,lo.wt,80,'b','linewidth',2);
    
end
set(gca, 'xTick', x);
set(gca, 'XTickLabels', {'5'; '10'; '20'; '40'; '80'});
title(strcat([name,', p = ', num2str(pval)]));
xlim([0 6]);