function [clow, chigh] = confidence_bounds(X,ci,bins,twosided)
 %finds two-sided confidence bounds
 % 
 
 X(isinf(X)) = nan;

 %a fully nan data point
 if numel(X(~isnan(X))) == 0
     clow = 0;
     chigh = 0;

 else

    [P,cvec] = histcounts(X(~isnan(X)), bins, 'Normalization','cdf');



    if twosided
        sig_low_idx = find(P > (1-ci)/2);
        if ~isempty(sig_low_idx)
            clow = cvec(sig_low_idx(1)); %first instance
        end

        sig_high_idx = find(P > 1-(1-ci)/2);
        if ~isempty(sig_high_idx)
            chigh = cvec(sig_high_idx(1)); %first instance
        end
    else
        sig_high_idx = find(P > ci);
        if ~isempty(sig_high_idx)
            chigh = cvec(sig_high_idx(1)); %first instance
        end
        clow = nan;
    end
    
 end



 
 