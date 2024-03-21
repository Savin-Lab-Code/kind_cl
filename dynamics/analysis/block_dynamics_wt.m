function [ltom, htom, mtol, mtoh, ltom_incong, htom_incong, sems] =...
    block_dynamics_wt(A, twin, smoothfactor, varargin)
%computes the change in z-scored wait time, normalized to mean wait time 
%   for each reward volume around each block transition. cmc 10/19/21.
%   inputs: A struct
%       twin = time window to analyze [-twin twin]
%       smoothfactor = smoothing factor. cmc has found 7-10 (trials) to 
%       be reasonable for most rats.
%   outputs: a matrix of the average change in wait time for each
%            transition. 
%       ltom = low to mixed
%       htom = high to mixed
%       mtol = mixed to low
%       mtoh = mixed to high
  
if isfield(A, 'pd') %just in case the input is an S struct.
    [A, ~, ~] = parse_data_from_mysql(A);
end
[~, A.reward] = convertreward(A.reward);

A.wait_time(logical(A.catch)) = (A.wait_time(logical(A.catch))-...
    mean(A.wait_time(logical(A.catch)), 'omitnan'))./...
    std(A.wait_time(logical(A.catch)), 'omitnan');

%convert everything into z-scored wait time for that volume.
A = deltawt(A); 

if isempty(varargin)
    dropthese = true(size(A.wait_time));
else
    dropthese = varargin{1};
end

%decide if doing causal smoothing, or standard moving window average
if numel(varargin) > 1
    usecausal = varargin{2};
else
    usecausal = false;
end
A.wait_time(~dropthese) = nan;

[lowtomix, hightomix,...
    mixtolow, mixtohigh,...
    lowtomix_incong, hightomix_incong] =...
    find_block_transition(A);

xvec = -twin:1:twin;
ltom = nan(length(lowtomix), twin*2+1);
htom = nan(length(hightomix), twin*2+1);
mtol = nan(length(mixtolow), twin*2+1);
mtoh = nan(length(mixtohigh), twin*2+1);
ltom_incong = nan(length(lowtomix), twin*2+1);
htom_incong = nan(length(hightomix), twin*2+1);

ctch = find(A.catch==1 & A.vios==0);

for j = 1:length(lowtomix)
    these = (lowtomix(j)-twin):(lowtomix(j)+twin);
    [c, ia] = intersect(these, ctch);
    ltom(j,ia) =  A.wait_time(c);        
end

for j = 1:length(hightomix)
    these = (hightomix(j)-twin):(hightomix(j)+twin);
    [c, ia] = intersect(these, ctch);
    htom(j,ia) =  A.wait_time(c);
end

for j = 1:length(mixtolow)
    these = (mixtolow(j)-twin):(mixtolow(j)+twin);
    [c, ia] = intersect(these, ctch);
    mtol(j,ia) =  A.wait_time(c);
end

for j = 1:length(mixtohigh)
    these = (mixtohigh(j)-twin):(mixtohigh(j)+twin);
    [c, ia] = intersect(these, ctch);
    mtoh(j,ia) =  A.wait_time(c);
end

for j = 1:length(lowtomix_incong)
    these = (lowtomix_incong(j)-twin):(lowtomix_incong(j)+twin);
    [c, ia] = intersect(these, ctch);
    ltom_incong(j,ia) =  A.wait_time(c);
end

for j = 1:length(hightomix_incong)
    these = (hightomix_incong(j)-twin):(hightomix_incong(j)+twin);
    [c, ia] = intersect(these, ctch);
    htom_incong(j,ia) =  A.wait_time(c);
end

sems = nan(6, 2*twin+1);

sems(1,:) = std(ltom, 'omitnan')./sqrt(size(ltom, 1));
ltom = mean(ltom, 1, 'omitnan'); ltom(twin+1) = ltom(twin-1);

sems(2,:) = std(htom, 'omitnan')./sqrt(size(htom, 1));
htom = mean(htom, 1, 'omitnan'); htom(twin+1) = htom(twin-1);

sems(3,:) = std(mtol, 'omitnan')./sqrt(size(mtol, 1));
mtol = mean(mtol, 1, 'omitnan'); mtol(twin+1) = mtol(twin-1);

sems(4,:) = std(mtoh, 'omitnan')./sqrt(size(mtoh, 1));
mtoh = mean(mtoh, 1, 'omitnan'); mtoh(twin+1) = mtoh(twin-1);

sems(5,:) = std(ltom_incong, 'omitnan')./sqrt(size(ltom_incong, 1));
ltom_incong = mean(ltom_incong, 1, 'omitnan');
ltom_incong(twin+1) = ltom_incong(twin-1);

sems(6,:) = std(htom_incong, 'omitnan')./sqrt(size(htom_incong, 1));
htom_incong = mean(htom_incong, 1, 'omitnan');
htom_incong(twin+1) = htom_incong(twin-1);

if usecausal
    ltom = causal_smooth(ltom, smoothfactor);
    htom = causal_smooth(htom, smoothfactor);
    mtol = causal_smooth(mtol, smoothfactor);
    mtoh = causal_smooth(mtoh, smoothfactor);
else
    ltom = smooth(ltom, smoothfactor);
    htom = smooth(htom, smoothfactor);
    mtol = smooth(mtol, smoothfactor);
    mtoh = smooth(mtoh, smoothfactor);
end

% xind = twin+1;
% ltom(xind:end) = smooth(ltom(xind:end), smoothfactor);
% ltom(1:twin) = flipud(smooth(fliplr(ltom(1:twin)), smoothfactor));
% 
% htom(xind:end) = smooth(htom(xind:end), smoothfactor);
% htom(1:twin) = flipud(smooth(fliplr(htom(1:twin)), smoothfactor));
% 
% mtol(xind:end) = smooth(mtol(xind:end), smoothfactor);
% mtol(1:twin) = flipud(smooth(fliplr(mtol(1:twin)), smoothfactor));
% 
% mtoh(xind:end) = smooth(mtoh(xind:end), smoothfactor);
% mtoh(1:twin) = flipud(smooth(fliplr(mtoh(1:twin)), smoothfactor));
% 
% htom_incong(xind:end) =...
%   smooth(htom_incong(xind:end), smoothfactor);
% htom_incong(1:twin) =...
%   flipud(smooth(fliplr(htom_incong(1:twin)), smoothfactor));
% 
% ltom_incong(xind:end) =...
%   smooth(ltom_incong(xind:end), smoothfactor);
% ltom_incong(1:twin) =...
%   flipud(smooth(fliplr(ltom_incong(1:twin)), smoothfactor));
end

function [A] = deltawt(A)

rew = [5 10 20 40 80];
for j = 1:length(rew)
    m = find(A.reward==rew(j) & A.catch==1 & A.vios==0);
    A.wait_time(m) = (A.wait_time(m)-mean(A.wait_time(m), 'omitnan'))./...
        std(A.wait_time(m), 'omitnan');
end

end

function [lowtomix, hightomix,...
    mixtolow, mixtohigh,...
    lowtomix_incong, hightomix_incong] =...
    find_block_transition(A)

ntrials = [0; cumsum(A.ntrials)];

[~, treal, tincong] = blockhmat_withvios(A);

lowtomix = cell(length(ntrials), 1);
hightomix = cell(length(ntrials), 1);
mixtolow = cell(length(ntrials), 1);
mixtohigh = cell(length(ntrials), 1);
lowtomix_incong = cell(length(ntrials), 1);
hightomix_incong = cell(length(ntrials), 1);

for sess = 1:length(ntrials)-1
    lowtomix{sess} = (find(treal(sess,:) == -2) + ntrials(sess))';
    hightomix{sess} = (find(treal(sess,:) == -1) + ntrials(sess))';
    
    mixtolow{sess} = (find(treal(sess,:) == 2) + ntrials(sess))';
    mixtohigh{sess} = (find(treal(sess,:) == 1) + ntrials(sess))';
    
    lowtomix_incong{sess} = (find(tincong(sess,:) == -2) +...
        ntrials(sess))';
    hightomix_incong{sess} = (find(tincong(sess,:) == -1) +...
        ntrials(sess))';
end

lowtomix = cell2mat(lowtomix);
hightomix = cell2mat(hightomix);
mixtolow = cell2mat(mixtolow);
mixtohigh = cell2mat(mixtohigh);
lowtomix_incong = cell2mat(lowtomix_incong);
hightomix_incong = cell2mat(hightomix_incong);

end
