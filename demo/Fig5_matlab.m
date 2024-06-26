% plotting the flow fields. supp figs in different file

addpath(genpath('~/projects/dynamics/'))

dytype = 'full_cl'; % for figure 5
eps = 0.01;


isgen = false;

num = 33;
reg_idx = 0;
idx = 10;
epoch = 'wait';
stage = 'block';
blocktype = 'mixed';
pref = '';

iv = [1,1,1];
res = [2,2,2];

%samples to use? need to dial in
usevec = [2,2,3];  % for ofc

fnum = 17;
zuse_idx = 1;


[f,g, beh, pi_decoder, ~, ~] = call(dytype, num,reg_idx,idx,epoch,stage, blocktype, isgen, pref,eps);
bgtype = 'block';

[h] = genflowfield(fnum, f,g,reg_idx, res, bgtype, zuse_idx);

figure(fnum)
xlim([-760.0000 , 570.0109]);
ylim([-858.2223,  465.4004]);


%% add the KE points

mask = g.labels > -1;
scatter(g.crds(mask,1), g.crds(mask,2),20, 'markeredgecolor','k')

figure(163)
clf
hold on
labs = sort(unique(g.labels));
for j = 1:numel(labs)
    if labs(j) > -1
        scatter(g.crds(g.labels==labs(j),1), g.crds(g.labels==labs(j),2),20)
    end
end

%% do a clustering????

%for now just pick a few points and calculate eigenvalues, plot
%eigenvectors
for k = 1:2
    switch k
        case 1
            figure(163)
        case 2
            figure(fnum)
    end
    pt_list = [1,257];

    for j = 1:numel(labs)
        if labs(j) ==-1
            continue
        end

        t = find(g.labels==labs(j));
        pt_idx = t(1);

        [efuns,evals] = schur(squeeze(g.jac_pc(pt_idx,:,:)));
        sc = 20;

        %quiver(g.crds(pt_idx,1), g.crds(pt_idx,2), sc*real(efuns(1,1)), sc*real(efuns(2,1)),'autoscale','off','color','k', 'linewidth',3)
        %quiver(g.crds(pt_idx,1), g.crds(pt_idx,2), sc*real(efuns(1,2)), sc*real(efuns(2,2)),'autoscale','off','color','r', 'linewidth',3)
        plot([g.crds(pt_idx,1),g.crds(pt_idx,1)+sc*real(efuns(1,1))],[g.crds(pt_idx,2),g.crds(pt_idx,2)+sc*real(efuns(2,1))],'linewidth',1,'color','k')
        plot([g.crds(pt_idx,1),g.crds(pt_idx,1)+sc*real(efuns(1,2))],[g.crds(pt_idx,2),g.crds(pt_idx,2)+sc*real(efuns(2,2))],'linewidth',1,'color','r')

        text(g.crds(pt_idx,1)+1.2*sc*real(efuns(1,1)), g.crds(pt_idx,2)+1.2*sc*real(efuns(2,1)),(num2str(evals(1,1))), 'color','k')
        text(g.crds(pt_idx,1)+1.2*sc*real(efuns(1,2)), g.crds(pt_idx,2)+1.2*sc*real(efuns(2,2)),(num2str(evals(2,2))), 'color','r')
        if g.isfixed(pt_idx)
            text(g.crds(pt_idx,1)-0.2*sc*real(efuns(1,2)), g.crds(pt_idx,2)-0.2*sc*real(efuns(2,2)),'fp', 'color','k','fontsize',15)
        else
            text(g.crds(pt_idx,1)-0.2*sc*real(efuns(1,2)), g.crds(pt_idx,2)-0.2*sc*real(efuns(2,2)),'sp', 'color','k','fontsize',15)
            disp(g.statediff(pt_idx))
        end


    end
end

%% add blocks

figure(fnum)
colors = {'blue','black','red'};
switchvec = [3,1,2]; %f.blockp is orderd in mixed, high, low. this switches it

for j = 1:3
    disp(j)
    switch j
        case 1
            duse = f.samps_low;
        case 2
            duse = f.samps_mixed;
        case 3
            duse = f.samps_high;
    end
    
    
    % get best sample
    ns = numel(duse);
    logpvals = nan(ns,3);
    zuse_m = zuse_idx;
    for k = 1:ns
        if size(duse{k}) > 1
             pc0 = duse{k}(2,1:3);
             [~,idx1] = min(abs(f.X0-pc0(1)));
             [~,idx2] = min(abs(f.Y0-pc0(2)));
             [~,idx3] = min(abs(f.Z0-pc0(3)));
             logpvals(k,:) = f.blockp(idx1,idx2,idx3,:);
        end
     end
     pvals = exp(logpvals)./sum(exp(logpvals),2);
     pvals(isnan(pvals)) = 0;

     [~,idx_use] = sort(pvals(:,switchvec(j)),'descend');

     
     nuse = 3;
     for m = 1:nuse
         duse_m = duse{idx_use(m)};  % choose ones and identify their spot
         disp(strcat('sample for ',num2str(m)))
         disp(size(duse_m))
        
        plot3(duse_m(2:end,1), duse_m(2:end,2),duse_m(2:end,3),'linewidth',2, 'color',colors{j})
        scatter3(duse_m(2,1), duse_m(2,2), duse_m(2,3),40, 'o',  'markerfacecolor',colors{j},'markeredgecolor',colors{j})
     end

end


%% do reg 1
addpath(genpath('~/projects/dynamics/'))
num = 23;
bgtype = 'pi';
reg_idx = 1;
isgen = false;

blocktype = 'mixed';
[fm,g, ~, ~] = call(dytype, num,reg_idx,idx,epoch,stage, blocktype, isgen);

res = [1,1,1];

fnum = 186;
[hm] = genflowfield(fnum, fm,g,reg_idx, res, bgtype, zuse_idx);
figure(186)
title('mixed')

%get the y dim for str
test = cellfun(@(x) mean(x(:,2),1),fm.samps_mixed);
ymean= mean(test);
[~,yuse_idx_m] = min(abs(ymean-fm.Y0));

logKEM = log10(0.5*abs(fm.sgrad_x(:,yuse_idx_m,zuse_idx)).^2);
pm = fm.pi(:,yuse_idx_m);


blocktype = 'high';
[fh,g, ~, ~] = call(dytype, num,reg_idx,idx,epoch,stage, blocktype, isgen);
fnum = 187;
[hh] = genflowfield(fnum, fh,g,reg_idx, res, bgtype, zuse_idx);
figure(187)
title('high')

%get the y dim for str
test = cellfun(@(x) mean(x(:,2),1),fh.samps_mixed);
ymean= mean(test);
[~,yuse_idx_h] = min(abs(ymean-fh.Y0));

logKEH = log10(0.5*abs(fh.sgrad_x(:,yuse_idx_m,zuse_idx)).^2);
ph = fh.pi(:,yuse_idx_h); 


blocktype = 'low';
[fl,g, ~, usepi] = call(dytype, num,reg_idx,idx,epoch,stage, blocktype, isgen);
fnum = 188;
[hl] = genflowfield(fnum, fl,g,reg_idx, res, bgtype, zuse_idx);
figure(188)
title('low')

%get the y dim for str
test = cellfun(@(x) mean(x(:,2),1),fl.samps_mixed);
ymean= mean(test);
[~,yuse_idx_l] = min(abs(ymean-fl.Y0));

logKEL = log10(0.5*abs(fl.sgrad_x(:,yuse_idx_l,zuse_idx)).^2);
pl = fl.pi(:,yuse_idx_l);
%%
prange = [0.97, 0.992];
pinterp = linspace(log(prange(1)), log(prange(2)), 100);
logKEL_interp = interp1(log(pl),smooth(logKEL),pinterp,'spline');
logKEH_interp = interp1(log(ph),smooth(logKEH),pinterp,'spline');
logKEM_interp = interp1(log(pm),smooth(logKEM),pinterp,'spline');

figure(237)
clf
hold on
%plot(log(pl), smooth(logKEL))
plot(pinterp, logKEL_interp,'color','blue','linewidth',2)
plot(pinterp, logKEH_interp,'color','red','linewidth',2)
plot(pinterp, logKEM_interp,'color','black','linewidth',2)
xlim([-0.0260 ,  -0.0054])


[~,xl] = min(smooth(logKEL));
[~,xh] = min(smooth(logKEH));
[~,xm] = min(smooth(logKEM));

vline(log(pl(xl)))
vline(log(ph(xh)))
vline(log(pm(xm)))
title('STR dynamics, wait')
ylabel('dynamics strength (log)')
xlabs = [ -0.025, -0.02,-0.015,-0.01];
xticks(xlabs)
xticklabels(num2str(exp(xlabs')))


%%

figure(311)
clf
plot(log(pl),smooth(logKEL),'color','blue','linewidth',2)
hold on
plot(log(pm),smooth(logKEM),'color','black','linewidth',2)
plot(log(ph),smooth(logKEH),'color','red','linewidth',2)
xlabel('p wait')
ylabel('log[KE]')

%xlim([0.962,0.97])

[~,xl] = min(smooth(logKEL));
[~,xh] = min(smooth(logKEH));
[~,xm] = min(smooth(logKEM));

vline(log(pl(xl)))
vline(log(ph(xh)))
vline(log(pm(xm)))
title('STR dynamics, wait')

%% reg 1 redo with a defined weight axis

num = 33;

reg_idx = 1;
isgen = false;

blocktype = 'mixed';
[fm,g, usepblock, ~] = call(dytype, num,reg_idx,idx,epoch,stage, blocktype, isgen);

wm = [0,50];

nw = 100;
weight_mixed_x = linspace(-wm(1),wm(1),nw);
weight_mixed_y = linspace(-wm(2),wm(2),nw);

%get the pwait values along this line
pw_weight_mixed = zeros(nw,1);
ke_weight_mixed = zeros(nw,1);

for j = 1:nw
    [~,idx_x] = min(abs(fm.X0-weight_mixed_x(j)));
    [~,idx_y] = min(abs(fm.Y0-weight_mixed_y(j)));
    pw_weight_mixed(j) = fm.pi(idx_x,idx_y);
    ke_weight_mixed(j) = log10(fm.sgrad_norm(idx_x,idx_y));
end


blocktype = 'high';
[fh,~, ~, ~] = call(dytype, num,reg_idx,idx,epoch,stage, blocktype, isgen);
wh = [194,108];

nw = 100;
weight_high_x = linspace(-wh(1),wh(1),nw);
weight_high_y = linspace(-wh(2),wh(2),nw);

%get the pwait values along this line
pw_weight_high = zeros(nw,1);
ke_weight_high = zeros(nw,1);

for j = 1:nw
    [~,idx_x] = min(abs(fh.X0-weight_high_x(j)));
    [~,idx_y] = min(abs(fh.Y0-weight_high_y(j)));
    pw_weight_high(j) = fh.pi(idx_x,idx_y);
    ke_weight_high(j) = log10(fh.sgrad_norm(idx_x,idx_y));
end

blocktype = 'low';
[fl,~, ~, ~] = call(dytype, num,reg_idx,idx,epoch,stage, blocktype, isgen);
wl = [190,108];
wm = [0,50];

nw = 100;
weight_low_x = linspace(-wl(1),wl(1),nw);
weight_low_y = linspace(-wl(2),wl(2),nw);

%get the pwait values along this line
pw_weight_low = zeros(nw,1);
ke_weight_low = zeros(nw,1);

for j = 1:nw
    [~,idx_x] = min(abs(fl.X0-weight_low_x(j)));
    [~,idx_y] = min(abs(fl.Y0-weight_low_y(j)));
    pw_weight_low(j) = fl.pi(idx_x,idx_y);
    ke_weight_low(j) = log10(fl.sgrad_norm(idx_x,idx_y));
end

%% interpolate between a given region
p1 = 0.97;
p2 = 0.985;
pw_weight_mixed = smooth(pw_weight_mixed);
ke_weight_mixed = smooth(ke_weight_mixed);
pw_mixed_interp = linspace(p1,p2,100);
[~,ke_x1] = min(abs(p1-pw_weight_mixed));
[~,ke_x2] = min(abs(p2-pw_weight_mixed));
ke_mixed_interp = interp1(pw_weight_mixed(ke_x2:ke_x1), ke_weight_mixed(ke_x2:ke_x1),pw_mixed_interp,'spline');


pw_weight_high = smooth(pw_weight_high);
ke_weight_high = smooth(ke_weight_high);
pw_high_interp = linspace(p1,p2,100);
[~,ke_x1] = min(abs(p1-pw_weight_high));
[~,ke_x2] = min(abs(p2-pw_weight_high));
ke_high_interp = interp1(pw_weight_high(ke_x2:ke_x1), ke_weight_high(ke_x2:ke_x1),pw_high_interp,'spline');

pw_weight_low = smooth(pw_weight_low);
ke_weight_low = smooth(ke_weight_low);
pw_low_interp = linspace(p1,p2,100);
[~,ke_x1] = min(abs(p1-pw_weight_low));
[~,ke_x2] = min(abs(p2-pw_weight_low));
ke_low_interp = interp1(pw_weight_low(ke_x2:ke_x1), ke_weight_low(ke_x2:ke_x1),pw_low_interp,'spline');


%%

figure(311)
clf
plot(log10(pw_low_interp),ke_low_interp,'color','blue','linewidth',2)
hold on
plot(log10(pw_mixed_interp),ke_mixed_interp,'color',[0.5,0,0.4],'linewidth',2)
plot(log10(pw_high_interp),ke_high_interp,'color','red','linewidth',2)
xlabel('p wait')
ylabel('log[KE]')

%xlim([0.962,0.97])

[~,xl] = min(ke_low_interp);
[~,xh] = min(ke_high_interp);
[~,xm] = min(ke_mixed_interp);

vline(log10(pw_low_interp(xl)));
vline(log10(pw_high_interp(xh)));
vline(log10(pw_mixed_interp(xm)));
title('STR dynamics, wait')


%% function calls

function [f,g,beh, pidecoder, usepblock, usepi] = call(dytype,num,reg_idx,idx,epoch,stage,blocktype, isgen, pref,eps)

    if reg_idx == 0
        usepi = false;
        usepblock = true;
        blocktype = 'mixed';
    else
        usepi = true;
        usepblock = false;
        
    end
    
    switch dytype
        case 'full_cl'      
            dbase = '/Users/dhocker/projects/dynamics/results/20231003/full_cl/';
            datadir = strcat(dbase,'dynamics/flows/');
            datadir_KE = strcat(dbase,'/dynamics/KEmin_constrained/');
        case 'full_cl_test'
            %datadir = '/Users/dhocker/projects/dynamics/results/rnn/ac/20230206_fullclstudy/full_cl/dynamics/';
            
            dbase = '/Users/dhocker/projects/dynamics/results/20231003/full_cl/';
            datadir = strcat(dbase,'flows_end/');
            switch eps
                case 0.1
                    et = 'eps1/';
                case 0.01
                    et = 'eps01/';
                case 0.001
                    et = 'eps001/';
                case 0.05
                    et = 'eps05/';
                otherwise
                    et = 'eps01/';
            end
            datadir_KE = strcat(dbase,'/ke_eps/',et);
  
    
        case 'nok_cl'
            datadir = '/Users/dhocker/projects/dynamics/results/20231003/nok_cl/dynamics/flows/';
            datadir_KE = '/Users/dhocker/projects/dynamics/results/20230206_clstudy/nok_cl/dynamics/KEmin_constrained/';
            
         
            
    end

    if reg_idx == 0
        if isgen
            datname = strcat(datadir,'flowfields_rnn_curric_',num2str(num),'_',stage,'_',num2str(idx),'reg_0_mixed_',epoch,'.mat');
        else

            datname = strcat(datadir,'flowfields_rnn_curric_',num2str(num),'_',stage,'_',num2str(idx),'reg_0_mixed_',epoch,'_boutique',pref,'.mat');
        end
    else
        if isgen
            datname = strcat(datadir,'flowfields_rnn_curric_',num2str(num),'_',stage,'_',num2str(idx),'reg_1_',blocktype,'_',epoch,'.mat');
        else
            datname = strcat(datadir,'flowfields_rnn_curric_',num2str(num),'_',stage,'_',num2str(idx),'reg_1_',blocktype,'_',epoch,'_boutique.mat');
        end
    end

    datname_KE = strcat(datadir_KE,'kemin_rnn_curric_',num2str(num),'_',stage,'_',num2str(idx),'reg_',num2str(reg_idx),'_',blocktype,'_',epoch,'.mat');
    
    f = load(datname);
    if isgen
        g = f;
    else       
        g = load(datname_KE);
    end

    beh = f;
    pidecoder = f; % TODO
    

    %rename the samples, because there was a mistake
    %incorrect ordering: mixed, high, low, 5, 10, 20,40,80,...
    %correct ordering  :  5     10,   20, 40 ,80,  M, H, L


    f.samps5new = f.samps_mixed;
    f.samps10new = f.samps_high;
    f.samps20new = f.samps_low;
    f.samps40new = f.samps_5;
    f.samps80new = f.samps_10;
    f.sampsMnew = f.samps_20;
    f.sampsHnew = f.samps_40;
    f.sampsLnew = f.samps_80;

    f.samps_5 = f.samps5new;
    f.samps_10 = f.samps10new;
    f.samps_20 = f.samps20new;
    f.samps_40 = f.samps40new;
    f.samps_80 = f.samps80new;
    f.samps_mixed = f.sampsMnew;
    f.samps_high = f.sampsHnew;
    f.samps_low = f.sampsLnew;

    f = rmfield(f,'samps5new');
    f = rmfield(f,'samps10new');
    f = rmfield(f,'samps20new');
    f = rmfield(f,'samps40new');
    f = rmfield(f,'samps80new');
    f = rmfield(f,'sampsMnew');
    f = rmfield(f,'sampsHnew');
    f = rmfield(f,'sampsLnew');

    

end


function [h] = genflowfield(fnum, f,~, reg_idx, res, bgtype, zuse_idx)
    % will make the figure, then output the handle

    res1 = res(1);
    res2 = res(2);
    res3 = res(3);
    
    % set up the background amplitude
    % PC 1 is first component
    % PC2 is second component
    % assumes data is in form of D(y,x) for imagesc, which always confuses me
    
    h = figure(fnum);
    clf
    hold on

    xvec = f.X0; %PC2
    yvec = f.Y0; % PC1. 
    zvec = f.Z0;
    
    nx = length(xvec);
    ny = length(yvec);
    
    for m = 1:numel(zuse_idx)
        zuse_m = zuse_idx(m);
        [X,Y] = meshgrid(xvec,yvec);
    
        %run the softmax
        switch bgtype
            case 'block'
                %blocp = f.blockp(1:res1:end,1:res2:end,1,:);
                blocp = f.blockp;
                if reg_idx == 0
                    pblocks = zeros(nx,ny,3);
                    for j = 1:nx
                        for k = 1:ny
                            pblocks(j,k,:) = softmax(squeeze(blocp(j,k,1,:)));
                        end
                    end
                end
        
                %try nan-out unwanted bits
                maskmixed = 0*ones(size(pblocks(:,:,1)));
                maskmixed(pblocks(:,:,1) < pblocks(:,:,2) + pblocks(:,:,3)) = nan;
        
                maskhigh = ones(size(pblocks(:,:,1)));
                maskhigh(pblocks(:,:,2) < pblocks(:,:,1) + pblocks(:,:,3)) = nan;
        
                masklow = 2*ones(size(pblocks(:,:,2)));
                masklow(pblocks(:,:,3) < pblocks(:,:,1) + pblocks(:,:,2)) = nan;
        
                maskall = nan*ones(3,size(maskmixed,1),size(maskmixed,2));
                maskall(1,:,:) = maskmixed;
                maskall(2,:,:) = maskhigh;
                maskall(3,:,:) = masklow;
        
                maskall = squeeze(sum(maskall,1,'omitnan'));
        
                
                shading interp
                hh = imagesc([xvec(1), xvec(end)], [yvec(1), yvec(end)],maskall');  % trying to plot 
                set(hh, 'AlphaData', 0.5-0.5*isnan(maskall'))

                map = [0.5,0,0.4;
                       1,0,0;
                       0,0,1;];
                colormap(map)

        case 'pi'
            shading interp
            surf(X,Y,zvec(zuse_m)*ones(size(Y)),log(f.pi(:,:,zuse_m)'),'edgecolor','none','facealpha',0.4)
            shading interp
            clim([-0.5,0]);

        otherwise
            colormap(parula)
            shading interp
            surf(X,Y,zvec(zuse_m)*ones(size(Y)),f.sgrad_norm(:,:,zuse_m)','edgecolor','none','facealpha',0.4)
            shading interp
            if reg_idx == 0
                clim([0,1000])
                disp('reg idx == 0')
            else
                clim([0,5000])
            end
        end
    
        xlim([xvec(1), xvec(end)])
        ylim([yvec(1), yvec(end)])
    
        xlabel('PC 1')
        ylabel('PC 2')
        set(gca,'ydir','normal')  % zero is now bottom left corner, 
        hold on
        colorbar
        title('2d')

        %return x,y,z, to lower res
        xvec = f.X0(1:res1:end); %PC2
        yvec = f.Y0(1:res2:end); % PC1. 
        zvec = f.Z0(1:res3:end);
        nx = length(xvec);
        ny = length(yvec);
        zuse_m = zuse_idx(m);
        [X,Y] = meshgrid(xvec,yvec);         
    
        dir1 = f.sgrad_x(:,:,zuse_m)./(sqrt(2*f.sgrad_norm(:,:,zuse_m))); % pc1 gradient direction
        dir2 = f.sgrad_y(:,:,zuse_m)./(sqrt(2*f.sgrad_norm(:,:,zuse_m))); % pc2 gradient direction
        dir3 = f.sgrad_z(:,:,zuse_m)./(sqrt(2*f.sgrad_norm(:,:,zuse_m)));
    
        U = dir1(1:res1:end, 1:res2:end)';
        V = dir2(1:res1:end, 1:res2:end)';
        W = dir3(1:res1:end, 1:res2:end)';

        switch bgtype
            case 'block'
                quiver(X,Y,U,V,1, 'color','white','linewidth',2)
            otherwise
                quiver3(X,Y,zvec(zuse_m)*ones(size(Y)),U,V,W,1,'linewidth',2, 'color',[0.5,0.5,0.5])
        end
        
    end 
end