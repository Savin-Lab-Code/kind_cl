% Fig 6B,C. flow field for OFC, and dynamics strength for STR

%% Fig 6B. OFC flow field
addpath(genpath('~/projects/dynamics/'))


dytype = 'full_cl'; %
eps = 0.01;




reg_idx = 0;
idx = 10;
epoch = 'wait';
stage = 'block';
pref = '';
bgtype = 'block';

iv = [1,1,1];
res = [2,2,2];

%samples to use? need to dial in
usevec = [2,2,3];  % for ofc

fnum = 17;
zuse_idx = 1;

% flow field data
f = load('flowfields_rnn_curric_33_block_10reg_0_mixed_wait_boutique.mat');
% KE data
g = load('kemin_rnn_curric_33_block_10reg_0_mixed_wait.mat');

[h] = genflowfield(fnum, f,reg_idx, res, bgtype, zuse_idx);

figure(fnum)


%% add the KE points

mask = g.labels > -1; % -1 is a noise cluster from DBSCAN
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
%pt_list = [1,20];
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


        %[efuns,evals] = eig(squeeze(g.jac_pc(pt_idx,:,:)))
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


%% add trajectories from each block
% grab trajectories in which pblock was particularly high. i.e., confident
% trials

figure(fnum)

%clf
%hold on
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


%% STR dynamics. Fig 6C
addpath(genpath('~/projects/dynamics/'))
num = 23;
bgtype = 'pi';
reg_idx = 1;
isgen = false;


fm = load('flowfields_rnn_curric_23_block_10reg_1_mixed_wait_boutique.mat');
res = [1,1,1];
fnum = 186;
[hm] = genflowfield(fnum, fm,reg_idx, res, bgtype, zuse_idx);
figure(186)
title('mixed')
%get the y dim for str
test = cellfun(@(x) mean(x(:,2),1),fm.samps_mixed);
ymean= mean(test);
[~,yuse_idx_m] = min(abs(ymean-fm.Y0));

logKEM = log10(0.5*abs(fm.sgrad_x(:,yuse_idx_m,zuse_idx)).^2);
pm = fm.pi(:,yuse_idx_m);


fh = load('flowfields_rnn_curric_23_block_10reg_1_high_wait_boutique.mat');
fnum = 187;
[hh] = genflowfield(fnum, fh,reg_idx, res, bgtype, zuse_idx);
figure(187)
title('high')
%get the y dim for str
test = cellfun(@(x) mean(x(:,2),1),fh.samps_mixed);
ymean= mean(test);
[~,yuse_idx_h] = min(abs(ymean-fh.Y0));

logKEH = log10(0.5*abs(fh.sgrad_x(:,yuse_idx_m,zuse_idx)).^2);
ph = fh.pi(:,yuse_idx_h); 


fl = load('flowfields_rnn_curric_23_block_10reg_1_low_wait_boutique.mat');
blocktype = 'low';
fnum = 188;
[hl] = genflowfield(fnum, fl,reg_idx, res, bgtype, zuse_idx);
figure(188)
title('low')
%get the y dim for str
test = cellfun(@(x) mean(x(:,2),1),fl.samps_mixed);
ymean= mean(test);
[~,yuse_idx_l] = min(abs(ymean-fl.Y0));
logKEL = log10(0.5*abs(fl.sgrad_x(:,yuse_idx_l,zuse_idx)).^2);
pl = fl.pi(:,yuse_idx_l);


% get 1D projections
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


%% function calls


function [h] = genflowfield(fnum, f, reg_idx, res, bgtype, zuse_idx)
    % will make the figure, then output the handle

    res1 = res(1);
    res2 = res(2);
    res3 = res(3);

    
    % get the mean value in the z dim
    
    
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
            %surf(X,Y,zvec(zuse_m)*ones(size(Y)),log(f.pi(1:res1:end,1:res2:end,zuse_m)'),'edgecolor','none','facealpha',0.4)
            surf(X,Y,zvec(zuse_m)*ones(size(Y)),log(f.pi(:,:,zuse_m)'),'edgecolor','none','facealpha',0.4)
            shading interp
            %caxis([-2.2,0])
            clim([-0.5,0]);

        otherwise
            colormap(parula)
            %h = imagesc([xvec(1), xvec(end)], [yvec(1), yvec(end)],f.sgrad_norm(:,:,zuse_m)')  % trying to plot 
            shading interp
            %surf(X,Y,zvec(zuse_m)*ones(size(Y)),f.sgrad_norm(1:res1:end,1:res2:end,zuse_m)','edgecolor','none','facealpha',0.4)
            surf(X,Y,zvec(zuse_m)*ones(size(Y)),f.sgrad_norm(:,:,zuse_m)','edgecolor','none','facealpha',0.4)
            shading interp
            if reg_idx == 0
                %caxis([0,300])
                clim([0,1000]);
                disp('reg idx == 0')
            else
                clim([0,5000]);
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
                %quiver3(X,Y,zvec(zuse_m)*ones(size(Y)),U,V,W,1, 'color','black')
                quiver(X,Y,U,V,1, 'color','white','linewidth',2)
            otherwise
                quiver3(X,Y,zvec(zuse_m)*ones(size(Y)),U,V,W,1,'linewidth',2, 'color',[0.5,0.5,0.5])
        end
        
    end 
end