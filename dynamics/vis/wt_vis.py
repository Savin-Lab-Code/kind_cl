import numpy as np
import matplotlib.pyplot as plt


def plotblocks(wt_dict, idx=1, name='title'):
    """
    plots sensitivity to reward, by blocks
    :param wt_dict: wait-time dicitionary from parsesimulation()
    :param idx: id for output figure
    :param name: plot title
    :return:
    """
    Vvec = np.array([5, 10, 20, 40, 80])
    fig = plt.figure(idx, figsize=(4, 4))
    # plt.errorbar(np.log2(Vvec),wt_dict['wt'],wt_dict['wt_sem'],label='all')
    plt.errorbar(np.log2(Vvec), wt_dict['wt_mixed'], wt_dict['wt_mixed_sem'], label='mixed', color='gray')
    plt.errorbar(np.log2(Vvec), wt_dict['wt_low'], wt_dict['wt_low_sem'], label='low', color='blue')
    plt.errorbar(np.log2(Vvec), wt_dict['wt_high'], wt_dict['wt_high_sem'], label='high', color='red')
    plt.xticks(np.log2(Vvec))
    plt.gca().set_xticklabels(Vvec)
    plt.legend()
    plt.title(name)
    if 'p_signrank' in wt_dict:
        text2plot = 'sign-rank p: {:.2e} \n ranksum p: {:.2e}'.format(wt_dict['p_signrank'], wt_dict['p_ranksum'])
        plt.annotate(text2plot, (0.7, 0.7), xytext=(0.7, 0.7), textcoords='axes fraction')
    plt.show()

    #normalize things
    mu = wt_dict['wt_mixed'][2]
    fig2 = plt.figure(idx+1, figsize=(4, 4))
    plt.errorbar(np.log2(Vvec), wt_dict['wt_mixed']/mu, wt_dict['wt_mixed_sem']/(mu**2), label='mixed', color='gray')
    plt.errorbar(np.log2(Vvec), wt_dict['wt_low']/mu, wt_dict['wt_low_sem']/(mu**2), label='low', color='blue')
    plt.errorbar(np.log2(Vvec), wt_dict['wt_high']/mu, wt_dict['wt_high_sem']/(mu**2), label='high', color='red')
    plt.xticks(np.log2(Vvec))
    plt.gca().set_xticklabels(Vvec)
    plt.legend()
    plt.title(name+': normed')
    if 'p_signrank' in wt_dict:
        text2plot = 'sign-rank p: {:.2e} \n ranksum p: {:.2e}'.format(wt_dict['p_signrank'], wt_dict['p_ranksum'])
        plt.annotate(text2plot, (0.7, 0.7), xytext=(0.7, 0.7), textcoords='axes fraction')
    plt.show()


    return fig, fig2

def plotblocks_subfig(wt_dict, fig, ax, idx, name='title'):
    """
    plots sensitivity to reward, by blocks, in a subfigure
    :param wt_dict: wait-time dicitionary from parsesimulation()
    :param idx: id for output figure
    :param name: plot title
    :return:
    """
    Vvec = np.array([5, 10, 20, 40, 80])

    # plt.errorbar(np.log2(Vvec),wt_dict['wt'],wt_dict['wt_sem'],label='all')
    ax[idx].errorbar(np.log2(Vvec), wt_dict['wt_mixed'], wt_dict['wt_mixed_sem'], label='mixed', color='gray')
    ax[idx].errorbar(np.log2(Vvec), wt_dict['wt_low'], wt_dict['wt_low_sem'], label='low', color='blue')
    ax[idx].errorbar(np.log2(Vvec), wt_dict['wt_high'], wt_dict['wt_high_sem'], label='high', color='red')
    ax[idx].set_xticks(np.log2(Vvec))
    ax[idx].set_xticklabels(Vvec)
    ax[idx].legend()
    ax[idx].set_title(name)
    if 'p_signrank' in wt_dict:
        text2plot = 'sign-rank p: {:.2e} \n ranksum p: {:.2e}'.format(wt_dict['p_signrank'], wt_dict['p_ranksum'])
        ax[idx].annotate(text2plot, (0.7, 0.7), xytext=(0.7, 0.7), textcoords='axes fraction')



def plotsensitivity(wt_dict, idx=1, name='title'):
    """
    will plot the sensitivity curve across all opt-outs
    :param wt_dict: wait time dictionary from parsesimulation()
    :param idx: id of the output figure
    :param name: title of plot
    :return:
    """
    Vvec = np.array([5, 10, 20, 40, 80])
    fig = plt.figure(idx, figsize=(4, 4))
    # plt.errorbar(np.log2(Vvec),wt_dict['wt'],wt_dict['wt_sem'],label='all')
    plt.errorbar(np.log2(Vvec), wt_dict['wt'], wt_dict['wt_sem'], label='mixed')
    plt.xticks(np.log2(Vvec))
    plt.gca().set_xticklabels(Vvec)
    plt.legend()
    plt.title(name)
    plt.show()

    return fig


# TODO add regression plotting
def plot_modelAgnosticTest(wtdat, reg_histdat, reg_blockdat, txt='1', includeblock=False):
    """
    plots results f model-agnostic tests
    :param wtdat: [p, p_sr, wt_lt, wt_gt]. rank sum result, sign-rank result, wt (prev < 20), wt (prev > 20)
    :param reg_histdat: [m_hist, rs_hist, pvec_hist]. trial history regression result. slope, rsquared, pvals
    :param reg_blockdat: [m_block, rs_block, pvec_block]. same, but regression of trial history + blocks.
    :param txt: (str) handle for title
    :param includeblock: (bool) for switching to regression with block as an oroginal regression coefficient
    :return: figure handles
    """
    p_rs = wtdat[0]
    p_sr = wtdat[1]
    wt_lt = wtdat[2]
    wt_gt = wtdat[3]

    if includeblock:
        m_hist = reg_blockdat[0]  #coefficients
        rs_hist = reg_blockdat[1]  #r-swuared
        pvec_hist = reg_blockdat[2]  # pvals for coefficients
    else:
        m_hist = reg_histdat[0]
        rs_hist = reg_histdat[1]
        pvec_hist = reg_histdat[2]


    # put in plotting
    n_lt = len(wt_lt)
    n_gt = len(wt_gt)

    lt_sem = np.std(wt_lt) / np.sqrt(n_lt)
    gt_sem = np.std(wt_gt) / np.sqrt(n_gt)
    lt_med = np.nanmedian(wt_lt)
    gt_med = np.nanmedian(wt_gt)

    f = plt.figure()
    plt.bar([1], [lt_med], color='b')
    plt.bar([2], [gt_med], color='r')
    plt.errorbar([1, 2], [lt_med, gt_med], [lt_sem, gt_sem], color='k', ls='none')
    plt.xticks([1, 2], labels=['< 20', '>20'])
    if includeblock:
        plt.title('model agnostic test w/ block: ' + txt)
    else:
        plt.title('model agnostic test: ' + txt)
    text2plot = 'sign-rank p: {:.2e} \n ranksum p: {:.2e}'.format(p_sr, p_rs)
    plt.annotate(text2plot, (0.7, 0.7), xytext=(0.7, 0.7), textcoords='axes fraction')

    # move to plot
    g = plt.figure()
    wtmax = np.max([np.max(wt_lt), np.max(wt_gt)])

    bins = np.linspace(0, wtmax, 40)
    plt.hist(wt_lt, label='<20', bins=bins, alpha=0.5, color='b')
    plt.hist(wt_gt, label='>20', bins=bins, alpha=0.5, color='r')
    plt.vlines(lt_med, 0, 10, color='b')
    plt.vlines(gt_med, 0, 10, color='r')
    plt.title('distributions of cond. wt. ' + txt)
    plt.legend()
    plt.xlabel('time (s)')
    plt.annotate(text2plot, (0, 0.7), xytext=(0, 0.7), textcoords='axes fraction')
    plt.show()

    h = plt.figure()
    nreg = len(m_hist)
    plt.plot(range(nreg), m_hist)
    plt.hlines(0, 0, nreg + 1, 'k')
    plt.xlim([0, nreg])
    plt.xlabel('trials back')
    if includeblock:
        text2plot = 'regressison on trial history w/ block: R2 = {:0.2f}: {}'.format(rs_hist, txt)
    else:
        text2plot = 'regressison on trial history: R2 = {:0.2f}: {}'.format(rs_hist, txt)
    plt.title(text2plot)

    if includeblock:
        for k in range(nreg):  # typo that I forgot to check if this is significant or not for last oen
            if pvec_hist[k] < 0.001:
                plt.annotate('*', (k, 0.0), fontsize=10)
        if nreg == 9:
            plt.xticks(range(0, nreg),labels=['0', '1', '2', '3', '4', '5', '6', 'B', 'B x r_0'])
        else:
            plt.xticks(range(0, nreg), labels=['0', '1', '2', '3', '4', '5', '6', 'B'])
    else:
        for k in range(nreg):
            if pvec_hist[k] < 0.001:
                plt.annotate('*', (k, 0.0), fontsize=10)




    return f, g, h
