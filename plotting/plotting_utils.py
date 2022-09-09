import numpy as np
import matplotlib.pyplot as plt

from feedback_bound import theorem1_continual_bound


def get_bias_std_lines(runs_stats, preds_key, filter_fxn=None, targs_key=None):
    if filter_fxn:
        runs_stats = [run for run in runs_stats if filter_fxn(run)]
    runs_biases = [stat[preds_key] for stat in runs_stats]

    # take minimum number of feedback rounds for any of the runs
    cutoff = min([len(run) for run in runs_biases])
    runs_biases = np.stack([run[:cutoff] for run in runs_biases])
    bias_line, bias_std = np.mean(runs_biases, axis=0), np.std(runs_biases, axis=0)
    
    if targs_key is None:
        return bias_line, bias_std
    baseline = np.mean([stat[targs_key] for stat in runs_stats])
    return bias_line, bias_std, baseline

def get_theory_line(bias_line, baseline, initial_train_set_size, human_samples_per_round, model_samples_per_round):
    delta_n0 = bias_line[0] - baseline
    return theorem1_continual_bound(len(bias_line) - 1, initial_train_set_size, human_samples_per_round, 
                                    model_samples_per_round, delta_n0) + baseline


def plot_lines(ax, bias_line, bias_std, theory_line, baseline, bias_label, theory_label, baseline_label,
               label_size=55, linewidth=5, alpha=0.3, add_delta=True, tick_params=False, xlabel=None, ylabel=None, title=None):
    
    ax.plot(bias_line, label=bias_label, linewidth=linewidth)
    ax.fill_between(range(len(bias_line)), bias_line-bias_std, bias_line+bias_std, alpha=alpha)
    ax.plot(theory_line, label=theory_label, linewidth=linewidth)
    ax.axhline(y=baseline, c='g', linestyle='--', label=baseline_label, linewidth=linewidth)
    
    if add_delta:
        pad = 0.05 * (bias_line[0] - baseline)
        ax.annotate('', xy=[0, baseline+pad], xytext=[0, bias_line[0]-pad], arrowprops=dict(arrowstyle='<->', linewidth=np.round(linewidth-2), color='dimgrey'))
        offset = 0.02 * len(bias_line)
        ax.annotate(r'$\delta_{n_0}$', xy=[offset, (bias_line[0]+baseline)/2], color='black', fontsize=label_size, va='center')
    
    if tick_params: ax.tick_params(axis='both', which='major', labelsize=label_size)
    if xlabel: ax.set_xlabel(xlabel, fontsize=label_size)
    if ylabel: ax.set_ylabel(ylabel, fontsize=label_size)
    if title: ax.set_title(title, fontsize=label_size)


def standard_2_plot(plot_name, bias_line_1, bias_std_1, theory_line_1, baseline_1, bias_line_2, bias_std_2, theory_line_2, baseline_2, 
                    bias_label, theory_label, baseline_label, xlabel, ylabel, left_title, right_title,
                    figsize=(30, 12), label_size=55, linewidth=5, alpha=0.3, add_delta=True):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=True, sharex=True)

    plot_lines(ax1, bias_line_1, bias_std_1, theory_line_1, baseline_1, 
               bias_label, theory_label, baseline_label, 
               label_size, linewidth, alpha, tick_params=True,
               xlabel=xlabel, ylabel=ylabel, title=left_title, add_delta=add_delta)

    plot_lines(ax2, bias_line_2, bias_std_2, theory_line_2, baseline_2, 
               bias_label, theory_label, baseline_label, 
               label_size, linewidth, alpha, tick_params=True,
               xlabel=xlabel, title=right_title, add_delta=add_delta)

    ax2.yaxis.set_tick_params(labelbottom=True)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, 1.15), prop={'size': label_size})
    fig.tight_layout()
    plt.savefig(plot_name, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.show()


def standard_2x2_plot(plot_name, bias_line_1, bias_std_1, theory_line_1, baseline_1, bias_line_2, bias_std_2, theory_line_2, baseline_2,
                      bias_line_1v2, bias_std_1v2, theory_line_1v2, baseline_1v2, bias_line_2v2, bias_std_2v2, theory_line_2v2, baseline_2v2,
                      bias_label, theory_label, baseline_label, xlabel, top_ylabel, bottom_ylabel, left_title, right_title,
                      figsize=(30, 20), label_size=55, linewidth=5, alpha=0.3, custom_xticks=None,
                      delt_tl=True, delt_tr=True, delt_bl=True, delt_br=True):

    fig, ((ax11, ax12), (ax21, ax22)) = plt.subplots(2, 2, figsize=figsize, sharey='row')

    plot_lines(ax11, bias_line_1, bias_std_1, theory_line_1, baseline_1, 
               bias_label, theory_label, baseline_label,
               label_size, linewidth, alpha, tick_params=True,
               ylabel=top_ylabel, title=left_title, add_delta=delt_tl)

    plot_lines(ax12, bias_line_2, bias_std_2, theory_line_2, baseline_2, 
               bias_label, theory_label, baseline_label,
               label_size, linewidth, alpha, tick_params=True,
               title=right_title, add_delta=delt_tr)

    plot_lines(ax21, bias_line_1v2, bias_std_1v2, theory_line_1v2, baseline_1v2, 
               bias_label, theory_label, baseline_label,
               label_size, linewidth, alpha, tick_params=True,
               xlabel=xlabel, ylabel=bottom_ylabel, add_delta=delt_bl)

    plot_lines(ax22, bias_line_2v2, bias_std_2v2, theory_line_2v2, baseline_2v2, 
               bias_label, theory_label, baseline_label,
               label_size, linewidth, alpha, tick_params=True,
               xlabel=xlabel, add_delta=delt_br)

    ax12.yaxis.set_tick_params(labelbottom=True)
    ax22.tick_params(axis='both', which='major', labelsize=label_size, labelleft=True)

    if custom_xticks:
        ax11.set_xticks(custom_xticks)
        ax12.set_xticks(custom_xticks)
        ax21.set_xticks(custom_xticks)
        ax22.set_xticks(custom_xticks)

    handles, labels = ax11.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, 1.09), prop={'size': label_size})
    fig.tight_layout()
    plt.savefig(plot_name, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.show()
