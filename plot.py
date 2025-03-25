# wandb-like plots

import numpy as np # no jnp here
import matplotlib
from matplotlib import pyplot as plt

# Wandb at home styling, will take better settings if someone has them

plt.style.use('default')
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams['axes.spines.right'] = False 
matplotlib.rcParams['axes.facecolor'] = '#f0f0f0'
matplotlib.rcParams['figure.facecolor'] = '#f0f0f0'
matplotlib.rcParams['grid.alpha'] = 0.4 
matplotlib.rcParams['axes.titlesize'] = 12
matplotlib.rcParams['axes.labelsize'] = 12
matplotlib.rcParams['xtick.labelsize'] = 10 
matplotlib.rcParams['ytick.labelsize'] = 10 
matplotlib.rcParams['legend.fontsize'] = 10 
matplotlib.rcParams['axes.titlecolor'] = 'grey'
matplotlib.rcParams['axes.labelcolor'] = 'grey'
matplotlib.rcParams['xtick.color'] = 'grey' 
matplotlib.rcParams['ytick.color'] = 'grey' 
matplotlib.rcParams['legend.labelcolor'] = 'grey' 

def plot_loss(train_hist, val_hist, i_eval, iter, run_name):

    plt.figure(figsize=(8, 4), dpi=100)
    ax = plt.gca()

    iter_train = range(len(train_hist))
    iter_eval = range(0, len(val_hist) * i_eval, i_eval)

    iter_eval = [x for x in iter_eval if x <= iter]   
    iter_eval = iter_eval[:len(val_hist)] # Ensures len

    l_train = plt.plot(iter_train, train_hist, label='Train', color='royalblue', linestyle='-', linewidth=2, marker='', alpha=0.7)
    val_line = plt.plot(iter_eval, val_hist, label='Val', color='palevioletred', linestyle='-', linewidth=2, marker='', alpha=0.7)

    plt.plot(iter_train[-1:], train_hist[-1:], marker='o', markersize=3, markerfacecolor='royalblue', markeredgecolor='none', linestyle='none')
        
    if val_hist:  # Check if val_hist is non-empty
        plt.plot(iter_eval[-1:], val_hist[-1:], marker='o', markersize=3, markerfacecolor='palevioletred', markeredgecolor='none', linestyle='none')

    plt.xlabel("Steps", labelpad=8, color='grey')
    plt.ylabel("Loss", labelpad=8, color='grey')
    plt.title(f"Train/Val Loss", fontsize=12, color='grey')

    legend = plt.legend(frameon=False, loc='upper right')

    for line in legend.get_lines():
        line.set_linewidth(2.5)
        line.set_solid_capstyle('round')

    ax.tick_params(axis='both', which='major', pad=8)

    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    y_ticks = [tick for tick in ax.get_yticks() if tick >= min(min(train_hist), min(val_hist)) and tick <= max(max(train_hist), max(val_hist))]

    ax.set_yticks(y_ticks[::2])
    ax.set_yticks(np.arange(min(y_ticks), max(y_ticks) + 0.4, 0.4), minor=False)

    plt.grid(axis='y', color='grey', linestyle='-', linewidth=0.5)
    plt.savefig(f"plots/{run_name}_plot.png", bbox_inches='tight', dpi=300)
    plt.clf()

