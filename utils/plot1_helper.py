
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib._color_data as mcd
from matplotlib import rc,rcParams

def plot_acc(x1, x2, x3, name):
    ylim = [-48, 51]
    #ylim2 = [0, 0.1]

    fig = plt.figure(figsize =(20, 9) )
    ax = fig.add_subplot(111)
    #ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.

    #n_groups = 7 # num of group of bars
    #n_groups = 2 # num of group of bars
    n_groups = 3 # num of group of bars

    index = np.arange(n_groups) * 2.5
    bar_width = 0.4
    opacity = 1.0
    gap = 0.1

    #ax.plot([-0.3, 4.3], [0, 0], 'k--', zorder=0)
    #ax.plot([-0.3, 3.7], [-39.67, -39.67], 'k--', zorder=0)
    #plt.text(3.8, -41.5, '= F(0)', fontsize=35.0)

    ax.plot([-0.3, 6.8], [0, 0], 'k--', zorder=0)
    ax.plot([-0.3, 5.9], [-39.67, -39.67], 'k--', zorder=0)
    plt.text(6.0, -41.5, '= F(0)', fontsize=35.0)

    lns1 = ax.bar(index + (bar_width + gap)*0, x1, bar_width,
                     alpha=opacity,
                     color='dodgerblue',
                     hatch='+',
                     label='Whole Sample ' + r'$F(X)$', zorder=1)

    lns3 = ax.bar(index + (bar_width + gap)*2, x3, bar_width,
                     alpha=opacity,
                     color="xkcd:" + 'salmon',
                     hatch='O',
                     label='Other Region ' + r'$F(X - x^*)$', zorder=2)

    lns2 = ax.bar(index + (bar_width + gap)*1, x2, bar_width,
                     alpha=opacity,
                     color="xkcd:" + 'turquoise',
                     hatch='/',
                     label='Focus Region ' + r"$F(x^*)$", zorder=3)


    plt.xticks(index + (bar_width + gap), ('Positive', 'Negative', ''))

    legend = fig.legend(loc = 'upper left', ncol=2, prop=dict(size=37), columnspacing=0.4,  borderaxespad=0.2, 
               bbox_to_anchor=(0,1), bbox_transform=ax.transAxes) 

    legend.get_frame().set_linewidth(5)

    if False:
        fig.legend(loc = 'upper left', ncol=2, prop=dict(weight='bold', size=36), 
               columnspacing=0.4, labelspacing=0.2, handletextpad=0.3,
               bbox_to_anchor=(0,1), bbox_transform=ax.transAxes) 
            # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html

    #ax.set_ylabel('Prediction Value', fontsize=50, labelpad=20)
    ax.set_ylabel('Prediction Value', fontsize=50, labelpad=10)

    ax.set_ylim(ylim)
    #ax2.set_ylim(ylim2)

    ax.xaxis.set_tick_params(labelsize=44)
    ax.yaxis.set_tick_params(labelsize=44)

    ax.set_yticks([-40, -20, 0, 20])

    ax.get_xticklabels()[0].set_weight("bold")

    plt.savefig(name, dpi = 300, bbox_inches='tight')


