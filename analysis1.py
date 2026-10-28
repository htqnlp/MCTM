import seaborn as snNew
import pandas as pdNew
import matplotlib.pyplot as pltNew

filepath = 'C:/Users/31007/Desktop'
fig_name = 'analysis1.eps'
fig_path = filepath + '/' + fig_name
snNew.set_style('white')
snNew.despine()

array = [[0.796, 0.736, 0.82],
         [0.80, 0.747, 0.846],
         [0.817, 0.752, 0.85],
         [0.808, 0.743, 0.826],
         [0.799, 0.742, 0.836]]

DetaFrame_cm = pdNew.DataFrame(array, [3, 6, 9, 12, 15], ['Ori.', 'Hard.', 'Trans.'])
fig = snNew.lineplot(data=DetaFrame_cm, markers=['o', 'd', 'h'])
fig.legend(loc='center right', bbox_to_anchor=(0.95, 0.33), ncol=1)
pltNew.xlabel('Number of Experts')
pltNew.ylabel('Similarity Tasks Score')
pltNew.xticks([3, 6, 9, 12, 15])
scatter_fig = fig.get_figure()
scatter_fig.savefig(fig_path, dpi=400)
pltNew.show()
