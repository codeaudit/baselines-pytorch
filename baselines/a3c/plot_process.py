import os
from time import sleep

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from bokeh.layouts import column
from bokeh.mpl import to_bokeh
from bokeh.plotting import show
from bokeh.io import output_file, save
from bokeh.charts import defaults
from bokeh.models import Div

sns.set(color_codes=True)

defaults.width = 800
defaults.height = 400
defaults.tools = 'pan,box_zoom,wheel_zoom,box_select,hover,resize,reset,save'


def plot_multiple_agents(results_path, results_name, num_workers):
    def _get_data_from_files():
        dfs = []
        for i_work in range(num_workers):
            dfs.append(pd.read_json(os.path.abspath(files_name.format(i_work)), lines=True))

        df = pd.DataFrame()
        df['epoch'] = dfs[0]['epoch']
        columns = list(dfs[0].columns.values)
        columns.remove('epoch')
        columns.remove('time')
        columns.remove('step')

        # ensure all matrices of the same size,
        # since not all agents necessarily reached the same epoch
        max_epochs = 0
        for d in dfs:
            epochs = d.shape[0]
            max_epochs = max(max_epochs, epochs)
        data = {}
        for c in columns:
            data[c] = np.vstack([df[c].as_matrix() for df in dfs])
        return data, columns

    files_name = os.path.join(results_path, results_name)
    plot_path = '{}.{}'.format(files_name, 'html')
    files_name += '_{}.json'

    data, columns = _get_data_from_files()

    figs = []
    for i, col in enumerate(columns):
        fig = plt.figure(i)
        axis = sns.tsplot(data=data[col], err_style="unit_traces")
        # axis = sns.tsplot(data=data[c], ci=[50, 70, 90], legend=True)
        axis.set_title(col)
        axis.set_xlabel('epochs')
        axis.set_ylabel(col)
        fig.add_axes(axis)
        figs.append(to_bokeh(fig))

    plot = column(Div(text='<h1 align="center">{}</h1>'.format(results_name)), *figs)
    output_file(plot_path, title=results_name)
    save(plot)


def plot_all_agents_loop(results_path, results_name, num_workers):

    first_time = True
    files_name = os.path.join(results_path, results_name)
    plot_path = '{}.{}'.format(files_name, 'html')

    # allow for all agents to create their plots
    sleep(20)

    while True:
        sleep(10)
        try:
            plot_multiple_agents(results_path, results_name, num_workers)
        except ValueError:
            pass
        if first_time:
            print('Plot file saved at: {}'.format(os.path.abspath(plot_path)))
            first_time = False

# if __name__ == '__main__':
#     plot_multiple_agents('/home/nadavb/rl/baselines-pytorch/baselines/a3c/results/pong', 'results', 8)
