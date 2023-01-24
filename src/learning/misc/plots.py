import matplotlib.pyplot as plt


def generate_BS_scatter_plot(df):
    iteration = '0'
    df.rename(columns={iteration: 'Density'}, inplace=True)
    df.plot.scatter(x='B', y='S', c='Density', colormap='jet', alpha=0.5)
    plt.show()
