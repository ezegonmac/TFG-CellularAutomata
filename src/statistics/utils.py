import numpy as np
from matplotlib import cm

def get_colors_by_threshold():
    """

    Returns:
        colors_by_threshold: dictionary of threshold value and color
    """
    
    thresholds = range(0, 10)
    
    cmap = cm.get_cmap('jet')
    colors = cmap(np.linspace(0, 1, len(thresholds)))
    colors_by_threshold = {thresholds[i]: colors[i] for i in range(len(thresholds))}
    
    return colors_by_threshold
