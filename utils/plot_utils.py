from matplotlib.colors import LinearSegmentedColormap, colorConverter

def simple_cmap(*colors, name='none'):
    """Create a colormap from a sequence of rgb values.
    cmap = simple_cmap((1,1,1), (1,0,0)) # white to red colormap
    cmap = simple_cmap('w', 'r')         # white to red colormap

    From Alex Williams
    """

    # check inputs
    n_colors = len(colors)
    if n_colors <= 1:
        raise ValueError('Must specify at least two colors')

    # make sure colors are specified as rgb
    colors = [colorConverter.to_rgb(c) for c in colors]

    # set up colormap
    r, g, b = colors[0]
    cdict = {'red': [(0.0, r, r)], 'green': [(0.0, g, g)], 'blue': [(0.0, b, b)]}
    for i, (r, g, b) in enumerate(colors[1:]):
        idx = (i+1) / (n_colors-1)
        cdict['red'].append((idx, r, r))
        cdict['green'].append((idx, g, g))
        cdict['blue'].append((idx, b, b))

    return LinearSegmentedColormap(name, {k: tuple(v) for k, v in cdict.items()})