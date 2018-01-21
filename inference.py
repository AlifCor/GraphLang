import spectral
import scipy
import numpy as np
from scipy import sparse
import pandas as pd


from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go


def graph_magic(all_weights):
    NEIGHBORS = 100
    weights = spectral.filter_neighbors(all_weights, NEIGHBORS)
    
    degrees = np.sum(weights, axis=0)

    D = np.diag(degrees)
    W = weights
    L = D - W

    inv_sqrt_D = np.diag(1 / np.diag(D**(0.5)))

    normalized_laplacian = inv_sqrt_D @ L @ inv_sqrt_D
    
    return sparse.linalg.eigsh(normalized_laplacian, k=10, which='SM') # which='SA' gives us similar results

def get_plotly_figure(y, eigenvectors, unknown_label):
    news_target_names = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']
    df_targets = pd.DataFrame(news_target_names)
    parent_cat = ['comp.', 'rec.', 'religion', '.politics.', 'sci.', 'misc.forsale']

    parent_cat_ind = []
    for c in parent_cat:
        ind = df_targets[df_targets[0].apply(lambda x: c in x).values].index.values
        if c == 'religion':
            ind = np.append(ind, 0)
        parent_cat_ind.append(ind)
    
    traces = []
    
    labels = sorted(set(y))
    labels_to_name = {l:parent_cat[l] for l in labels[:len(parent_cat_ind)]}
    labels_to_name[unknown_label] = 'noname'


    for label in labels:
        label_mask = y == label
        #'''
        axis_x = eigenvectors[:, 1][label_mask]
        axis_y = eigenvectors[:, 2][label_mask]
        axis_z = eigenvectors[:, 3][label_mask]
        
        trace = go.Scatter3d(
            x=axis_x,
            y=axis_y,
            z=axis_z,
            name=labels_to_name[label],
            mode='markers',
            marker=dict(
                size=12,
                color=label,
                line=dict(
                    width=2,
                    color='black'
                )
            )
        )
    
        traces.append(trace)
        
    layout = go.Layout(
        hovermode= 'closest',
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        )
    )

    data = traces

    fig = go.Figure(data=data, layout=layout)
    return fig


def plot_new_point(X, y, vectorizer, message):
    
    data = vectorizer.transform((message))
    X_2 = scipy.sparse.csr_matrix.todense(data)
    y_2 = 10
    new_X = np.append(X, X_2, axis=0)
    new_y = np.append(y, y_2)
    distances = spectral.features_to_dist_matrix(new_X, metric='cosine')

    if np.count_nonzero(np.isnan(distances)) > 0:
        print('there are some nan')
        distances = np.nan_to_num(distances, copy=False)
    
    all_weights = spectral.dist_to_adj_matrix(distances, 'gaussian')
    
    eigenvalues, eigenvectors = graph_magic(all_weights)
    
    return get_plotly_figure(new_y, eigenvectors, y_2)