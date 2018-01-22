import itertools

import numpy as np
import plotly.graph_objs as go
from matplotlib import pyplot as plt
from plotly.offline import iplot


# A function to plot the confusion matrix, taken from
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
# sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        div = cm.sum(axis=1)[:, np.newaxis]
        div[div == 0] = 1
        cm = cm.astype('float') / div

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot3D(eigenvectors, pred, infos, label_to_name, node_size=2, opacity=0.9):
    # copy past from above
    traces = []
    my_axis = dict(
        showbackground=False,
        zeroline=False,
        ticks=False,
        showgrid=False,
        showspikes=False,
        showticklabels=False,
        showtickprefix=False,
        showexponent=False)

    for label in sorted(set(pred)):
        label_mask = pred == label
        x = eigenvectors[:, 1][label_mask]
        y = eigenvectors[:, 2][label_mask]
        z = eigenvectors[:, 3][label_mask]
        trace = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            hoverinfo='text+name',
            name=label_to_name[label],
            mode='markers',
            marker=dict(
                size= node_size if label < 500 else node_size*5,
                color= label,
                colorscale= 'Portland',
                opacity= opacity if label < 500 else 1
            ),
            text=infos[label_mask]
        )

        traces.append(trace)
        layout = go.Layout(
            hovermode='closest',
            margin=dict(
                l=0,
                r=0,
                b=0,
                t=0
            ),
            scene=go.Scene(dict(
                xaxis=my_axis,
                yaxis=my_axis,
                zaxis=my_axis
            ))
        )

    data = traces

    fig = go.Figure(data=data, layout=layout)
    return iplot(fig)


def dis2text(arr, tr_id2name):
    text = ""
    for i in range(len(arr)):
        text += tr_id2name[i] + ' : ' + str(int(arr[i] * 1000) / 10) + "%" + "<br>"
    return text
