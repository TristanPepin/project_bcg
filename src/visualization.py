import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import os

from src import loading


def plot_confusion_matrix(cf_matrix):
    group_names = ['TN','FP','FN','TP']
    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
    plt.show()


def plot_history(history,name):
    plt.figure()
    plt.plot(history.history['loss'],label='loss')
    plt.plot(history.history['val_loss'],label='val_loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model {} history'.format(name))
    plt.show()

    if not os.path.exists(loading.PATH_MODELS + '/history_{}.png'.format(name)):
        plt.savefig(loading.PATH_MODELS + '/history_{}.png'.format(name))


def plot_ROC_curve(fpr,tpr,name):

    plt.figure()
    plt.plot(fpr,tpr)
    plt.xlabel('FRP')
    plt.ylabel('TRP')
    plt.title('ROC CURVE')
    plt.show()


def plot_gain_threshold(vec_threshold,gain):
    plt.figure()
    plt.plot(vec_threshold,gain)
    plt.xlabel('Threshold')
    plt.ylabel('Gain per client')
    plt.title('Best threshold : {}'.format(vec_threshold[np.argmax(gain)]))
    plt.show()

