import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 


from src import preprocessing, loading,metrics, visualization

def generate_model():

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv1D(filters = 32,kernel_size=3,activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Conv1D(filters = 32,kernel_size=3,activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(32,activation='relu',kernel_regularizer='l2'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(16,activation='relu',kernel_regularizer='l2'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(8,activation='relu',kernel_regularizer='l2'))
    model.add(tf.keras.layers.Dense(1,activation='sigmoid'))

    return model


def fit_churn_model(df,epochs=10,name='churn_model'):
    X,y = preprocessing.generate_training_data(df)
    X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y)

    try :
        model = loading.load_model(name=name)
        print('Loading trained model...')
    except :
        print('Generating new model...')
        model = generate_model()
    
    model.compile(loss='binary_crossentropy', optimizer='adam')
    history = model.fit(X_train,y_train,validation_data = (X_test,y_test),epochs=epochs)

    yh = (model.predict(X_test) > 0.5).astype(int)

    metrics.print_scores(y_test,yh)
    if epochs > 1 : visualization.plot_history(history,name)

    # Complete training on the all dataset
    model.fit(X_test,y_test,epochs=epochs)

    try :
        if not os.path.exists(loading.PATH_MODELS):
            os.mkdir(loading.PATH_MODELS)
        model.save(loading.PATH_MODELS + name)

    except :
        print("Problems saving the model...")

    return model



def find_threshold(model,df,n_thre = 5):

    X,y = preprocessing.generate_training_data(df,return_all=True,verbose=False)

    client_value = []
    future_value = []

    for ind in range(len(X)) :
        client_value.append(np.mean(X[ind,:,0],where=X[ind,:,0]>0))
        future_value.append(np.mean(X[ind,-preprocessing.NB_QUARTER_CHURNER:,0],where=X[ind,-preprocessing.NB_QUARTER_CHURNER:,0]>0))

    future_value=np.nan_to_num(future_value)
    client_value=np.nan_to_num(client_value)

    tp_gain = (1-metrics.PROMOTION_RATE)*client_value*metrics.PROBA_CONVERT_PROMOTION
    fp_cost = metrics.PROMOTION_RATE*future_value


    X = X[:,:-preprocessing.NB_QUARTER_CHURNER,:]

    vec_threshold = np.linspace(0.1,0.9,n_thre)
    gain = []
    bar = tqdm(vec_threshold)

    for thre in bar:
        bar.set_description('Thresh {}'.format(thre))
        yh = (model.predict(X)>thre).astype(int).squeeze()
        fp = np.where((yh != y)&(yh==1))[0]
        tp = np.where((yh == y)&(y==1))[0]
        gain.append((np.sum(tp_gain[tp])-np.sum(fp_cost[fp]))/len(X))

    print('Mean gain per client using our approach : {}'.format(np.max(gain)))
    visualization.plot_gain_threshold(vec_threshold,gain) 

    return vec_threshold[np.argmax(gain)]


def evaluate_model(df,name = 'churn_model',n_thre = 50):
    model = loading.load_model(name=name)
    X,y = preprocessing.generate_training_data(df,return_all=False,verbose=False)

    proba_churn = model.predict(X)
    print('AUC : {}'.format(metrics.roc_auc_score(y,proba_churn)))
    fpr,tpr = metrics.compute_rates(y,proba_churn)

    visualization.plot_ROC_curve(fpr,tpr,name)

    print('Choosing threshold that maximizes expected revenues...')
    thre = find_threshold(model,df,n_thre)
    print('Labelling churn if proba > {:.2f}'.format(thre))
    yh = (proba_churn > thre).astype(int)
    cf_matrix = metrics.compute_confusion_matrix(y,yh)
    visualization.plot_confusion_matrix(cf_matrix)
    metrics.print_scores(y,yh)


def print_info_model(name = 'churn_model'):
    try :
        model = loading.load_model(name=name)
        print(model.summary())
    except:
        raise NameError('Model not fitted yet.')
    


    