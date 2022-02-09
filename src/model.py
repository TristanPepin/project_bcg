from tokenize import Name
import tensorflow as tf
from src import preprocessing, loading
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, roc_curve, roc_auc_score
import numpy as np
from tqdm import tqdm
import os

import matplotlib.pyplot as plt

NB_FEATURES = 2
NB_TS = 12 
DISCOUNT_FACTOR = 0.5
PROMOTION_RATE = 0.2
NB_SKIP_BEG = 2

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

    print("F1 score : {}".format(f1_score(y_test,yh)))
    print("Accuracy score : {}".format(accuracy_score(y_test,yh)))
    print("Recall score : {}".format(recall_score(y_test,yh)))
    print("Precision score : {}".format(precision_score(y_test,yh)))

    plt.figure()
    plt.plot(history.history['loss'],label='loss')
    plt.plot(history.history['val_loss'],label='val_loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Entropy')
    plt.show()


    # Complete training on the all dataset
    model.fit(X,y,epochs=5)

    if not os.path.exists(loading.PATH_MODELS):
        os.mkdir(loading.PATH_MODELS)
    model.save(loading.PATH_MODELS + name)

    if not os.path.exists(loading.PATH_MODELS + name + '/history.png'):
        plt.savefig(loading.PATH_MODELS + name + '/history.png')



    return model
    


def find_threshold(model,df,n_thre = 5):

    X,y = preprocessing.generate_training_data(df,return_all=True)
    promotion_cost = []
    future_value = []
    for ind in range(len(X)) :
        promotion_cost.append(np.max(X[ind,-preprocessing.NB_QUARTER_CHURNER:,0]))
        future_value.append(np.mean(X[ind,:,0],where=X[ind,:,0]>0))

    future_value=np.nan_to_num(future_value)
    promotion_cost = PROMOTION_RATE*np.array(promotion_cost)

    future_value = future_value/max(future_value)
    promotion_cost = promotion_cost/max(promotion_cost)

    X = X[:,:-preprocessing.NB_QUARTER_CHURNER,:]

    vec_threshold = np.linspace(0.1,0.9,n_thre)
    cost = []
    bar = tqdm(vec_threshold)

    for thre in bar:
        bar.set_description('Thresh {}'.format(thre))
        yh = (model.predict(X)>thre).astype(int)
        fp = np.where((yh != y)&(yh==1))[0]
        fn = np.where((yh != y)&(yh==0))[0]
        cost.append(np.sum(future_value[fn])+np.sum(promotion_cost[fp]))
         
    plt.plot(vec_threshold,cost)
    plt.xlabel('Threshold')
    plt.ylabel('Cost')
    plt.title('Best threshold : {}'.format(vec_threshold[np.argmin(cost)]))
    plt.show()
   
    return vec_threshold[np.argmin(cost)]



def find_threshold_f1(model,df,n_thre = 50):

    X,y = preprocessing.generate_training_data(df,return_all=False)

    vec_threshold = np.linspace(0.1,0.9,n_thre)
    cost = []
    bar = tqdm(vec_threshold)

    for thre in bar:
        bar.set_description('Thresh {}'.format(thre))
        yh = (model.predict(X)>thre).astype(int)
        cost.append(f1_score(y,yh))
         
    plt.plot(vec_threshold,cost)
    plt.xlabel('Threshold')
    plt.ylabel('Cost')
    plt.title('Best threshold : {}'.format(vec_threshold[np.argmax(cost)]))
    plt.show()
   
    return vec_threshold[np.argmax(cost)]


def evaluate_model(df,name = 'churn_model'):
    try :
        model = loading.load_model(name=name)
        X,y = preprocessing.generate_training_data(df,return_all=False)

        proba_churn = model.predict(X)

        fpr,tpr,_ = roc_curve(y,proba_churn)

        plt.figure()
        plt.plot(fpr,tpr)
        plt.xlabel('FRP')
        plt.ylabel('TRP')
        plt.title('ROC CURVE')
        plt.savefig(loading.PATH_MODELS + name + 'roc_curve')
        plt.show();

        print('AUC : {}'.format(roc_auc_score(y,proba_churn)))

        print('Choosing threshold that maximizes f1 score...')
        thre = find_threshold_f1(model,df,n_thre = 10)
        yh = (proba_churn > thre).astype(int)
        
        print("F1 score : {}".format(f1_score(y,yh)))
        print("Accuracy score : {}".format(accuracy_score(y,yh)))
        print("Recall score : {}".format(recall_score(y,yh)))
        print("Precision score : {}".format(precision_score(y,yh)))

    except :
        model = fit_churn_model(df,fit_churn_model)


def print_info_model(name = 'churn_model'):
    try :
        model = loading.load_model(name=name)
        print(model.summary())
    except:
        raise NameError('Model not fitted yet.')
    


    