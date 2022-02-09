import pandas as pd
import tensorflow as tf

PATH_DATA = './data/'
FILE = 'transactions_dataset.csv'
COL_NAMES = ['date_order','date_invoice','product_id ','client_id','sales_net','quantity','order_channel','branch_id']
PATH_MODELS = './models/'

def load_data(path=PATH_DATA,file=FILE,delimiter=';',nrows=None,skiprows=None):
    if nrows is not None and skiprows is not None :
        return pd.read_csv(path +file,delimiter=delimiter,header=0,nrows=int(nrows),skiprows = int(skiprows),names=COL_NAMES)
    else :
        return pd.read_csv(path +file,delimiter=delimiter)

def load_model(path=PATH_MODELS,name='churn_model'):
    return tf.keras.models.load_model(path+name)