import numpy as np
import pandas as pd
import itertools
import os
from tqdm import tqdm

from src import loading



COLS_USED = ['client_id','sales_net','quantity','year','quarter']
FINAL_COLS = ['sales_net','quantity']
NB_DATA = 63319315
NB_QUARTER_CHURNER = 3
NB_SKIP_BEG = 2


def generate_datecols(df):
    df.date_order = pd.to_datetime(df.date_order)
    df['year'] = df.date_order.dt.year
    df['quarter'] = df.date_order.dt.quarter
    

def generate_list():

    df = loading.load_data()
    generate_datecols(df)

    np.savetxt('./data/list_year.txt',df.year.unique())
    np.savetxt('./data/list_quarter.txt',df.quarter.unique())
    np.savetxt('./data/list_clients.txt',df.client_id.unique())

def generate_template_df():

    list_client = list(np.sort(np.loadtxt(loading.PATH_DATA + 'list_clients.txt',dtype=int)))
    list_year = list(np.sort(np.loadtxt(loading.PATH_DATA + 'list_year.txt',dtype=int)))
    list_quarter = list(np.sort(np.loadtxt(loading.PATH_DATA + '/list_quarter.txt',dtype=int)))

    index = pd.MultiIndex.from_tuples(list(itertools.product(list_client,list_year,list_quarter)),
                                  names = ['client_id','year','quarter'])

    return pd.DataFrame(index=index,columns=FINAL_COLS)


def generate_df(max_rows = 1e6):

    if os.path.exists('./data/final_df.csv'):
        return pd.read_csv('./data/final_df.csv')

    if not os.path.exists('./data/list_clients.txt'):
        generate_list()

    df = generate_template_df()
    n_rows = len(df)

    raw_df = generate_template_df()

    for steps in tqdm(range(int(NB_DATA/max_rows))):

        df_temp = loading.load_data(nrows=max_rows,skiprows=steps*max_rows)
        generate_datecols(df_temp)
        df_temp = df_temp[COLS_USED].groupby(['client_id','year','quarter']).sum()

        if not steps :
            df.update(df_temp)
        else :
            raw_temp = raw_df.copy()
            raw_temp.update(df_temp)

            df.fillna(0,inplace=True)
            raw_temp.fillna(0,inplace=True)

            df = df.add(raw_temp)

    df.to_csv('./data/final_df.csv')

    return df.reset_index()


def generate_training_data(df,return_all = False):

    list_client = list(np.sort(np.loadtxt(loading.PATH_DATA + 'list_clients.txt',dtype=int)))
    list_year = list(np.sort(np.loadtxt(loading.PATH_DATA + 'list_year.txt',dtype=int)))
    list_quarter = list(np.sort(np.loadtxt(loading.PATH_DATA + 'list_quarter.txt',dtype=int)))

    nb_clients = len(list_client)
    nb_ts = len(list_year)*len(list_quarter)
    data = []

    y = []
    X = df[FINAL_COLS].values

    print('Generating churn labels...')
    for i in tqdm(range(nb_clients)) :
        y.append(0) if np.sum(X[i*12:(i+1)*12,-1][-NB_QUARTER_CHURNER:]) else y.append(1)

    if not return_all :
        X = X.reshape((nb_clients,nb_ts,len(FINAL_COLS)))[:,NB_SKIP_BEG:-NB_QUARTER_CHURNER,:]
    else :
        X = X.reshape((nb_clients,nb_ts,len(FINAL_COLS)))[:,NB_SKIP_BEG:,:]


    print("X shape : {}".format(X.shape))
        
    return X,np.array(y)
