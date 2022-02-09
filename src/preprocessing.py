import numpy as np
import pandas as pd
import itertools
import os
from tqdm import tqdm

from src import loading



COLS_USED = ['client_id','sales_net','quantity','year','quarter']
FINAL_COLS = ['sales_net','quantity']
NB_DATA = 63319315
NB_QUARTER_CHURNER = 2
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
        df = pd.read_csv('./data/final_df.csv')
    else :

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


    # Delete clients that have already churned :
    churn_filter = df[(df.year == 2019) & (df.quarter<=NB_QUARTER_CHURNER)][['client_id','sales_net']].groupby('client_id').sum()
    to_drop = churn_filter[churn_filter.sales_net == 0].index
    df.set_index(['client_id']).drop(to_drop).reset_index()
    df.to_csv('./data/final_df.csv',index=False)

    list_client = list(np.sort(np.loadtxt(loading.PATH_DATA + 'list_clients.txt',dtype=int)))
    np.savetxt('./data/list_clients.txt',df.client_id.unique())

    return df.reset_index()


def generate_training_data(df,return_all = False,verbose=1,nb_clients = None):

    if nb_clients is None :
        list_client = list(np.sort(np.loadtxt(loading.PATH_DATA + 'list_clients.txt',dtype=int)))
        nb_clients = len(list_client)

    list_year = list(np.sort(np.loadtxt(loading.PATH_DATA + 'list_year.txt',dtype=int)))
    list_quarter = list(np.sort(np.loadtxt(loading.PATH_DATA + 'list_quarter.txt',dtype=int)))
    nb_ts = len(list_year)*len(list_quarter)
    data = []

    y = []
    X = df[FINAL_COLS].values

    if verbose :
        print('Generating churn labels...')
        iter = tqdm(range(nb_clients))
    else :
        iter = range(nb_clients)

    for i in iter :
        y.append(0) if np.sum(X[i*12:(i+1)*12,-1][-NB_QUARTER_CHURNER:]) else y.append(1)

    if not return_all :
        X = X.reshape((nb_clients,nb_ts,len(FINAL_COLS)))[:,NB_SKIP_BEG:-NB_QUARTER_CHURNER,:]
    else :
        X = X.reshape((nb_clients,nb_ts,len(FINAL_COLS)))[:,NB_SKIP_BEG:,:]


    if verbose : print("X shape : {}".format(X.shape))
        
    return X,np.array(y)


def generate_client_infos():

    if os.path.exists(loading.PATH_DATA+'data_client.csv'):
        return pd.read_csv(loading.PATH_DATA+'data_client.csv')
    df = loading.load_data()
    df.date_order = pd.to_datetime(df.date_order)
    df_client = df[['client_id','product_id','date_order']].groupby(['client_id','date_order']).count().reset_index()
    df_sales = df_client.reset_index()[['client_id','product_id']].groupby('client_id').agg(mean_prod_per_com = ('product_id','mean'),
                                                                             max_order_per_com = ('product_id','max'))

    time_difference = df_client.date_order - df_client[['client_id','date_order']].groupby('client_id').shift(1).date_order
    df_client['time_difference'] = time_difference

    df_temporal = df_client[['client_id','time_difference']].groupby('client_id').agg(mean_time_difference = ('time_difference','mean'),
                                                                                  max_time_difference = ('time_difference','max')).dropna().reset_index()
    df_temporal.max_time_difference = df_temporal.max_time_difference.dt.days
    df_temporal.mean_time_difference = df_temporal.mean_time_difference.dt.days

    df_add = df_sales.merge(df_temporal,on='client_id',how='left')

    df_client = df[['sales_net','quantity','client_id']].groupby('client_id').agg(mean_quantity = ('quantity','mean'),
                                                                              max_quantity = ('quantity','max'),
                                                                              mean_sales = ('sales_net','mean'),
                                                                              max_sales = ('sales_net','max'))

    df = df_client.reset_index().merge(df_add,on='client_id',how='left').set_index('client_id')
    df.to_csv(loading.PATH_DATA+'data_client.csv')
    return df