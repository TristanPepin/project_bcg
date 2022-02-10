import streamlit as st
import src
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import plotly.express as px

@st.cache
def load():
    df = pd.read_csv('./data/final_df.csv')
    df_infos = pd.read_csv('./data/data_client.csv')
    list_client = np.sort(np.loadtxt(src.loading.PATH_DATA + 'list_clients.txt',dtype=int))
    return df,df_infos,list_client



model = src.load_model()
df,df_infos,list_client = load()

user_input = st.selectbox("Client ID",list_client)
df_client = df[df.client_id == user_input]
ts = df_client.year + (df_client.quarter - 1)/4
df_client['ts'] = ts
df_infos_client = df_infos[df_infos.client_id == user_input]

X,y = src.generate_training_data(df_client,return_all=True,verbose=0,nb_clients=1)

proba_churn = model.predict(X[:,-8:,:])

if len(df_infos_client):
    st.text("Nomber of products per order (mean)  : {:.2f}".format(df_infos_client.mean_prod_per_com.values[0]))
    # Ajouter tt les autres features + écrire ça proprement

st.text("Proba of churn : {:.2f}".format(proba_churn[0][0]))
st.plotly_chart(px.line(df_client, x="ts", y="sales_net", title='Sales'))
st.plotly_chart(px.line(df_client, x="ts", y="quantity", title='Quantity'))








