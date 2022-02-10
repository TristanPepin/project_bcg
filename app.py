import streamlit as st
import src
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

THRESHOLD = 0.65

st.set_page_config(layout="wide")

@st.cache
def load():
    df = pd.read_csv('./data/final_df.csv')
    df_infos = pd.read_csv('./data/data_client.csv')
    list_client = np.sort(np.loadtxt(src.loading.PATH_DATA + 'list_clients.txt',dtype=int))
    return df,df_infos,list_client


model = src.load_model()
df,df_infos,list_client = load()

with st.sidebar :
    st.title('Client informations')
    user_input = st.selectbox("Client ID",list_client)


df_client = df[df.client_id == user_input]
ts = df_client.year + (df_client.quarter - 1)/4
df_client['ts'] = ts
df_infos_client = df_infos[df_infos.client_id == user_input]

X,y = src.generate_training_data(df_client,return_all=True,verbose=0,nb_clients=1)
proba_churn = model.predict(X[:,-9:-1,:])

# filtering last data for plot, always null
df_client = df_client[df_client.ts < df_client.ts.max()]


c1, c2 = st.columns((5, 1))

if proba_churn > THRESHOLD:
    with c1 : 
        message = '<p style="color:Red; font-size: 42px;">Potential churn detected.</p>'
        st.markdown(message, unsafe_allow_html=True)
    with c2 :
        c2.image('./appdata/red-flag.png',width=70)
else:

    with c1 : 
        message = '<p style="color:Green; font-size: 42px;">Normal behaviour detected.</p>'
        st.markdown(message, unsafe_allow_html=True)
    with c2 :
        c2.image('./appdata/flag.png',width=70)

st.plotly_chart(px.line(df_client, x="ts", y="sales_net", title='Sales'))

with st.sidebar:

    message = '<p style="font-size: 21px;">Churn probability</p>'
    st.markdown(message, unsafe_allow_html=True)

    c1, c2 = st.columns((1, 4))

    with c1:
        st.image('./appdata/risk.png',use_column_width =True)

    with c2 :
        if proba_churn > THRESHOLD:
            message = '<p style="text-align: center;color:Red; font-size: 17px;">P = {:.2f} %.</p>'.format(proba_churn[0][0]*100)
        else :
            message = '<p style="text-align: center;color:Green; font-size: 17px;">P = {:.2f} %.</p>'.format(proba_churn[0][0]*100)
        st.markdown(message, unsafe_allow_html=True)


    if len(df_infos_client):
        message = '<p style="font-size: 21px;">Mean time between 2 orders</p>'
        st.markdown(message, unsafe_allow_html=True)
        c1, c2 = st.columns((1, 5))
        with c1:
            st.image('./appdata/fast-time.png',use_column_width =True)
        with c2:
            message = '<p style="text-align: center; font-size: 17px;"> {:.2f} days</p>'.format(df_infos_client.mean_time_difference.values[0])
            st.markdown(message, unsafe_allow_html=True)


        message = '<p style="font-size: 21px;">Mean number of products per order</p>'
        st.markdown(message, unsafe_allow_html=True)
        c1, c2 = st.columns((1, 5))
        with c1:
            st.image('./appdata/shopping-bag.png',use_column_width =True)
        with c2:
            message = '<p style="text-align: center; font-size: 17px;"> {:.2f} products</p>'.format(df_infos_client.mean_prod_per_com.values[0])
            st.markdown(message, unsafe_allow_html=True)



        message = '<p style="font-size: 21px;">Mean sales</p>'
        st.markdown(message, unsafe_allow_html=True)
        c1, c2 = st.columns((1, 5))
        with c1:
            st.image('./appdata/dollar-symbol.png',use_column_width =True)
        with c2:
            message = '<p style="text-align: center; font-size: 17px;"> {:.2f} $</p>'.format(df_infos_client.mean_sales.values[0])
            st.markdown(message, unsafe_allow_html=True)
