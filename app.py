import streamlit as st
import src
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


@st.cache
def load():
    df = src.generate_df()
    df_infos = src.generate_client_infos()
    model = src.load_model()
    list_client = np.sort(np.loadtxt(src.loading.PATH_DATA + 'list_clients.txt',dtype=int))

    return df,df_infos,model,list_client


df,df_infos,model,list_client = load()


user_input = st.selectbox("Client ID",list_client)
df_client = df[df.client_id == user_input]
df_client['ts'] = df_client.year + (df_client.quarter - 1)/4
df_infos_client = df_infos[df_infos.client_id == user_input]

X,y = src.generate_training_data(df_client,return_all=True,verbose=0,nb_clients=1)

proba_churn = model.predict(X[:,-8:,:])

if len(df_infos_client):
    st.text("Nomber of products per order (mean)  : {:.2f}".format(df_infos_client.mean_prod_per_com))
    # Ajouter tt les autres features + écrire ça proprement

st.text("Proba of churn : {:.2f}".format(proba_churn[0][0]))
st.pyplot(px.line(df_client, x="ts", y="sales_net", title='Sales'))
st.pyplot(px.line(df_client, x="ts", y="quantity", title='Quantity'))



