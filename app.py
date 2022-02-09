import streamlit as st
import src
import numpy as np
import matplotlib.pyplot as plt

df = src.generate_df()
list_client = np.sort(np.loadtxt(src.loading.PATH_DATA + 'list_clients.txt',dtype=int))
model = src.load_model()

user_input = st.selectbox("Client ID",list_client)
df_client = df[df.client_id == user_input]

X,y = src.generate_training_data(df_client,return_all=True,verbose=0,nb_clients=1)

fig,ax = plt.subplots()
ts = df_client.year + (df_client.quarter - 1)/4
ax.plot(ts,df_client.sales_net,label='sales')
ax.plot(ts,df_client.quantity,label='quantity')
ax.legend()
ax.set_xlabel('Year')


proba_churn = model.predict(X[:,-7:,:])

st.text("Proba of churn : {}".format(proba_churn[0][0]))
st.pyplot(fig)



