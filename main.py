
import streamlit as st
import Introduccion as i
import Regla_asociación as r
import Metricas as m
import Clustering as c
import Regresión_lógistica as rl
import Árbol_decisión as a

st.set_page_config(
    page_title="Inteligencia artificial",
    page_icon="random",
    layout="centered",
    initial_sidebar_state="expanded",
 )

Opcion=st.sidebar.selectbox("Opciones",("Inicio","Reglas de asociación","Metricas de distancia",
"Clustering", "Regresión lógistica", "Árbol de decisión"))
if Opcion == "Reglas de asociación":
    r.show()
elif Opcion == "Metricas de distancia":
    m.show()
elif Opcion == "Clustering":
    c.show()
elif Opcion == "Regresión lógistica":
    rl.show()
elif Opcion == "Árbol de decisión":
    a.show()
else:
    i.show()
   
   


