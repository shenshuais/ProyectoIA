import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.spatial import distance
def show():
    st.title("Metricas de distancia")
    file=st.file_uploader("Elege un archivo de entrada")
    if file != None:
        Hipoteca = pd.read_csv(file)
        st.dataframe(Hipoteca)

        Opcion=st.sidebar.radio("Seleccionar la metrica",("--","Euclidiana","Chebyshev",
        "Cityblock","Minkowski"))
        if Opcion == "Euclidiana":

            st.write("Euclidiana")
            DstEuclidiana= cdist(Hipoteca.iloc[0:10], Hipoteca.iloc[0:10], metric='euclidean')
            MEuclideana = pd.DataFrame(DstEuclidiana)
            st.write(MEuclideana)

        elif Opcion == "Chebyshev":
            st.write("Chebyshev")
            Dstchebyshev= cdist(Hipoteca, Hipoteca, metric='chebyshev')
            Mchebyshev = pd.DataFrame(Dstchebyshev)
            st.write( Mchebyshev)
        elif Opcion == "Cityblock":
            st.write("Cityblock")
            Dstcityblock= cdist(Hipoteca, Hipoteca, metric='cityblock' )
            Mcityblock = pd.DataFrame(Dstcityblock)
            st.write(Mcityblock)
        elif Opcion == "Minkowski":
            st.write("Minkowski")
            pe = st.slider('Lambda', 1.0, 10.0,step=0.5)
            Dstminkowski= cdist(Hipoteca, Hipoteca, metric='minkowski', p=pe)
            Mminkowski = pd.DataFrame(Dstminkowski)
            st.write(Mminkowski)
        else:
            st.write("Ingrese una opcion de la barra lateral")