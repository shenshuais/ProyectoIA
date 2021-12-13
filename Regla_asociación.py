from PIL.Image import NONE
import streamlit as st
import pandas as pd                 
import numpy as np                  
import matplotlib.pyplot as plt     
from apyori import apriori

def show():
    st.title("Reglas de asociación")
    file=st.file_uploader("Elege un archivo de entrada")
    if file != None:
        DatosMovies = pd.read_csv(file, header=None)
        st.dataframe(DatosMovies)
        size = DatosMovies.shape
        st.subheader("El tamaño del archivo es")
        st.text(size)
        st.subheader("Lista de los datos")
        Transacciones = DatosMovies.values.reshape(-1).tolist() 
        ListaM = pd.DataFrame(Transacciones)
        ListaM['Frecuencia'] = 0
        ListaM = ListaM.groupby(by=[0], as_index=False).count().sort_values(by=['Frecuencia'], ascending=True) #Conteo by=[0] ordena por cero
        ListaM['Porcentaje'] = (ListaM['Frecuencia'] / ListaM['Frecuencia'].sum()) #Porcentaje
        ListaM = ListaM.rename(columns={0 : 'Item'})#aqui renambra la columna
        st.dataframe(ListaM)
        
        st.subheader("Grafica de los datos")
        figure = plt.figure(figsize=(16,20), dpi=300)
        plt.ylabel('Item')
        plt.xlabel('Frecuencia')
        plt.barh(ListaM['Item'], width=ListaM['Frecuencia'], color='blue')
        plt.show()
        st.pyplot(figure)

        MoviesLista = DatosMovies.stack().groupby(level=0).apply(list).tolist()
        st.subheader("Ingrese los datos")
        soporte = st.slider('Soporte', 0.0, 1.0,step=0.01)
        confianza = st.slider('Confianza', 0.0, 1.0,step=0.01)
        elevacion = st.slider('Elevación', 1.0, 10.0,step=0.01)
        if soporte > 0 and confianza > 0 and elevacion > 1:
            ReglasC1 = apriori(MoviesLista, 
            min_support=soporte, 
            min_confidence=confianza, 
            min_lift=elevacion)

            st.subheader("Resultado")
            ResultadosC1 = list(ReglasC1)
            st.write("El algoritmo encontro "+str(len(ResultadosC1))+" Reglas")
            text=""
            for item in ResultadosC1:
                #El primer índice de la lista
                Emparejar = item[0]
                items = [x for x in Emparejar]
                st.subheader("Regla: " + str(item[0]))

                #El segundo índice de la lista
                st.text("Soporte: " + str(item[1]))

                #El tercer índice de la lista
                st.text("Confianza: " + str(item[2][0][2]))
                st.text("Lift: " + str(item[2][0][3])) 

                text=st.text_area("Intepretacion:", key=item.index)


        else:
            st.write("Ingrese datos validos")
           


        



