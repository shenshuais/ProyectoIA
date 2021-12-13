import streamlit as st
import pandas as pd               
import numpy as np                
import matplotlib.pyplot as plt   
import seaborn as sns 
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import graphviz
from sklearn.tree import export_graphviz
from sklearn.tree import plot_tree 
from sklearn.tree import export_text
from sklearn.tree import DecisionTreeClassifier


def show():
    st.title("Árbol de decisión")
    file=st.file_uploader("Elege un archivo de entrada")
    if file != None:
        algortimo = st.radio("Seleccione un algortimo de árbol de decisión",
            ('Clasificación', 'Pronóstico'))
        archivo = pd.read_csv(file)
        st.dataframe(archivo)
        st.subheader("Descripción de los datos")
        st.dataframe(archivo.describe())

        st.subheader("Mapa de colores")
        fig = plt.figure(figsize=(14,7))
        MatrizCorr = np.triu(archivo.corr())
        sns.heatmap(archivo.corr(), cmap='RdBu_r', annot=True, mask=MatrizCorr)
        st.pyplot(fig)

        if algortimo == 'Pronóstico' :
            st.subheader("Pronóstico")
            #Selecciones de variables predictoras
            opciones = []
            for variable in archivo.columns:
                opciones.append(variable)
            variablespre = st.multiselect("Seleccionar las variables predictoras",opciones, default=None)
            variableapro = st.multiselect("Seleccionar la variable a pronósticar",opciones, default=None)
            
            #Asegurar que el tamaño de la lista es mayor a cero
            if len(variablespre) > 0 and len(variableapro) > 0:
                st.subheader("Tabla de las variables predictoras")
                X = np.array(archivo[variablespre])
                st.dataframe(X)

                st.subheader("Tabla de la variable a pronósticar")
                Y = np.array(archivo[variableapro])
                st.dataframe(Y)

                #División de datos
                st.text("División de datos, por defecto es 20% y semilla es 1234")
                division=st.number_input('Tamaño de prueba', value=0.2)
                semilla =st.number_input('Semilla',value=1234)
                X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, 
                                                                                    test_size = division, 
                                                                                    random_state = semilla,
                                                                                    shuffle = True)
                st.text("Datos de entrenamiento")
                st.dataframe(X_train)
                st.dataframe(Y_train)

                PronosticoAD = DecisionTreeRegressor()
                PronosticoAD.fit(X_train, Y_train)

                st.text("Generando el pronóstico")
                Y_Pronostico = PronosticoAD.predict(X_validation)
                st.dataframe(Y_Pronostico)

                st.text("Comparando con datos de validación")
                Valores = pd.DataFrame(Y_validation, Y_Pronostico)
                st.dataframe(Valores)

                st.text("Parametros del modelo")
                st.text('Criterio: '+str(PronosticoAD.criterion))
                st.text('Importancia variables: '+str(PronosticoAD.feature_importances_))
                st.write("MAE: %.4f" % mean_absolute_error(Y_validation, Y_Pronostico))
                st.write("MSE: %.4f" % mean_squared_error(Y_validation, Y_Pronostico))
                st.write("RMSE: %.4f" % mean_squared_error(Y_validation, Y_Pronostico, squared=False))   #True devuelve MSE, False devuelve RMSE
                st.write('Score: %.4f' % r2_score(Y_validation, Y_Pronostico))

                st.text("Importancia de las variables")
                Importancia = pd.DataFrame({'Variable': list(archivo[variablespre]),
                            'Importancia': PronosticoAD.feature_importances_}).sort_values('Importancia', ascending=False)
                st.dataframe(Importancia)

                elementos = export_graphviz(PronosticoAD,feature_names=variablespre)
                arbolP=graphviz.Source(elementos)
                arbolP.format = 'png'
                arbolP.render('Árbol_Decisión')
                
                st.subheader("Descargar el árbol de decisión")
                with open("Árbol_Decisión.png", "rb") as file:
                            btn = st.download_button(label="Descarga",data=file,file_name="Árbol_Decisión.png",mime="image/png")
           
                st.text("Otra representación")
                Reporte = export_text(PronosticoAD, feature_names = variablespre)
                st.text(Reporte)

                with st.container():
                    st.subheader("Nuevas clasificaciones")
                    paciente = []
                    d={}
                    for i in variablespre:
                        paciente.append(st.text_input(i, key=i, value= "0.0"))
                    
                    for i in range(0, len(variablespre), 1):
                        d[variablespre[i]]=float(paciente[i])
                    
                    pacienteID1 = pd.DataFrame(d,index=[0])
                    st.text("El resultado de pronónostico es: "+str(PronosticoAD.predict(pacienteID1)))


        else:
            st.subheader("Clasificación")
            opciones = []
            for variable in archivo.columns:
                opciones.append(variable)
            variablespre = st.multiselect("Seleccionar las variables predictoras",opciones, default=None)
            variableclase = st.multiselect("Seleccionar la variable clase",opciones, default=None)
                
                #Asegurar que el tamaño de la lista es mayor a cero
            if len(variablespre) > 0 and len(variableclase) > 0:
                    st.subheader("Tabla de las variables predictoras")
                    X = np.array(archivo[variablespre])
                    st.dataframe(X)

                    st.subheader("Tabla de la variable clase")
                    Y = np.array(archivo[variableclase])
                    st.dataframe(Y)

                    #División de datos
                    st.text("División de datos, por defecto es 20% y semilla es 0")
                    division=st.number_input('Tamaño de prueba', value=0.2)
                    semilla =st.number_input('Semilla', value=0)
                    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, 
                                                                                        test_size = division, 
                                                                                        random_state = semilla,
                                                                                        shuffle = True)
                    st.text("Datos de entrenamiento")
                    st.dataframe(X_train)
                    st.dataframe(Y_train)

                    ClasificacionAD = DecisionTreeClassifier()
                    ClasificacionAD.fit(X_train, Y_train)

                    st.text("Etiqueta las clasificaciones")
                    Y_Clasificacion = ClasificacionAD.predict(X_validation)
                    st.dataframe(Y_Clasificacion)

                    st.text("Comparado con datos de validación")
                    Valores = pd.DataFrame(Y_validation, Y_Clasificacion)
                    st.write(Valores)

                    st.text("El Score del modelo es: "+str(ClasificacionAD.score(X_validation, Y_validation)))

                    st.subheader("Validación del modelo")
                    st.subheader("Matriz de validación")
                    Y_Clasificacion = ClasificacionAD.predict(X_validation)
                    Matriz_Clasificacion = pd.crosstab(Y_validation.ravel(), 
                                                    Y_Clasificacion, 
                                                    rownames=['Reales'], 
                                                    colnames=['Clasificación']) 
                    st.write(Matriz_Clasificacion)
                    st.text("Reporte de la clasificación")
                    st.text('Criterio: '+str(ClasificacionAD.criterion))
                    print('Importancia variables: '+str(ClasificacionAD.feature_importances_))
                    st.text("Exactitud: "+str (ClasificacionAD.score(X_validation, Y_validation)))
                    st.text(classification_report(Y_validation, Y_Clasificacion))

                    st.text("Importancia de las variables")
                    Importancia = pd.DataFrame({'Variable': list(archivo[variablespre]),
                            'Importancia': ClasificacionAD.feature_importances_}).sort_values('Importancia', ascending=False)
                    st.dataframe(Importancia)

                    elementos = export_graphviz(ClasificacionAD,feature_names=variablespre)
                    arbolP=graphviz.Source(elementos)
                    arbolP.format = 'png'
                    arbolP.render('Árbol_Decisión')
                    
                    st.subheader("Descargar el árbol de decisión")
                    with open("Árbol_Decisión.png", "rb") as file:
                                btn = st.download_button(label="Descarga",data=file,file_name="Árbol_Decisión.png",mime="image/png")
            
                    st.text("Otra representación")
                    Reporte = export_text(ClasificacionAD, feature_names = variablespre)
                    st.text(Reporte)

                    with st.container():
                        st.subheader("Nuevas clasificaciones")
                        paciente = []
                        d={}
                        for i in variablespre:
                            paciente.append(st.text_input(i, key=i, value= "0.0"))
                        
                        for i in range(0, len(variablespre), 1):
                            d[variablespre[i]]=float(paciente[i])
                        
                        pacienteID1 = pd.DataFrame(d,index=[0])
                        st.text("El resultado de pronónostico es: "+str(ClasificacionAD.predict(pacienteID1)))
                
                


            
        
