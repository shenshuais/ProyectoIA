import streamlit as st
import pandas as pd               
import numpy as np                
import matplotlib.pyplot as plt   
import seaborn as sns 
from sklearn import linear_model
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

def show():
    st.title("Regresión lógistica")
    file=st.file_uploader("Elege un archivo de entrada")
    if file != None:
        Hipoteca = pd.read_csv(file)
        st.dataframe(Hipoteca)
        st.subheader("Conteo")
        st.text(Hipoteca.groupby('comprar').size())

        st.subheader("Correlación de los datos")
        #st.pyplot(sns.pairplot(Hipoteca, hue='comprar'))

        st.subheader("Matriz de correlación")
        CorrHipoteca = Hipoteca.corr(method='pearson')
        st.dataframe(CorrHipoteca)

        st.subheader("Mapa de colores")
        fig = plt.figure(figsize=(14,7))
        MatrizCorr = np.triu(CorrHipoteca)
        sns.heatmap(CorrHipoteca, cmap='RdBu_r', annot=True, mask=MatrizCorr)
        st.pyplot(fig)

        #Selecciones de variables predictoras
        opciones = []
        for variable in Hipoteca.columns:
            opciones.append(variable)
        variablesPre = st.multiselect("Seleccionar las variables predictoras",opciones, default=None)
        variableclase = st.multiselect("Seleccionar la variable clase",opciones, default=None)
        #Asegurar que el tamaño de la lista es mayor a cero
        if len(variablesPre) > 0 and len(variableclase) > 0:

            st.subheader("Tabla de las variables predictoras")
            X = np.array(Hipoteca[variablesPre])
            st.dataframe(X)

            st.subheader("Tabla de la variable clase")
            Y = np.array(Hipoteca[variableclase])
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

            #Aplicación del algoritmo
            Clasificacion = linear_model.LogisticRegression()
            Clasificacion.fit(X_train, Y_train) 
            st.text("Predicciones probabilísticas")
            Probabilidad = Clasificacion.predict_proba(X_validation)
            st.dataframe(Probabilidad)
            st.text("Predición final")
            Predicciones = Clasificacion.predict(X_validation)
            st.dataframe(Predicciones)

            st.text("El score es "+str(Clasificacion.score(X_validation, Y_validation)))
            
            st.subheader("Matriz de validación")
            Y_Clasificacion = Clasificacion.predict(X_validation)
            Matriz_Clasificacion = pd.crosstab(Y_validation.ravel(), 
                                            Y_Clasificacion, 
                                            rownames=['Reales'], 
                                            colnames=['Clasificación']) 
            st.write(Matriz_Clasificacion)
            st.text("Reporte de la clasificación")
            st.text("Exactitud: "+str (Clasificacion.score(X_validation, Y_validation)))
            st.text(classification_report(Y_validation, Y_Clasificacion))

            st.text("Ecuación del modelo")
            st.text("Intercept:"+str (Clasificacion.intercept_))
            st.text("Coeficientes: \n"+str(Clasificacion.coef_))


            
            with st.container():
                st.subheader("Nuevas clasificaciones")
                paciente = []
                d={}
                for i in variablesPre:
                    paciente.append(st.number_input(i, key=i))

                for i in range(0, len(variablesPre), 1):
                    d[variablesPre[i]]=paciente[i]
                
                HipotecaID251 = pd.DataFrame(d,index=[0])

                st.text("El resultado del clasificación es: " +str(Clasificacion.predict(HipotecaID251)))
           


        



        