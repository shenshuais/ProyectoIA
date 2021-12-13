import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from kneed import KneeLocator




def show():
    st.title("Clustering")
    file=st.file_uploader("Elege un archivo de entrada")
    if file != None:
        algortimo = st.radio("Seleccione un algortimo de clustering",
            ('Jerarquico', 'Particional'))
        Hipoteca = pd.read_csv(file)
        st.dataframe(Hipoteca)
        st.subheader("Conteo")
        st.text(Hipoteca.groupby('comprar').size())

        st.subheader("Correlación de los datos")
        st.pyplot(sns.pairplot(Hipoteca, hue='comprar'))

        st.subheader("Matriz de correlación")
        CorrHipoteca = Hipoteca.corr(method='pearson')
        st.dataframe(CorrHipoteca)

        st.subheader("Mapa de colores")
        fig = plt.figure(figsize=(14,7))
        MatrizCorr = np.triu(CorrHipoteca)
        sns.heatmap(CorrHipoteca, cmap='RdBu_r', annot=True, mask=MatrizCorr)
        st.pyplot(fig)

        st.subheader("Seleccionar variables")
        opciones = []
        for variable in Hipoteca.columns:
            opciones.append(variable)
        options = st.multiselect("Seleccionar las variables",opciones, default=None)

        #Asegurar que el tamaño de la lista es mayor a cero
        if len(options) > 0 :
            st.subheader("Tabla de las variables seleccionadas")
            MatrizHipoteca = np.array(Hipoteca[options])
            st.dataframe(MatrizHipoteca)

            #Normalización del la matriz
            st.subheader("Tabla normalizada")
            estandarizar = StandardScaler()                               # Se instancia el objeto StandardScaler o MinMaxScaler 
            MEstandarizada = estandarizar.fit_transform(MatrizHipoteca)
            st.dataframe(MEstandarizada)

            #Aplicación del agoritmo
            if algortimo == 'Jerarquico':
                st.subheader("Jerarquico")
                y = st.slider('Linea de corte', 0.0, 10.0,step=0.1)
                figure = plt.figure(figsize=(10, 7))
                plt.title("Casos de "+ file.name)
                plt.xlabel(file.name)
                plt.ylabel('Distancia')
                Arbol = shc.dendrogram(shc.linkage(MEstandarizada, method='complete', metric='euclidean'))
                plt.axhline(y, color='orange', linestyle='--')
                st.pyplot(figure)

                ncluster = st.number_input('Cantidad de clusters',value=7, step=1)
                MJerarquico = AgglomerativeClustering(n_clusters=ncluster, linkage='complete', affinity='euclidean')
                MJerarquico.fit_predict(MEstandarizada)
                st.write("Se crea las etiquetas")
                st.text(MJerarquico.labels_)

                st.write("Añadir las etiquetas")
                Hipoteca = Hipoteca.drop(columns=['comprar'])
                Hipoteca['clusterH'] = MJerarquico.labels_
                st.dataframe(Hipoteca)

                st.write(Hipoteca.groupby(['clusterH'])['clusterH'].count())

                st.write("Obtener los centroides")
                CentroidesH = Hipoteca.groupby('clusterH').mean()
                st.write(CentroidesH)

                st.text_area('Intepretaciones:')


            else:
                st.subheader("Particional")

                #Aplicando el método de codo
                st.write("Encontrando el número de clusters adecuados")
                SSE = []
                for i in range(2, 12):
                    km = KMeans(n_clusters=i, random_state=0)
                    km.fit(MEstandarizada)
                    SSE.append(km.inertia_)

                figura = plt.figure(figsize=(10, 7))
                plt.plot(range(2, 12), SSE, marker='o')
                plt.xlabel('Cantidad de clusters *k*')
                plt.ylabel('SSE')
                plt.title('Elbow Method')
                st.pyplot(figura)

                kl = KneeLocator(range(2, 12), SSE, curve="convex", direction="decreasing")
                x = int(kl.elbow)
                st.set_option('deprecation.showPyplotGlobalUse', False)
                figure = plt.style.use('ggplot')
                kl.plot_knee()
                st.pyplot(figure)
                st.text("La cantidad de clusters adecuado es: " + str(x))

                #Aplicando el algortimo
                MParticional = KMeans(n_clusters=4, random_state=0).fit(MEstandarizada)
                MParticional.predict(MEstandarizada)
                st.write("Se crea las etiquetas")
                st.text(MParticional.labels_)

                st.write("Añadir las etiquetas")
                Hipoteca = Hipoteca.drop(columns=['comprar'])
                Hipoteca['clusterP'] = MParticional.labels_
                st.dataframe(Hipoteca)

                st.write(Hipoteca.groupby(['clusterP'])['clusterP'].count())
                
                st.write("Obtener los centroides")
                CentroidesP = Hipoteca.groupby('clusterP').mean()
                st.write(CentroidesP)

                st.text_area('Intepretaciones:')


