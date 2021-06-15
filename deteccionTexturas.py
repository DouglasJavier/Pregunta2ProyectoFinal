# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 17:37:32 2021

@author: douglas
"""

from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data
from skimage.color import label2rgb
from scipy.spatial import distance as dist
#importamos librerias
import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse

#Configuracion de nuestro descriptor LBP

radio = 3
numpuntos = 8 *radio
METHOD = 'uniform'
plt.gray()
#Definimos una funcion que va a recibir la imagen 


#########################################################################################

#Ahora vamos a clasificar varias texturas utilizando la ganacia de la informacion

# Definimos una funcion encargad de calcular la ganancia de la informacion
def ganancia_informacion(p,q):
    p = np.asarray(p)
    q = np.asarray(q)
    filtro = np.logical_and(p != 0, q != 0)
    return np.sum(p[filtro] * np.log2(p[filtro]/q[filtro]))
#Ahora creamos una funcion que va a calcular elLBP y el histograma
def calculo(refs, image):
    mejor_puntaje=0.75
    mejor_nombre= None
    lbp = local_binary_pattern(image, numpuntos, radio, 'uniform') #Calculamos el LBP uniforme
    intervalos = int (lbp.max() + 1)
    hist, _ = np.histogram(lbp, density = True, bins = intervalos, range = (0, intervalos)) #Creamos el histograma
    #bobo = cv2.imread("punto_bobo.jpg",0)
    #hist = cv2.calcHist([lbp], [0], None, [256], [0,256])
    for name, ref in refs.items():
        ref_hist, _=np.histogram(ref, density = True, bins = intervalos, range = (0, intervalos))
        #ref_hist = cv2.calcHist([ref], [0], None, [59], [0,59])
        puntaje = ganancia_informacion(hist, ref_hist) #Comparamos la referencia del histogrma con el histograma calculado
        #puntaje=cv2.compareHist(ref_hist, hist, cv2.HISTCMP_CORREL)
        #print("puntaje ")
        #print(puntaje)
    
        if puntaje < mejor_puntaje:
            mejor_puntaje = puntaje
            mejor_nombre = name
            
    return mejor_nombre
#Cargamos texturas de referencia
# Mostramos la textura 1
jersey = cv2.imread("punto_jersey.jpg",0) #Leemos la imagen
#plt.imshow(jersey) #Mostramos la imagen
#plt.show()

# Mostramos la textura 2
bobo = cv2.imread("punto_bobo.jpg",0) #Leemos la imagen
#plt.imshow(bobo) #Mostramos la imagen
#plt.show()

# Mostramos la textura 2
bobo1 = cv2.imread("punto_bobo2.jpg",0) #Leemos la imagen
#plt.imshow(bobo1) #Mostramos la imagen
#plt.show()

# Mostramos la textura 2
bobo2 = cv2.imread("punto_bobo3.jpg",0) #Leemos la imagen
#plt.imshow(bobo2) #Mostramos la imagen
#plt.show()

cuero = cv2.imread("cuero.jpg",0) #Leemos la imagen
#plt.imshow(cuero) #Mostramos la imagen
#plt.show()

toalla = cv2.imread("toalla.jpg",0) #Leemos la imagen
#plt.imshow(toalla) #Mostramos la imagen
#plt.show()

ladrillos = cv2.imread("ladrillos.jpg",0) #Leemos la imagen
#plt.imshow(bobo2) #Mostramos la imagen
#plt.show()

panal = cv2.imread("panal.jpg",0) #Leemos la imagen
#plt.imshow(bobo2) #Mostramos la imagen
#plt.show()


elastico1 = cv2.imread("punto_elastico1.jpg",0) #Leemos la imagen
#plt.imshow(elastico1) #Mostramos la imagen
#plt.show()


elastico2 = cv2.imread("punto_elastico2.jpg",0) #Leemos la imagen
#plt.imshow(elastico2) #Mostramos la imagen
#plt.show()



# Mostramos la imagen de prueba
img = cv2.imread("imagen_prueba4.jpg",0) #Leemos la imagen
plt.imshow(img) #Mostramos la imagen
plt.show()

cv2.waitKey(0)

def hist (ax, lbp):            #Ahora definimos una funcion para determinar el histograma
    n_conte = int (lbp.max()+1)  #Definimos el valor maximo de LBP para visualizarlo
    return ax.hist (lbp.ravel(), density = True, bins = n_conte, range=(0, n_conte), facecolor='0.5')


#Vamos a claisficarlas

refs = {'punto jersey' : local_binary_pattern(jersey, numpuntos, radio, METHOD),
       'punto bobo' : local_binary_pattern(bobo, numpuntos, radio, METHOD),
       'punto bobo' : local_binary_pattern(bobo1, numpuntos, radio, METHOD),
        'punto bobo' : local_binary_pattern(bobo2, numpuntos, radio, METHOD),
       'punto elastico' : local_binary_pattern(elastico1, numpuntos, radio, METHOD),
       'punto elastico' : local_binary_pattern(elastico2, numpuntos, radio, METHOD),
        #'cuero' : local_binary_pattern(cuero, numpuntos, radio, METHOD),
        #'ladrillos' : local_binary_pattern(ladrillos, numpuntos, radio, METHOD),
        #'panal' : local_binary_pattern(panal, numpuntos, radio, METHOD)
        
        }
print('Imagenes comparadas con LBP: ')
height, width = img.shape

#print('\nOriginal : Resultados: ', calculo(refs,fragmento))

texturasDetec=[]
for i in range(0,height,50):
    for j in range (0,width,50):
        if (i+50) < height and (j + 50) < width :
            fragmento=img[i:i+50 , j:j+50]
            #print("----------------------------------------------------------")
            #print("y =",i," x=",j)
            textura = calculo(refs,fragmento)
            #print('\nOriginal : Resultados: ', textura )
            if textura not in texturasDetec and textura != None:
                texturasDetec.append(textura)
            #print("----------------------------------------------------------")
print(texturasDetec)


#Plotemaos los histogramas

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6))= plt.subplots(nrows = 2, ncols = 3, figsize = (9,6)) #Configuracion de subplots

plt.gray()

#Empezamos con el subplot 1 y 4 es decir la primera columna
ax1.imshow(local_binary_pattern(bobo, numpuntos, radio, METHOD))
ax1.axis('off')
hist(ax4, local_binary_pattern(bobo, numpuntos, radio, METHOD))
ax4.set_ylabel('Porcentaje')

#Empezamos con el subplot 2 - 5 es decir la primera columna
ax2.imshow(local_binary_pattern(elastico1, numpuntos, radio, METHOD))
ax2.axis('off')
hist(ax5, refs['punto elastico'])
ax5.set_xlabel('Valores LBP Uniformes')

#Terminamos con la tercera columna
ax3.imshow(local_binary_pattern(jersey, numpuntos, radio, METHOD))
ax3.axis('off')
hist(ax6, refs['punto jersey'])
plt.show()
"""

#Empezamos con el subplot 1 y 4 es decir la primera columna
ax1.imshow(local_binary_pattern(ladrillos, numpuntos, radio, METHOD))
ax1.axis('off')
hist(ax4, local_binary_pattern(ladrillos, numpuntos, radio, METHOD))
ax4.set_ylabel('Porcentaje')

#Empezamos con el subplot 2 - 5 es decir la primera columna
ax2.imshow(local_binary_pattern(elastico1, numpuntos, radio, METHOD))
ax2.axis('off')
hist(ax5, refs['punto elastico'])
ax5.set_xlabel('Valores LBP Uniformes')

#Terminamos con la tercera columna
ax3.imshow(local_binary_pattern(panal, numpuntos, radio, METHOD))
ax3.axis('off')
hist(ax6, refs['panal'])
plt.show()

"""
