#programacion de un algoritmo EDA

import numpy as np
import matplotlib.pyplot as plt
import imageio

#funcion para generar rutas aleatorias
def rutaRandom(num_ciudades):
    return np.random.permutation(num_ciudades)

#funcion para calcular la distancia total de la ruta
def calcular_distancia_total(ruta, matriz_distancia):
    total_distancia = 0
    for i in range(len(ruta) -1 ):
        total_distancia += matriz_distancia[ruta[i],ruta[i+1]]
    total_distancia += matriz_distancia[ruta[-1],ruta[0]] # regresa a la ciudad de origen
    return total_distancia

#Calcular las probabilidades de transicion entre ciudades
def calcular_probabilidad_transicion(ruta_seleccionada, num_ciudades):
    probabilidades = np.zeros((num_ciudades,num_ciudades))
    for ruta in ruta_seleccionada:
        for i in range(len(ruta) - 1):
            probabilidades[ruta[i],ruta[i+1]] += 1
        probabilidades[ruta[-1],ruta[0]] += 1 #regresa a la ciudad de origen
    probabilidades /= len(ruta_seleccionada)
    return probabilidades

#funcion para generar una nueva ruta basada en las probabilidades de transicion
def generar_nueva_ruta(probabilidades, num_ciudades):
    nueva_ruta = [np.random.choice(num_ciudades)]
    while len(nueva_ruta) < num_ciudades:
        siguiente_ciudad_probabilidad = probabilidades[nueva_ruta[-1]]
        siguiente_ciudad = np.random.choice(num_ciudades, p=siguiente_ciudad_probabilidad/siguiente_ciudad_probabilidad.sum())
        if siguiente_ciudad not in nueva_ruta:
            nueva_ruta.append(siguiente_ciudad)
    return nueva_ruta

#parametros del problema
num_ciudades = 10
tam_poblacion = 100
num_generaciones = 10
tam_seleccion = 20

#generar una matriz de distancias aleatoria simetrica
matriz_distancias = np.random.rand(num_ciudades,num_ciudades)
matriz_distancias = (matriz_distancias + matriz_distancias.T)/2
np.fill_diagonal(matriz_distancias, 0)

#inicializacion de la poblacion
poblacion = [rutaRandom(num_ciudades) for _ in range(tam_poblacion)]

#almacenar la mejor solucion en cada generacion

mejores_soluciones = []
images = []

#EDA --> Proceso iterativo
for generacion in range(num_generaciones):
    print("Generacion : {}".format(generacion))
    #Evaluar la poblacion
    fitness = np.array([calcular_distancia_total(ruta, matriz_distancias) for ruta in poblacion])

    #Seleccion: Elergir las mejores soluciones
    indices_seleccionados =np.argsort(fitness)[:tam_seleccion]
    poblacion_seleccionada = [poblacion[i] for i in indices_seleccionados]

    #Modelado de distribucion: Calcular probabilidades de transicion
    probabilidades = calcular_probabilidad_transicion(poblacion_seleccionada, num_ciudades)

    #Muestreo: Generar nueva poblacion
    poblacion = [generar_nueva_ruta(probabilidades, num_ciudades) for _ in range(tam_seleccion)]

    #Almacenar la mejor solucion
    mejores_soluciones.append(np.min(fitness))

    #guardar la imagen de la generacion actual
    mejor_ruta_index = np.argmin(fitness)
    mejor_ruta = poblacion[mejor_ruta_index]
    fig, ax = plt.subplots()
    ciudades = np.arange(num_ciudades)
    ax.scatter(ciudades, np.zeros(num_ciudades), c="red")
    for i in range(num_ciudades):
        ax.text(ciudades[i], mejor_ruta[i+1], [0,0], 'b-')
    ax.plot([mejor_ruta[-1], mejor_ruta[0]], [0,0], 'b-')
    ax.set_title(f'Generación {generacion + 1}, Distancia: {mejores_soluciones[-1]:.2f}')
    plt.axis('off')

    # Guardar la figura como imagen
    filename = f'generation_{generacion + 1}.png'
    plt.savefig(filename)
    images.append(imageio.imread(filename))
    plt.close(fig)

# Crear el GIF
imageio.mimsave('tsp_evolution.gif', images, fps=2)

# Mostrar el GIF
from IPython.display import Image

Image(filename='tsp_evolution.gif')

# Mejor solución encontrada
best_route_index = np.argmin(fitness)
best_route = poblacion[best_route_index]
best_distance = calcular_distancia_total(best_route, matriz_distancias)
print(f"Mejor ruta: {best_route}, distancia total = {best_distance}")