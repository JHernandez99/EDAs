import numpy as np
import matplotlib.pyplot as plt
import imageio


# Función para generar una ruta aleatoria
def generate_random_route(num_cities):
    return np.random.permutation(num_cities)


# Función para calcular la distancia total de una ruta
def calculate_total_distance(route, distance_matrix):
    total_distance = 0
    for i in range(len(route) - 1):
        total_distance += distance_matrix[route[i], route[i + 1]]
    total_distance += distance_matrix[route[-1], route[0]]  # Regresar a la ciudad de origen
    return total_distance


# Función para calcular las probabilidades de transición entre ciudades
def calculate_transition_probabilities(selected_routes, num_cities):
    probabilities = np.zeros((num_cities, num_cities))
    for route in selected_routes:
        for i in range(len(route) - 1):
            probabilities[route[i], route[i + 1]] += 1
        probabilities[route[-1], route[0]] += 1  # Regresar a la ciudad de origen
    probabilities /= len(selected_routes)
    return probabilities


# Función para generar una nueva ruta basada en las probabilidades de transición
def generate_new_route(probabilities, num_cities):
    new_route = [np.random.choice(num_cities)]
    while len(new_route) < num_cities:
        current_city = new_route[-1]
        next_city_probabilities = probabilities[current_city]
        possible_next_cities = [i for i in range(num_cities) if i not in new_route]

        if len(possible_next_cities) == 0:
            break

        # Normalizar las probabilidades solo para las ciudades posibles
        normalized_probabilities = np.array(
            [next_city_probabilities[i] if i in possible_next_cities else 0 for i in range(num_cities)])
        total_prob = normalized_probabilities.sum()

        if total_prob == 0:
            # Si la suma de probabilidades es 0, elegir una ciudad aleatoria de las posibles
            next_city = np.random.choice(possible_next_cities)
        else:
            normalized_probabilities /= total_prob
            next_city = np.random.choice(num_cities, p=normalized_probabilities)

        new_route.append(next_city)
    return new_route


# Parámetros del problema
num_cities = 10
population_size = 100
num_generations = 50
selection_size = 20

# Generar posiciones aleatorias para las ciudades en una cuadrícula de 100x100
city_positions = np.random.randint(0, 100, size=(num_cities, 2))

# Generar una matriz de distancias basada en las posiciones de las ciudades
distance_matrix = np.zeros((num_cities, num_cities))
for i in range(num_cities):
    for j in range(num_cities):
        distance_matrix[i, j] = np.linalg.norm(city_positions[i] - city_positions[j])

# Inicialización de la población
population = [generate_random_route(num_cities) for _ in range(population_size)]

# Almacenar la mejor solución en cada generación
best_solutions = []
images = []

# EDA: Proceso iterativo
for generation in range(num_generations):
    # Evaluar la población
    fitness = np.array([calculate_total_distance(route, distance_matrix) for route in population])

    # Selección: Elegir las mejores soluciones
    selected_indices = np.argsort(fitness)[:selection_size]
    selected_population = [population[i] for i in selected_indices]

    # Modelado de distribución: Calcular probabilidades de transición
    probabilities = calculate_transition_probabilities(selected_population, num_cities)

    # Muestreo: Generar nueva población
    population = [generate_new_route(probabilities, num_cities) for _ in range(population_size)]

    # Almacenar la mejor solución
    best_solutions.append(np.min(fitness))

    # Guardar la imagen de la generación actual
    best_route_index = np.argmin(fitness)
    best_route = population[best_route_index]

    fig, ax = plt.subplots()
    ax.scatter(city_positions[:, 0], city_positions[:, 1], c='red')
    for i, pos in enumerate(city_positions):
        ax.text(pos[0], pos[1], str(i), fontsize=12, ha='right')

    for i in range(num_cities - 1):
        start_city = best_route[i]
        end_city = best_route[i + 1]
        ax.plot([city_positions[start_city, 0], city_positions[end_city, 0]],
                [city_positions[start_city, 1], city_positions[end_city, 1]], 'b-')
    ax.plot([city_positions[best_route[-1], 0], city_positions[best_route[0], 0]],
            [city_positions[best_route[-1], 1], city_positions[best_route[0], 1]], 'b-')

    ax.set_title(f'Generación {generation + 1}, Distancia: {best_solutions[-1]:.2f}')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    plt.axis('off')

    # Guardar la figura como imagen
    filename = f'generation_{generation + 1}.png'
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
best_route = population[best_route_index]
best_distance = calculate_total_distance(best_route, distance_matrix)
print(f"Mejor ruta: {best_route}, distancia total = {best_distance}")
