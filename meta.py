import random
import numpy as np
import matplotlib.pyplot as plt

# Génération des villes aléatoires
def generate_cities(n):
    return np.random.rand(n, 2) * 100  

# Calcul de la distance totale d'un trajet
def total_distance(route, cities):
    return sum(np.linalg.norm(cities[route[i]] - cities[route[i+1]]) for i in range(len(route)-1)) + \
           np.linalg.norm(cities[route[-1]] - cities[route[0]])  

# Génération de la population initiale
def create_population(size, n):
    return [random.sample(range(n), n) for _ in range(size)]

# Sélection des meilleurs trajets
def selection(population, cities):
    return sorted(population, key=lambda route: total_distance(route, cities))[:len(population)//2]

# Croisement (mélange de deux parents)
def crossover(parent1, parent2):
    cut = random.randint(1, len(parent1) - 2)
    return parent1[:cut] + [c for c in parent2 if c not in parent1[:cut]]

# Mutation (échange deux villes)
def mutate(route):
    i, j = random.sample(range(len(route)), 2)
    route[i], route[j] = route[j], route[i]
    return route

# Algorithme génétique principal
def genetic_algorithm(n_cities=10, pop_size=50, generations=100):
    cities = generate_cities(n_cities)
    population = create_population(pop_size, n_cities)
    best_distances = []

    for _ in range(generations):
        selected = selection(population, cities)
        next_gen = selected[:]

        while len(next_gen) < pop_size:
            p1, p2 = random.sample(selected, 2)
            child = mutate(crossover(p1, p2))
            next_gen.append(child)

        population = next_gen
        best_distances.append(total_distance(population[0], cities))

    # Affichage des résultats
    print("Meilleur chemin :", population[0])
    print("Distance optimale :", total_distance(population[0], cities))

    # Tracé du meilleur chemin
    best_route = population[0]
    plt.figure(figsize=(6, 6))
    plt.scatter(cities[:, 0], cities[:, 1], c='red')
    for i in range(len(best_route)):
        start, end = best_route[i], best_route[(i+1) % len(best_route)]
        plt.plot([cities[start, 0], cities[end, 0]], [cities[start, 1], cities[end, 1]], 'b-')
    plt.title("Meilleur Chemin Trouvé")
    plt.show()

    # Graphique de convergence
    plt.plot(best_distances)
    plt.xlabel("Générations")
    plt.ylabel("Distance")
    plt.title("Évolution de la Solution")
    plt.show()

# Exécution de l'algorithme
genetic_algorithm()
