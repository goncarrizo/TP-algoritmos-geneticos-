# Algoritmo Genético Canónico para maximizar la función f(x) = (x/coef)^2
# Hacer un programa que utilice un Algoritmo Genético Canónico para buscar un máximo de la función:
# 
# f(x) = (x/coef)2 en el dominio [0 , 230 -1]
#
# donde coef = 230 -1
#
# teniendo en cuenta los siguientes datos:
#
# –Probabilidad de Crossover = 0,75
# –Probabilidad de Mutación = 0,05
# –Población Inicial: 10 individuos
# –Ciclos del programa: 20
# –Método de Selección: Ruleta
# –Método de Crossover: 1 Punto
# –Método de Mutación: invertida
#Opción A:
#El programa debe mostrar, finalmente, el Cromosoma correspondiente al valor máximo, el valor máximo, mínimo y promedio obtenido de cada población.
#También calcular desviación estándar del fitness por generación. Esto muestra visualmente si la población converge (desviación baja) o si sigue diversa (desviación alta). Agregar explicaciones de lo observado en las conclusiones.
#Medir el tiempo de cómputo
#Agrega una tabla con tiempo de ejecución promedio para cada variante (20, 100, 200 corridas) y cada método (ruleta, torneo, elitismo). Analizar el trade-off  (situación en la que se debe sacrificar un objetivo, característica o beneficio para alcanzar otro).entre calidad de solución y costo computacional.
#Mostrar la impresión de las tablas de mínimos, promedios y máximos para 20, 100 y 200 corridas.
#Deben presentarse las gráficas de los valores Máximos, Mínimos y Promedios de la función objetivo por cada generación luego de correr el algoritmo genético 20, 100 y 200 iteraciones (una gráfica por cada conjunto de iteraciones)
#Realizar comparaciones de las salidas corriendo el mismo programa en distintos ciclos de corridas y además realizar todos los cambios que considere oportunos en los parámetros de entrada de manera de enriquecer sus conclusiones.

import random 
import math
import time
import statistics

# Intentar importar librerías opcionales
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False 

# ==================== PARÁMETROS ====================
COEF = (2 ** 30) - 1  # Coeficiente
MAX_VALUE = COEF  # Máximo valor posible en el dominio
CHROMOSOME_LENGTH = 30  # Longitud del cromosoma binario

# Parámetros del AG
PROB_CROSSOVER = 0.75
PROB_MUTATION = 0.05
INITIAL_POPULATION = 10
GENERATIONS = 20  # Será variable

# ==================== FUNCIONES AUXILIARES ====================

def binary_to_decimal(binary_chromosome):
    """Convierte un cromosoma binario a su representación decimal."""
    return int(''.join(map(str, binary_chromosome)), 2)

def decimal_to_binary(decimal_value, length=CHROMOSOME_LENGTH):
    """Convierte un número decimal a cromosoma binario."""
    binary = format(decimal_value, f'0{length}b')
    return [int(bit) for bit in binary]

def fitness_function(x):
    """Calcula el fitness: f(x) = (x/coef)^2"""
    return (x / COEF) ** 2

def evaluate_population(population):
    """Evalúa la población y retorna los fitness values."""
    fitnesses = []
    for chromosome in population:
        x = binary_to_decimal(chromosome)
        f = fitness_function(x)
        fitnesses.append(f)
    return fitnesses

def create_initial_population(pop_size):
    """Crea una población inicial aleatoria."""
    population = []
    for _ in range(pop_size):
        chromosome = [random.randint(0, 1) for _ in range(CHROMOSOME_LENGTH)]
        population.append(chromosome)
    return population

def selection_roulette(population, fitnesses):
    """Selección por ruleta (selecciona dos padres)."""
    # Normalizar fitness para que sean positivos si es necesario
    min_fitness = min(fitnesses)
    adjusted_fitnesses = [f - min_fitness + 1 for f in fitnesses]  # +1 para evitar cero
    
    total_fitness = sum(adjusted_fitnesses)
    probabilities = [f / total_fitness for f in adjusted_fitnesses]
    
    # Función auxiliar para seleccionar por probabilidad
    def weighted_choice(items, weights):
        r = random.random()
        cumsum = 0
        for item, weight in zip(items, weights):
            cumsum += weight
            if r <= cumsum:
                return item
        return items[-1]
    
    # Seleccionar dos padres con reemplazo
    parent1 = weighted_choice(population, probabilities)
    parent2 = weighted_choice(population, probabilities)
    
    return parent1.copy(), parent2.copy()

def crossover_one_point(parent1, parent2):
    """Crossover de un punto."""
    if random.random() < PROB_CROSSOVER:
        point = random.randint(1, CHROMOSOME_LENGTH - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    else:
        return parent1.copy(), parent2.copy()

def mutation_inverted(chromosome):
    """Mutación invertida (invierte bits aleatoriamente)."""
    mutated = chromosome.copy()
    for i in range(CHROMOSOME_LENGTH):
        if random.random() < PROB_MUTATION:
            mutated[i] = 1 - mutated[i]  # Invertir el bit
    return mutated

def genetic_algorithm(num_generations, population_size=INITIAL_POPULATION):
    """Ejecuta el algoritmo genético canónico."""
    # Inicialización
    population = create_initial_population(population_size)
    
    # Variables para almacenar estadísticas
    stats_per_generation = {
        'generation': [],
        'max_fitness': [],
        'min_fitness': [],
        'avg_fitness': [],
        'std_fitness': [],
        'best_chromosome': [],
        'best_value': []
    }
    
    # Generaciones
    for gen in range(num_generations):
        # Evaluar población actual
        fitnesses = evaluate_population(population)
        
        # Calcular estadísticas
        max_fitness = max(fitnesses)
        min_fitness = min(fitnesses)
        avg_fitness = sum(fitnesses) / len(fitnesses)
        
        # Calcular desviación estándar manualmente
        if len(fitnesses) > 1:
            variance = sum((f - avg_fitness) ** 2 for f in fitnesses) / (len(fitnesses) - 1)
            std_fitness = math.sqrt(variance)
        else:
            std_fitness = 0
        
        # Guardar el mejor cromosoma de esta generación
        best_idx = fitnesses.index(max_fitness)
        best_chromosome = population[best_idx]
        best_value = binary_to_decimal(best_chromosome)
        
        # Almacenar estadísticas
        stats_per_generation['generation'].append(gen + 1)
        stats_per_generation['max_fitness'].append(max_fitness)
        stats_per_generation['min_fitness'].append(min_fitness)
        stats_per_generation['avg_fitness'].append(avg_fitness)
        stats_per_generation['std_fitness'].append(std_fitness)
        stats_per_generation['best_chromosome'].append(best_chromosome.copy())
        stats_per_generation['best_value'].append(best_value)
        
        # Crear nueva población
        new_population = []
        while len(new_population) < population_size:
            # Selección
            parent1, parent2 = selection_roulette(population, fitnesses)
            
            # Crossover
            child1, child2 = crossover_one_point(parent1, parent2)
            
            # Mutación
            child1 = mutation_inverted(child1)
            child2 = mutation_inverted(child2)
            
            new_population.append(child1)
            if len(new_population) < population_size:
                new_population.append(child2)
        
        population = new_population[:population_size]
    
    return population, stats_per_generation

def print_results(num_generations, stats):
    """Imprime los resultados del algoritmo genético."""
    print("\n" + "="*80)
    print(f"RESULTADOS - ALGORITMO GENÉTICO ({num_generations} generaciones)")
    print("="*80)
    
    # Mejor solución encontrada
    final_best_idx = stats['max_fitness'].index(max(stats['max_fitness']))
    final_best_chromosome = stats['best_chromosome'][final_best_idx]
    final_best_value = stats['best_value'][final_best_idx]
    final_best_fitness = stats['max_fitness'][final_best_idx]
    
    print(f"\nMejor cromosoma encontrado: {final_best_chromosome}")
    print(f"Mejor valor de x: {final_best_value}")
    print(f"Mejor fitness: {final_best_fitness:.6f}")
    
    # Tabla de estadísticas por generación
    print("\nTabla de estadísticas por generación:")
    print(f"{'Gen':<5} {'Máximo':<12} {'Mínimo':<12} {'Promedio':<12} {'Desv. Est.':<12}")
    print("-" * 55)
    
    for i in range(len(stats['generation'])):
        gen = stats['generation'][i]
        max_f = stats['max_fitness'][i]
        min_f = stats['min_fitness'][i]
        avg_f = stats['avg_fitness'][i]
        std_f = stats['std_fitness'][i]
        print(f"{gen:<5} {max_f:<12.6f} {min_f:<12.6f} {avg_f:<12.6f} {std_f:<12.6f}")
    
    return None

def plot_results(num_generations, stats):
    """Genera gráficos de los resultados."""
    if not MATPLOTLIB_AVAILABLE:
        print("\n⚠️  matplotlib no está disponible. Se omitirán los gráficos.")
        return
    
    generations = stats['generation']
    max_fitness = stats['max_fitness']
    min_fitness = stats['min_fitness']
    avg_fitness = stats['avg_fitness']
    std_fitness = stats['std_fitness']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Algoritmo Genético - {num_generations} Generaciones', fontsize=16)
    
    # Gráfico 1: Fitness máximo, mínimo y promedio
    axes[0, 0].plot(generations, max_fitness, 'g-', label='Máximo', linewidth=2)
    axes[0, 0].plot(generations, min_fitness, 'r-', label='Mínimo', linewidth=2)
    axes[0, 0].plot(generations, avg_fitness, 'b-', label='Promedio', linewidth=2)
    axes[0, 0].set_xlabel('Generación')
    axes[0, 0].set_ylabel('Fitness')
    axes[0, 0].set_title('Evolución del Fitness')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Gráfico 2: Desviación estándar
    axes[0, 1].plot(generations, std_fitness, 'purple', linewidth=2)
    axes[0, 1].set_xlabel('Generación')
    axes[0, 1].set_ylabel('Desviación Estándar')
    axes[0, 1].set_title('Desviación Estándar del Fitness (Convergencia)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Gráfico 3: Valor de x del mejor cromosoma por generación
    best_x_values = stats['best_value']
    axes[1, 0].plot(generations, best_x_values, 'orange', linewidth=2)
    axes[1, 0].set_xlabel('Generación')
    axes[1, 0].set_ylabel('Valor de x')
    axes[1, 0].set_title('Mejor Valor de x por Generación')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Gráfico 4: Rango de variación (Max - Min)
    range_fitness = [max_fitness[i] - min_fitness[i] for i in range(len(max_fitness))]
    axes[1, 1].plot(generations, range_fitness, 'brown', linewidth=2)
    axes[1, 1].set_xlabel('Generación')
    axes[1, 1].set_ylabel('Rango (Máx - Mín)')
    axes[1, 1].set_title('Rango de Variación del Fitness')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# ==================== PROGRAMA PRINCIPAL ====================

if __name__ == "__main__":
    print("="*80)
    print("ALGORITMO GENÉTICO CANÓNICO")
    print("Función: f(x) = (x/coef)^2, donde coef = 2^30 - 1")
    print("="*80)
    
    # Ejecutar el algoritmo para diferentes números de generaciones
    execution_times = {}
    all_results = {}
    
    num_gen_list = [20, 100, 200]
    
    for num_gen in num_gen_list:
        print(f"\n\n{'#'*80}")
        print(f"Ejecutando con {num_gen} generaciones...")
        print(f"{'#'*80}")
        
        start_time = time.time()
        final_population, stats = genetic_algorithm(num_gen)
        end_time = time.time()
        
        execution_time = end_time - start_time
        execution_times[num_gen] = execution_time
        all_results[num_gen] = stats
        
        # Mostrar resultados
        print_results(num_gen, stats)
        
        print(f"\nTiempo de ejecución: {execution_time:.4f} segundos")
        
        # Generar gráficos
        plot_results(num_gen, stats)
    
    # Tabla comparativa de tiempos de ejecución
    print("\n\n" + "="*80)
    print("COMPARATIVA DE TIEMPOS DE EJECUCIÓN")
    print("="*80)
    print(f"{'Generaciones':<20} {'Tiempo (segundos)':<20}")
    print("-" * 40)
    for num_gen in num_gen_list:
        print(f"{num_gen:<20} {execution_times[num_gen]:<20.4f}")
    
    # Análisis de convergencia
    print("\n\n" + "="*80)
    print("ANÁLISIS DE CONVERGENCIA")
    print("="*80)
    for num_gen in num_gen_list:
        final_std = all_results[num_gen]['std_fitness'][-1]
        initial_std = all_results[num_gen]['std_fitness'][0]
        print(f"\nGeneraciones: {num_gen}")
        print(f"  Desviación estándar inicial: {initial_std:.6f}")
        print(f"  Desviación estándar final: {final_std:.6f}")
        print(f"  Reducción: {((initial_std - final_std) / initial_std * 100):.2f}%")
        if final_std < initial_std * 0.1:
            print("  → Población CONVERGIÓ (baja desviación)")
        else:
            print("  → Población mantuvo DIVERSIDAD (desviación moderada/alta)")

