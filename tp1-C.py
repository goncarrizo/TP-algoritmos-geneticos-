#Opción C: aplicar lo enunciado en la opción A pero con método de Selección con Elitismo
# Algoritmo Genético Canónico para maximizar la función f(x) = (x/coef)^2
# Variante con Elitismo - preservación de los mejores individuos

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
GENERATIONS = 100  # Ejecutar con 100 iteraciones como se solicita
ELITE_SIZE = 2  # Número de individuos élite a preservar

# ==================== FUNCIONES AUXILIARES ====================

def binario_a_decimal(cromosoma_binario):
    """Convierte un cromosoma binario a su representación decimal."""
    return int(''.join(map(str, cromosoma_binario)), 2)

def decimal_a_binario(valor_decimal, longitud=CHROMOSOME_LENGTH):
    """Convierte un número decimal a cromosoma binario."""
    binario = format(valor_decimal, f'0{longitud}b')
    return [int(bit) for bit in binario]

def funcion_fitness(x):
    """Calcula el fitness: f(x) = (x/coef)^2"""
    return (x / COEF) ** 2

def evaluar_poblacion(poblacion):
    """Evalúa la población y retorna los valores de fitness."""
    aptitudes = []
    for cromosoma in poblacion:
        x = binario_a_decimal(cromosoma)
        f = funcion_fitness(x)
        aptitudes.append(f)
    return aptitudes

def crear_poblacion_inicial(tamaño_poblacion):
    """Crea una población inicial aleatoria."""
    poblacion = []
    for _ in range(tamaño_poblacion):
        cromosoma = [random.randint(0, 1) for _ in range(CHROMOSOME_LENGTH)]
        poblacion.append(cromosoma)
    return poblacion

def seleccion_ruleta(poblacion, aptitudes):
    """Selección por ruleta (selecciona dos padres)."""
    # Normalizar fitness para que sean positivos si es necesario
    minimo_aptitud = min(aptitudes)
    aptitudes_ajustadas = [f - minimo_aptitud + 1 for f in aptitudes]  # +1 para evitar cero
    
    aptitud_total = sum(aptitudes_ajustadas)
    probabilidades = [f / aptitud_total for f in aptitudes_ajustadas]
    
    # Función auxiliar para seleccionar por probabilidad
    def opcion_ponderada(items, pesos):
        r = random.random()
        suma_acumulada = 0
        for item, peso in zip(items, pesos):
            suma_acumulada += peso
            if r <= suma_acumulada:
                return item
        return items[-1]
    
    # Seleccionar dos padres con reemplazo
    padre1 = opcion_ponderada(poblacion, probabilidades)
    padre2 = opcion_ponderada(poblacion, probabilidades)
    
    return padre1.copy(), padre2.copy()

def obtener_individuos_elite(poblacion, aptitudes, tamaño_elite):
    """
    Obtiene los mejores individuos (élite) de la población actual.
    
    Parámetros:
    - poblacion: Lista de cromosomas
    - aptitudes: Lista de valores de fitness correspondientes
    - tamaño_elite: Número de individuos a seleccionar
    
    Retorna: Lista con los cromosomas élite
    """
    # Crear pares (índice, aptitud)
    aptitud_indexada = [(i, f) for i, f in enumerate(aptitudes)]
    
    # Ordenar por aptitud en forma descendente
    aptitud_indexada.sort(key=lambda x: x[1], reverse=True)
    
    # Extraer los indices de los mejores
    indices_elite = [idx for idx, _ in aptitud_indexada[:tamaño_elite]]
    
    # Retornar los cromosomas élite
    cromosomas_elite = [poblacion[i].copy() for i in indices_elite]
    
    return cromosomas_elite

def cruce_un_punto(padre1, padre2):
    """Cruce de un punto."""
    if random.random() < PROB_CROSSOVER:
        punto = random.randint(1, CHROMOSOME_LENGTH - 1)
        hijo1 = padre1[:punto] + padre2[punto:]
        hijo2 = padre2[:punto] + padre1[punto:]
        return hijo1, hijo2
    else:
        return padre1.copy(), padre2.copy()

def mutacion_invertida(cromosoma):
    """Mutación invertida (invierte bits aleatoriamente)."""
    mutado = cromosoma.copy()
    for i in range(CHROMOSOME_LENGTH):
        if random.random() < PROB_MUTATION:
            mutado[i] = 1 - mutado[i]  # Invertir el bit
    return mutado

def algoritmo_genetico_con_elitismo(num_generaciones, tamaño_poblacion=INITIAL_POPULATION, tamaño_elite=ELITE_SIZE):
    """
    Ejecuta el algoritmo genético con ELITISMO.
    
    El elitismo es un método que mejora la convergencia del algoritmo genético
    preservando los mejores individuos de cada generación.
    
    Proceso:
    1. Se evalúa la población actual
    2. Se seleccionan los mejores 'tamaño_elite' individuos (élite)
    3. Se crea la nueva población mediante selección y operadores genéticos
    4. Se reemplazan los peores individuos con la élite preservada
    
    Esto garantiza que la mejor solución encontrada nunca se pierda.
    """
    # Inicialización
    poblacion = crear_poblacion_inicial(tamaño_poblacion)
    
    # Variables para almacenar estadísticas
    estadisticas_por_generacion = {
        'generacion': [],
        'maxima_aptitud': [],
        'minima_aptitud': [],
        'promedio_aptitud': [],
        'desviacion_aptitud': [],
        'mejor_cromosoma': [],
        'mejor_valor': []
    }
    
    # Generaciones
    for gen in range(num_generaciones):
        # Evaluar población actual
        aptitudes = evaluar_poblacion(poblacion)
        
        # PRESERVAR LA ÉLITE: Obtener los mejores individuos
        cromosomas_elite = obtener_individuos_elite(poblacion, aptitudes, tamaño_elite)
        
        # Calcular estadísticas
        maxima_aptitud = max(aptitudes)
        minima_aptitud = min(aptitudes)
        promedio_aptitud = sum(aptitudes) / len(aptitudes)
        
        # Calcular desviación estándar manualmente
        if len(aptitudes) > 1:
            varianza = sum((f - promedio_aptitud) ** 2 for f in aptitudes) / (len(aptitudes) - 1)
            desviacion_aptitud = math.sqrt(varianza)
        else:
            desviacion_aptitud = 0
        
        # Guardar el mejor cromosoma de esta generación
        indice_mejor = aptitudes.index(maxima_aptitud)
        mejor_cromosoma = poblacion[indice_mejor]
        mejor_valor = binario_a_decimal(mejor_cromosoma)
        
        # Almacenar estadísticas
        estadisticas_por_generacion['generacion'].append(gen + 1)
        estadisticas_por_generacion['maxima_aptitud'].append(maxima_aptitud)
        estadisticas_por_generacion['minima_aptitud'].append(minima_aptitud)
        estadisticas_por_generacion['promedio_aptitud'].append(promedio_aptitud)
        estadisticas_por_generacion['desviacion_aptitud'].append(desviacion_aptitud)
        estadisticas_por_generacion['mejor_cromosoma'].append(mejor_cromosoma.copy())
        estadisticas_por_generacion['mejor_valor'].append(mejor_valor)
        
        # Crear nueva población
        nueva_poblacion = []
        
        # Agregar la élite directamente (sin modificación)
        for cromosoma_elite in cromosomas_elite:
            nueva_poblacion.append(cromosoma_elite)
        
        # Llenar el resto de la población con descendientes
        while len(nueva_poblacion) < tamaño_poblacion:
            # Selección por ruleta
            padre1, padre2 = seleccion_ruleta(poblacion, aptitudes)
            
            # Cruce
            hijo1, hijo2 = cruce_un_punto(padre1, padre2)
            
            # Mutación
            hijo1 = mutacion_invertida(hijo1)
            hijo2 = mutacion_invertida(hijo2)
            
            nueva_poblacion.append(hijo1)
            if len(nueva_poblacion) < tamaño_poblacion:
                nueva_poblacion.append(hijo2)
        
        poblacion = nueva_poblacion[:tamaño_poblacion]
    
    return poblacion, estadisticas_por_generacion

def imprimir_resultados(num_generaciones, estadisticas):
    """Imprime los resultados del algoritmo genético con elitismo."""
    print("\n" + "="*80)
    print(f"RESULTADOS - ALGORITMO GENÉTICO CON ELITISMO ({num_generaciones} generaciones)")
    print("="*80)
    
    # Mejor solución encontrada
    indice_mejor_final = estadisticas['maxima_aptitud'].index(max(estadisticas['maxima_aptitud']))
    mejor_cromosoma_final = estadisticas['mejor_cromosoma'][indice_mejor_final]
    mejor_valor_final = estadisticas['mejor_valor'][indice_mejor_final]
    mejor_aptitud_final = estadisticas['maxima_aptitud'][indice_mejor_final]
    
    print(f"\nMejor cromosoma encontrado: {mejor_cromosoma_final}")
    print(f"Mejor valor de x: {mejor_valor_final}")
    print(f"Mejor aptitud: {mejor_aptitud_final:.6f}")
    print(f"Generación en que se encontró: {indice_mejor_final + 1}")
    
    # Tabla de estadísticas por generación
    print("\nTabla de estadísticas por generación:")
    print(f"{'Gen':<5} {'Máximo':<12} {'Mínimo':<12} {'Promedio':<12} {'Desv. Est.':<12}")
    print("-" * 55)
    
    # Mostrar primeras 10 generaciones y últimas 10
    total_generaciones = len(estadisticas['generacion'])
    
    if total_generaciones <= 20:
        # Si son menos de 20 generaciones, mostrar todas
        for i in range(len(estadisticas['generacion'])):
            gen = estadisticas['generacion'][i]
            max_a = estadisticas['maxima_aptitud'][i]
            min_a = estadisticas['minima_aptitud'][i]
            prom_a = estadisticas['promedio_aptitud'][i]
            desv_a = estadisticas['desviacion_aptitud'][i]
            print(f"{gen:<5} {max_a:<12.6f} {min_a:<12.6f} {prom_a:<12.6f} {desv_a:<12.6f}")
    else:
        # Mostrar primeras 10
        for i in range(10):
            gen = estadisticas['generacion'][i]
            max_a = estadisticas['maxima_aptitud'][i]
            min_a = estadisticas['minima_aptitud'][i]
            prom_a = estadisticas['promedio_aptitud'][i]
            desv_a = estadisticas['desviacion_aptitud'][i]
            print(f"{gen:<5} {max_a:<12.6f} {min_a:<12.6f} {prom_a:<12.6f} {desv_a:<12.6f}")
        
        print("...")
        
        # Mostrar últimas 10
        for i in range(total_generaciones - 10, total_generaciones):
            gen = estadisticas['generacion'][i]
            max_a = estadisticas['maxima_aptitud'][i]
            min_a = estadisticas['minima_aptitud'][i]
            prom_a = estadisticas['promedio_aptitud'][i]
            desv_a = estadisticas['desviacion_aptitud'][i]
            print(f"{gen:<5} {max_a:<12.6f} {min_a:<12.6f} {prom_a:<12.6f} {desv_a:<12.6f}")
    
    return None

def graficar_resultados(num_generaciones, estadisticas):
    """Genera gráficos de los resultados para análisis posterior."""
    if not MATPLOTLIB_AVAILABLE:
        print("\n⚠️  matplotlib no está disponible. Se omitirán los gráficos.")
        return
    
    generaciones = estadisticas['generacion']
    maxima_aptitud = estadisticas['maxima_aptitud']
    minima_aptitud = estadisticas['minima_aptitud']
    promedio_aptitud = estadisticas['promedio_aptitud']
    desviacion_aptitud = estadisticas['desviacion_aptitud']
    
    fig, ejes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Algoritmo Genético con Elitismo - {num_generaciones} Generaciones', fontsize=16)
    
    # Gráfico 1: Aptitud máximo, mínimo y promedio
    ejes[0, 0].plot(generaciones, maxima_aptitud, 'g-', label='Máximo', linewidth=2, marker='o', markersize=3)
    ejes[0, 0].plot(generaciones, minima_aptitud, 'r-', label='Mínimo', linewidth=2, marker='s', markersize=3)
    ejes[0, 0].plot(generaciones, promedio_aptitud, 'b-', label='Promedio', linewidth=2, marker='^', markersize=3)
    ejes[0, 0].set_xlabel('Generación')
    ejes[0, 0].set_ylabel('Aptitud')
    ejes[0, 0].set_title('Evolución de la Aptitud')
    ejes[0, 0].legend()
    ejes[0, 0].grid(True, alpha=0.3)
    
    # Gráfico 2: Desviación estándar (Convergencia)
    ejes[0, 1].plot(generaciones, desviacion_aptitud, 'purple', linewidth=2, marker='o', markersize=3)
    ejes[0, 1].fill_between(generaciones, desviacion_aptitud, alpha=0.3, color='purple')
    ejes[0, 1].set_xlabel('Generación')
    ejes[0, 1].set_ylabel('Desviación Estándar')
    ejes[0, 1].set_title('Desviación Estándar de la Aptitud (Convergencia)')
    ejes[0, 1].grid(True, alpha=0.3)
    
    # Gráfico 3: Valor de x del mejor cromosoma por generación
    mejores_valores_x = estadisticas['mejor_valor']
    ejes[1, 0].plot(generaciones, mejores_valores_x, 'orange', linewidth=2, marker='D', markersize=2)
    ejes[1, 0].set_xlabel('Generación')
    ejes[1, 0].set_ylabel('Valor de x')
    ejes[1, 0].set_title('Mejor Valor de x por Generación (Efecto del Elitismo)')
    ejes[1, 0].grid(True, alpha=0.3)
    
    # Gráfico 4: Rango de variación (Máx - Mín)
    rango_aptitud = [maxima_aptitud[i] - minima_aptitud[i] for i in range(len(maxima_aptitud))]
    ejes[1, 1].plot(generaciones, rango_aptitud, 'brown', linewidth=2, marker='*', markersize=5)
    ejes[1, 1].fill_between(generaciones, rango_aptitud, alpha=0.3, color='brown')
    ejes[1, 1].set_xlabel('Generación')
    ejes[1, 1].set_ylabel('Rango (Máx - Mín)')
    ejes[1, 1].set_title('Rango de Variación de la Aptitud')
    ejes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def graficar_analisis_adicional(estadisticas):
    """Genera gráficos adicionales para análisis más profundo."""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    generaciones = estadisticas['generacion']
    maxima_aptitud = estadisticas['maxima_aptitud']
    promedio_aptitud = estadisticas['promedio_aptitud']
    
    fig, ejes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Análisis Adicional del Elitismo', fontsize=14)
    
    # Gráfico 1: Mejora iterativa (diferencia con generación anterior)
    mejoras = [0]  # Primera generación no tiene mejora
    for i in range(1, len(maxima_aptitud)):
        mejora = maxima_aptitud[i] - maxima_aptitud[i-1]
        mejoras.append(mejora)
    
    ejes[0].bar(generaciones, mejoras, color='skyblue', edgecolor='navy')
    ejes[0].axhline(y=0, color='red', linestyle='--', linewidth=1)
    ejes[0].set_xlabel('Generación')
    ejes[0].set_ylabel('Mejora de Aptitud')
    ejes[0].set_title('Mejora Iterativa de la Mejor Aptitud')
    ejes[0].grid(True, alpha=0.3, axis='y')
    
    # Gráfico 2: Brecha entre máximo y promedio
    brecha = [maxima_aptitud[i] - promedio_aptitud[i] for i in range(len(maxima_aptitud))]
    ejes[1].plot(generaciones, brecha, 'green', linewidth=2, marker='o', markersize=3)
    ejes[1].fill_between(generaciones, brecha, alpha=0.3, color='green')
    ejes[1].set_xlabel('Generación')
    ejes[1].set_ylabel('Brecha (Máx - Promedio)')
    ejes[1].set_title('Brecha entre el Mejor y el Promedio de la Población')
    ejes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# ==================== PROGRAMA PRINCIPAL ====================

if __name__ == "__main__":
    print("="*80)
    print("ALGORITMO GENÉTICO CANÓNICO - CON ELITISMO")
    print("Función: f(x) = (x/coef)^2, donde coef = 2^30 - 1")
    print(f"Tamaño de élite: {ELITE_SIZE} individuos")
    print("="*80)
    
    print(f"\n\n{'#'*80}")
    print(f"Ejecutando con {GENERATIONS} generaciones (como se solicita)...")
    print(f"{'#'*80}")
    
    tiempo_inicio = time.time()
    poblacion_final, estadisticas = algoritmo_genetico_con_elitismo(GENERATIONS)
    tiempo_fin = time.time()
    
    tiempo_ejecucion = tiempo_fin - tiempo_inicio
    
    # Mostrar resultados
    imprimir_resultados(GENERATIONS, estadisticas)
    
    print(f"\nTiempo de ejecución: {tiempo_ejecucion:.4f} segundos")
    
    # Generar gráficos principales
    print("\nGenerando gráficos principales...")
    graficar_resultados(GENERATIONS, estadisticas)
    
    # Generar gráficos adicionales de análisis
    print("Generando gráficos de análisis adicional...")
    graficar_analisis_adicional(estadisticas)
    
    # Análisis de convergencia
    print("\n\n" + "="*80)
    print("ANÁLISIS DE CONVERGENCIA Y EFECTO DEL ELITISMO")
    print("="*80)
    
    desviacion_final = estadisticas['desviacion_aptitud'][-1]
    desviacion_inicial = estadisticas['desviacion_aptitud'][0]
    
    print(f"\nDesviación estándar inicial: {desviacion_inicial:.6f}")
    print(f"Desviación estándar final: {desviacion_final:.6f}")
    print(f"Reducción: {((desviacion_inicial - desviacion_final) / desviacion_inicial * 100):.2f}%")
    
    if desviacion_final < desviacion_inicial * 0.1:
        print("→ Población CONVERGIÓ FUERTEMENTE (muy baja desviación)")
    elif desviacion_final < desviacion_inicial * 0.5:
        print("→ Población CONVERGIÓ (desviación moderada)")
    else:
        print("→ Población mantuvo DIVERSIDAD (desviación alta)")
    
    # Análisis de mejora del mejor fitness
    print("\n" + "="*80)
    print("ANÁLISIS DE MEJORA DE LA MEJOR APTITUD")
    print("="*80)
    
    aptitud_inicial = estadisticas['maxima_aptitud'][0]
    aptitud_final = estadisticas['maxima_aptitud'][-1]
    mejora_total = aptitud_final - aptitud_inicial
    mejora_porcentual = (mejora_total / aptitud_inicial * 100) if aptitud_inicial > 0 else 0
    
    print(f"\nMejor aptitud inicial: {aptitud_inicial:.6f}")
    print(f"Mejor aptitud final: {aptitud_final:.6f}")
    print(f"Mejora total: {mejora_total:.6f}")
    print(f"Mejora porcentual: {mejora_porcentual:.2f}%")
    
    # Estadísticas adicionales
    print("\n" + "="*80)
    print("ESTADÍSTICAS ADICIONALES")
    print("="*80)
    
    valores_promedio_aptitud = estadisticas['promedio_aptitud']
    print(f"\nPromedio de aptitud promedio: {sum(valores_promedio_aptitud) / len(valores_promedio_aptitud):.6f}")
    print(f"Desviación estándar de promedios: {statistics.stdev(valores_promedio_aptitud):.6f}")
    
    # Contar generaciones de mejora
    cantidad_mejoras = 0
    for i in range(1, len(estadisticas['maxima_aptitud'])):
        if estadisticas['maxima_aptitud'][i] > estadisticas['maxima_aptitud'][i-1]:
            cantidad_mejoras += 1
    
    print(f"\nGeneraciones con mejora: {cantidad_mejoras} de {GENERATIONS} ({cantidad_mejoras/GENERATIONS*100:.2f}%)")
    print(f"Generación de mayor mejora: Gen {estadisticas['maxima_aptitud'].index(max(estadisticas['maxima_aptitud'])) + 1}")
