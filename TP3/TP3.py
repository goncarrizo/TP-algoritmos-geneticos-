import csv
import random
import matplotlib.pyplot as plt

ciudades = [
    "Cdad. de Bs. As.", "Córdoba", "Corrientes", "Formosa", "La Plata", "La Rioja",
    "Mendoza", "Neuquén", "Paraná", "Posadas", "Rawson", "Resistencia",
    "Río Gallegos", "S.F.d.V.d. Catamarca", "S.M. de Tucumán", "S.S. de Jujuy",
    "Salta", "San Juan", "San Luis", "Santa Fe", "Santa Rosa", "Sgo. Del Estero",
    "Ushuaia", "Viedma"
]

# Coordenadas aproximadas (Longitud, Latitud) para graficar en el mapa
coordenadas = {
    0: (-58.3816, -34.6037), 1: (-64.1888, -31.4201), 2: (-58.8341, -27.4692), 
    3: (-58.1781, -26.1849), 4: (-57.9545, -34.9215), 5: (-66.8507, -29.4135), 
    6: (-68.8383, -32.8908), 7: (-68.0591, -38.9516), 8: (-60.5290, -31.7320), 
    9: (-55.8961, -27.3671), 10: (-65.1023, -43.3002), 11: (-58.9839, -27.4606), 
    12: (-69.2181, -51.6226), 13: (-65.7795, -28.4696), 14: (-65.2226, -26.8083), 
    15: (-65.2971, -24.1946), 16: (-65.4117, -24.7821), 17: (-68.5253, -31.5375), 
    18: (-66.3356, -33.3017), 19: (-60.7000, -31.6333), 20: (-64.2833, -36.6167), 
    21: (-64.2615, -27.7951), 22: (-68.3030, -54.8019), 23: (-62.9967, -40.8135)
}

def cargar_distancias(ruta_archivo):
    distancias = []
    try:
        with open(ruta_archivo, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                distancias.append([int(x) for x in row])
        return distancias
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {ruta_archivo}.")
        exit()

distancias = cargar_distancias('distancias.csv')

# --- Módulo Gráfico ---
def graficar_ruta(ruta, titulo):
    """Genera un mapa conectando las coordenadas de la ruta calculada."""
    x = [coordenadas[nodo][0] for nodo in ruta]
    y = [coordenadas[nodo][1] for nodo in ruta]
    
    plt.figure(figsize=(8, 12))
    plt.plot(x, y, marker='o', linestyle='-', color='b', markersize=6)
    
    # Etiquetar cada ciudad en el mapa
    for i, nodo in enumerate(ruta[:-1]): 
        plt.text(coordenadas[nodo][0] + 0.2, coordenadas[nodo][1], ciudades[nodo], fontsize=8)
    
    # Marcar el inicio con rojo
    plt.plot(x[0], y[0], marker='s', color='red', markersize=8, label="Inicio/Fin")
    
    plt.title(titulo)
    plt.xlabel("Longitud")
    plt.ylabel("Latitud")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()

# --- Heurística ---
def vecino_mas_cercano(inicio):
    n = len(ciudades)
    visitadas = [False] * n
    ruta = [inicio]
    visitadas[inicio] = True
    distancia_total = 0
    actual = inicio

    for _ in range(n - 1):
        dist_minima = float('inf')
        siguiente = -1
        for j in range(n):
            if not visitadas[j] and distancias[actual][j] < dist_minima:
                dist_minima = distancias[actual][j]
                siguiente = j
        
        ruta.append(siguiente)
        visitadas[siguiente] = True
        distancia_total += dist_minima
        actual = siguiente

    distancia_total += distancias[actual][inicio]
    return ruta, distancia_total

def mejor_ruta_heuristica():
    mejor_ruta = []
    min_distancia = float('inf')
    mejor_inicio = -1
    
    for i in range(len(ciudades)):
        ruta, dist = vecino_mas_cercano(i)
        if dist < min_distancia:
            min_distancia = dist
            mejor_ruta = ruta
            mejor_inicio = i
            
    return mejor_ruta, min_distancia, mejor_inicio

# --- Algoritmo Genético ---
def calcular_fitness(cromosoma, ciudad_inicio=0):
    distancia = distancias[ciudad_inicio][cromosoma[0]]
    for i in range(len(cromosoma) - 1):
        distancia += distancias[cromosoma[i]][cromosoma[i+1]]
    distancia += distancias[cromosoma[-1]][ciudad_inicio]
    return distancia

def inicializar_poblacion(tamano_poblacion):
    poblacion = []
    ciudades_a_visitar = list(range(1, 24))
    for _ in range(tamano_poblacion):
        individuo = ciudades_a_visitar[:]
        random.shuffle(individuo)
        poblacion.append(individuo)
    return poblacion

def seleccion_torneo(poblacion, k=3):
    seleccionados = random.sample(poblacion, k)
    seleccionados.sort(key=lambda x: calcular_fitness(x))
    return seleccionados[0]

def crossover_ciclico(p1, p2):
    n = len(p1)
    hijo1 = [-1] * n
    hijo2 = [-1] * n
    
    idx = 0
    while True:
        hijo1[idx] = p1[idx]
        val_p2 = p2[idx]
        idx = p1.index(val_p2)
        if idx == 0: break
    for i in range(n):
        if hijo1[i] == -1: hijo1[i] = p2[i]
            
    idx = 0
    while True:
        hijo2[idx] = p2[idx]
        val_p1 = p1[idx]
        idx = p2.index(val_p1)
        if idx == 0: break
    for i in range(n):
        if hijo2[i] == -1: hijo2[i] = p1[i]
            
    return hijo1, hijo2

def mutacion_swap(individuo, prob_mutacion):
    if random.random() < prob_mutacion:
        idx1, idx2 = random.sample(range(len(individuo)), 2)
        individuo[idx1], individuo[idx2] = individuo[idx2], individuo[idx1]
    return individuo

def algoritmo_genetico(N=50, M=200, prob_mutacion=0.15, prob_crossover=0.85):
    poblacion = inicializar_poblacion(N)
    mejor_historico = None
    distancia_mejor_historico = float('inf')

    for _ in range(M):
        poblacion.sort(key=lambda x: calcular_fitness(x))
        if calcular_fitness(poblacion[0]) < distancia_mejor_historico:
            mejor_historico = poblacion[0][:]
            distancia_mejor_historico = calcular_fitness(poblacion[0])
            
        nueva_poblacion = [poblacion[0]]

        while len(nueva_poblacion) < N:
            padre1 = seleccion_torneo(poblacion)
            padre2 = seleccion_torneo(poblacion)
            
            if random.random() < prob_crossover:
                hijo1, hijo2 = crossover_ciclico(padre1, padre2)
            else:
                hijo1, hijo2 = padre1[:], padre2[:]
            
            hijo1 = mutacion_swap(hijo1, prob_mutacion)
            hijo2 = mutacion_swap(hijo2, prob_mutacion)
            
            nueva_poblacion.extend([hijo1, hijo2])
            
        poblacion = nueva_poblacion[:N]

    return mejor_historico, distancia_mejor_historico

# --- Menú Principal ---
def menu():
    while True:
        print("\n--- PROBLEMA DEL VIAJANTE (TSP) ---")
        print("1. Heurística Vecino Más Cercano ")
        print("2. Mejor recorrido heurístico global ")
        print("3. Algoritmo Genético N=50, M=200 ")
        print("4. Salir")
        
        opcion = input("Seleccione una opción: ")
        
        if opcion == '1':
            print("\nCiudades disponibles (0-23):")
            for i, c in enumerate(ciudades):
                print(f"{i}: {c}")
                
            try:
                inicio = int(input("\nIngrese el número de la ciudad de origen (0 a 23): "))
                if inicio < 0 or inicio > 23:
                    print("Error: El número debe estar entre 0 y 23.")
                    continue
                    
                ruta, dist = vecino_mas_cercano(inicio)
                ruta_cerrada = ruta + [ruta[0]]
                ruta_nombres = [ciudades[i] for i in ruta_cerrada]
                
                print(f"\nRuta: {' -> '.join(ruta_nombres)}")
                print(f"Distancia Total: {dist} km")
                
                # Graficar mapa
                graficar_ruta(ruta_cerrada, f"TSP Heurística - Origen: {ciudades[inicio]} - Dist: {dist}km")
                
            except ValueError:
                print("Error: Entrada inválida. Ingrese un número.")
            
        elif opcion == '2':
            ruta, dist, inicio = mejor_ruta_heuristica()
            ruta_cerrada = ruta + [ruta[0]]
            ruta_nombres = [ciudades[i] for i in ruta_cerrada]
            
            print(f"\nMejor ciudad de inicio: {ciudades[inicio]}")
            print(f"Ruta: {' -> '.join(ruta_nombres)}")
            print(f"Distancia Mínima Encontrada: {dist} km")
            
            # Graficar mapa
            graficar_ruta(ruta_cerrada, f"Mejor Heurística Global - Origen: {ciudades[inicio]} - Dist: {dist}km")
            
        elif opcion == '3':
            print("\nCalculando con Algoritmo Genético...")
            mejor_cromosoma, dist = algoritmo_genetico(N=50, M=200)
            ruta_completa = [0] + mejor_cromosoma + [0]
            ruta_nombres = [ciudades[i] for i in ruta_completa]
            
            print(f"Ruta Optimizada (AG): {' -> '.join(ruta_nombres)}")
            print(f"Distancia Total: {dist} km")
            
            # Graficar mapa
            graficar_ruta(ruta_completa, f"Algoritmo Genético (N=50, M=200) - Dist: {dist}km")
            
        elif opcion == '4':
            break
        else:
            print("Opción no válida.")

if __name__ == "__main__":
    menu()