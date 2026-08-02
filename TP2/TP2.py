"""TP2-Unico: Un solo archivo con las 3 soluciones pedidas y un menú.

Opciones:
1) Búsqueda exhaustiva (fuerza bruta) y medición de tiempo.
2) Algoritmo goloso con comparación (si es posible) con la exhaustiva.
3) Instancia de 3 elementos (pesos en gramos) — ejecutar ambos y analizar.

Ejecutar: python TP2-Ab/TP2_unico.py
"""
import itertools
import time
import math
from typing import List, Dict, Tuple


def mochila_exhaustiva(elementos: List[Dict], capacidad: int) -> Tuple[List[Dict], int, float, List[Tuple[List[Dict], int, int]]]:
    t0 = time.perf_counter()
    n = len(elementos)
    factibles = []
    for r in range(n + 1):
        for comb in itertools.combinations(elementos, r):
            peso = sum(it['weight'] for it in comb)
            valor = sum(it['value'] for it in comb)
            if peso <= capacidad:
                factibles.append((list(comb), peso, valor))

    factibles.sort(key=lambda x: (-x[2], x[1]))
    mejor, mejor_peso, mejor_valor = ([], 0, 0)
    if factibles:
        mejor, mejor_peso, mejor_valor = factibles[0]

    tiempo = time.perf_counter() - t0
    return mejor, mejor_valor, tiempo, factibles


def mochila_golosa(elementos: List[Dict], capacidad: int) -> Tuple[List[Dict], int, float]:
    t0 = time.perf_counter()
    elementos_ordenados = sorted(elementos, key=lambda it: (it['value'] / it['weight']) if it['weight'] > 0 else math.inf, reverse=True)
    elegidos = []
    peso = 0
    valor = 0
    for it in elementos_ordenados:
        if peso + it['weight'] <= capacidad:
            elegidos.append(it)
            peso += it['weight']
            valor += it['value']

    tiempo = time.perf_counter() - t0
    return elegidos, valor, tiempo


def imprimir_subconjunto(subset: List[Dict]) -> str:
    if not subset:
        return '[]'
    return '[' + ', '.join(it['name'] for it in subset) + ']'


def run_punto1():
    print('\n--- Punto 1: Búsqueda exhaustiva (datos fijos de la consigna) ---')
    # Datos fijos (volumen en cm^3 y valor $) según la imagen proporcionada
    elementos = [
        {'name': '1', 'weight': 150, 'value': 20},
        {'name': '2', 'weight': 325, 'value': 40},
        {'name': '3', 'weight': 600, 'value': 50},
        {'name': '4', 'weight': 805, 'value': 36},
        {'name': '5', 'weight': 430, 'value': 25},
        {'name': '6', 'weight': 1200, 'value': 64},
        {'name': '7', 'weight': 770, 'value': 54},
        {'name': '8', 'weight': 60, 'value': 18},
        {'name': '9', 'weight': 930, 'value': 46},
        {'name': '10','weight': 353, 'value': 28},
    ]
    capacidad = 4200

    print(f"Capacidad mochila: {capacidad} cm^3. Elementos: {len(elementos)}")
    for it in elementos:
        print(f" - {it['name']}: volumen={it['weight']} cm^3, valor=${it['value']}")

    mejor, valor, tiempo, factibles = mochila_exhaustiva(elementos, capacidad)
    print('\nResultado (exhaustiva):')
    print('  Mejor subconjunto:', imprimir_subconjunto(mejor))
    print('  Valor total: $', valor)
    print(f'  Tiempo (exhaustiva): {tiempo:.6f} s')
    print('\nTop 10 subconjuntos factibles (valor-desc):')
    for i, (subset, w, v) in enumerate(factibles[:10], start=1):
        items = ','.join(it['name'] for it in subset) if subset else 'vacío'
        print(f'  {i:>2}. valor=${v} peso={w} items={items}')


def run_punto2():
    print('\n--- Punto 2: Algoritmo goloso (datos fijos) y comparación ---')
    # Usar los mismos datos fijos que en el punto 1 (tabla de 10 objetos)
    elementos = [
        {'name': '1', 'weight': 150, 'value': 20},
        {'name': '2', 'weight': 325, 'value': 40},
        {'name': '3', 'weight': 600, 'value': 50},
        {'name': '4', 'weight': 805, 'value': 36},
        {'name': '5', 'weight': 430, 'value': 25},
        {'name': '6', 'weight': 1200, 'value': 64},
        {'name': '7', 'weight': 770, 'value': 54},
        {'name': '8', 'weight': 60, 'value': 18},
        {'name': '9', 'weight': 930, 'value': 46},
        {'name': '10','weight': 353, 'value': 28},
    ]
    capacidad = 4200

    print(f"Capacidad mochila: {capacidad} cm^3. Elementos: {len(elementos)}")
    for it in elementos:
        print(f" - {it['name']}: volumen={it['weight']} cm^3, valor=${it['value']}")

    elegidos, valor_goloso, tiempo_goloso = mochila_golosa(elementos, capacidad)
    print('\nResultado (goloso):')
    print('  Subconjunto elegido:', imprimir_subconjunto(elegidos))
    print('  Valor total: $', valor_goloso)
    print(f'  Tiempo (goloso): {tiempo_goloso:.6f} s')

    # Comparación con exhaustiva (10 elementos -> viable)
    mejor, valor_exh, tiempo_exh, factibles = mochila_exhaustiva(elementos, capacidad)
    print('\nComparación con exhaustiva:')
    print('  Mejor (exhaustiva):', imprimir_subconjunto(mejor), 'valor=$', valor_exh)
    print(f'  Tiempo (exhaustiva): {tiempo_exh:.6f} s')
    if valor_goloso == valor_exh:
        print('  Conclusión: el goloso encontró la solución óptima en este caso.')
    else:
        print('  Conclusión: el goloso NO encontró la solución óptima en este caso.')
        print('  Diferencia en valor:', valor_exh - valor_goloso)


def run_punto3():
    print('\n--- Punto 3: Instancia 3 elementos (pesos en gramos) ---')
    elementos = [
        {'name': 'A', 'weight': 1800, 'value': 72},
        {'name': 'B', 'weight': 600,  'value': 36},
        {'name': 'C', 'weight': 1200, 'value': 60},
    ]
    capacidad = 3000
    mejor_exh, valor_exh, tiempo_exh, factibles = mochila_exhaustiva(elementos, capacidad)
    elegidos_goloso, valor_goloso, tiempo_goloso = mochila_golosa(elementos, capacidad)

    print('\nInstancia: capacidad=3000 g')
    for it in elementos:
        print(f" - {it['name']}: peso={it['weight']} g, valor=${it['value']}")

    # A) Solución: mostrar resultados de ambos algoritmos
    print('\nA) Solución — Resultados de los algoritmos')
    print('\n  Exhaustiva:')
    print('    Mejor subconjunto:', imprimir_subconjunto(mejor_exh))
    print('    Valor total: $', valor_exh)
    print(f'    Tiempo (exhaustiva): {tiempo_exh:.6f} s')

    print('\n  Goloso:')
    print('    Subconjunto elegido:', imprimir_subconjunto(elegidos_goloso))
    print('    Valor total: $', valor_goloso)
    print(f'    Tiempo (goloso): {tiempo_goloso:.6f} s')

    # B) Análisis y conclusiones
    print('\nB) Análisis y conclusiones')
    if valor_goloso == valor_exh:
        print('  - El algoritmo goloso encontró la solución óptima en esta instancia.')
    else:
        print('  - El algoritmo goloso NO encontró la solución óptima en esta instancia.')
        print('  - Diferencia en valor (exhaustiva - goloso):', valor_exh - valor_goloso)

    print('\nListado de subconjuntos factibles ordenados (valor-desc):')
    for i, (subset, w, v) in enumerate(factibles, start=1):
        items = ','.join(it['name'] for it in subset) if subset else 'vacío'
        print(f'  {i:>2}. valor=${v} peso={w} items={items}')


def main():
    while True:
        print('\n=== Menú unificado TP2 ===')
        print('1) Punto 1 - Exhaustiva')
        print('2) Punto 2 - Goloso y comparación')
        print('3) Punto 3 - Ejemplo 3 elementos (3000 g)')
        print('q) Salir')
        op = input('\nElija opción: ').strip().lower()
        if op == '1':
            run_punto1()
        elif op == '2':
            run_punto2()
        elif op == '3':
            run_punto3()
        elif op == 'q':
            print('Saliendo.')
            break
        else:
            print('Opción no válida.')


if __name__ == '__main__':
    main()
