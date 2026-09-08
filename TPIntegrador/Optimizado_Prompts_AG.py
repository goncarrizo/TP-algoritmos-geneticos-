import os
import random
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

# ==========================================
# 1. MÓDULO FRACTAL: BOX-COUNTING Y GAUSSIANA
# ==========================================

def calculate_fractal_dimension(image_array):
    """
    Calcula la dimensión fractal D mediante el método de Box-Counting
    sobre una imagen binarizada con detector de bordes Canny.
    """
    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_array

    edges = cv2.Canny(gray, 100, 200)
    binary = edges > 0

    H, W = binary.shape
    min_dim = min(H, W)
    p = int(np.floor(np.log2(min_dim)))
    sizes = 2**np.arange(p, 1, -1)

    counts = []
    for size in sizes:
        nH = H // size
        nW = W // size
        grid = binary[:nH * size, :nW * size].reshape(nH, size, nW, size)
        counts.append(np.sum(grid.any(axis=(1, 3))))

    if len(counts) < 2 or np.sum(counts) == 0:
        return 1.0

    coeffs = np.polyfit(np.log(1.0 / sizes), np.log(counts), 1)
    return float(coeffs[0])

def gaussian_fractal_score(D, D_opt=1.4, sigma=0.3):
    """
    Transforma la dimensión fractal D en un score normalizado [0, 1]
    basado en el rango estético preferido (Taylor et al., 2001).
    """
    return np.exp(-((D - D_opt) ** 2) / (2 * (sigma ** 2)))


# ==========================================
# 2. MÓDULO DE EVALUACIÓN Y FITNESS
# ==========================================

class Evaluator:
    def __init__(self, w1=0.4, w2=0.3, w3=0.3):
        self.w1 = w1  # Peso Coherencia CLIP C(p)
        self.w2 = w2  # Peso Estética LAION E(p)
        self.w3 = w3  # Peso Fractal F_fractal(p)
        self.cache = {}

    def generate_image_mock(self, prompt):
        """Generador simulado para pruebas locales sin GPU."""
        np.random.seed(abs(hash(prompt)) % (2**32))
        return np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)

    def get_clip_score(self, prompt, image):
        """Sustituir con la llamada real a CLIP model/processor."""
        return random.uniform(0.6, 0.95)

    def get_laion_aesthetic_score(self, image):
        """Sustituir con la llamada real al modelo LAION Aesthetic Predictor."""
        return random.uniform(0.5, 0.9)

    def evaluate(self, prompt):
        if prompt in self.cache:
            return self.cache[prompt]

        image = self.generate_image_mock(prompt)

        c_p = self.get_clip_score(prompt, image)
        e_p = self.get_laion_aesthetic_score(image)
        
        D = calculate_fractal_dimension(image)
        f_fractal = gaussian_fractal_score(D)

        fitness = (self.w1 * c_p) + (self.w2 * e_p) + (self.w3 * f_fractal)

        metrics = {
            "fitness": fitness,
            "clip_score": c_p,
            "aesthetic_score": e_p,
            "fractal_D": D,
            "fractal_score": f_fractal
        }
        self.cache[prompt] = metrics
        return metrics


# ==========================================
# 3. ALGORITMO GENÉTICO (OPERADORES Y POBLACIÓN)
# ==========================================

class GeneticPromptOptimizer:
    def __init__(self, seed_prompts, population_size=20, mutation_rate=0.25, crossover_rate=0.7):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.evaluator = Evaluator(w1=0.4, w2=0.3, w3=0.3)
        
        self.modifiers_pool = [
            "artstation trending", "8k resolution", "unreal engine 5",
            "octane render", "dramatic lighting", "hyperrealistic",
            "cinematic composition", "golden hour", "intricate details",
            "photorealistic", "soft focus", "vibrant colors", "volumetric fog"
        ]

        self.initial_prompts = random.sample(seed_prompts, min(population_size, len(seed_prompts)))
        self.population = list(self.initial_prompts)

    def crossover(self, parent1, parent2):
        if random.random() > self.crossover_rate:
            return parent1, parent2

        tokens1 = [t.strip() for t in parent1.split(",") if t.strip()]
        tokens2 = [t.strip() for t in parent2.split(",") if t.strip()]

        if len(tokens1) < 2 or len(tokens2) < 2:
            return parent1, parent2

        cut1 = random.randint(1, len(tokens1) - 1)
        cut2 = random.randint(1, len(tokens2) - 1)

        child1_tokens = tokens1[:cut1] + tokens2[cut2:]
        child2_tokens = tokens2[:cut2] + tokens1[cut1:]

        return ", ".join(child1_tokens), ", ".join(child2_tokens)

    def mutate(self, prompt):
        if random.random() > self.mutation_rate:
            return prompt

        tokens = [t.strip() for t in prompt.split(",") if t.strip()]
        mutation_type = random.choice(["add", "replace", "remove"])

        if mutation_type == "add" or len(tokens) <= 2:
            new_mod = random.choice(self.modifiers_pool)
            if new_mod not in tokens:
                tokens.append(new_mod)

        elif mutation_type == "replace" and len(tokens) > 0:
            idx = random.randint(0, len(tokens) - 1)
            tokens[idx] = random.choice(self.modifiers_pool)

        elif mutation_type == "remove" and len(tokens) > 3:
            idx = random.randint(0, len(tokens) - 1)
            tokens.pop(idx)

        return ", ".join(tokens)

    def tournament_selection(self, evaluated_pop, k=3):
        selected = random.sample(evaluated_pop, k)
        selected.sort(key=lambda x: x["metrics"]["fitness"], reverse=True)
        return selected[0]["prompt"]

    def run(self, num_generations=10):
        print(f"--- Iniciando Bucle Evolutivo ({num_generations} Generaciones) ---\n")

        history = []
        initial_eval = []

        # Evaluar la población inicial (Gen 0)
        for p in self.initial_prompts:
            m = self.evaluator.evaluate(p)
            initial_eval.append({"prompt": p, "metrics": m})

        for gen in range(num_generations):
            evaluated_pop = []
            for prompt in self.population:
                metrics = self.evaluator.evaluate(prompt)
                evaluated_pop.append({"prompt": prompt, "metrics": metrics})

            evaluated_pop.sort(key=lambda x: x["metrics"]["fitness"], reverse=True)
            
            best_ind = evaluated_pop[0]
            avg_fitness = np.mean([ind["metrics"]["fitness"] for ind in evaluated_pop])

            print(f"Gen [{gen+1}/{num_generations}] | Mejor Fitness: {best_ind['metrics']['fitness']:.4f} | Promedio: {avg_fitness:.4f}")
            print(f"  -> Mejor Prompt: '{best_ind['prompt'][:80]}...'")

            history.append({
                "generation": gen + 1,
                "best_prompt": best_ind["prompt"],
                "best_fitness": best_ind["metrics"]["fitness"],
                "avg_fitness": avg_fitness,
                "clip_score": best_ind["metrics"]["clip_score"],
                "aesthetic_score": best_ind["metrics"]["aesthetic_score"],
                "fractal_D": best_ind["metrics"]["fractal_D"]
            })

            # Elitismo: Conservar los 2 mejores
            new_population = [evaluated_pop[0]["prompt"], evaluated_pop[1]["prompt"]]

            # Reproducción (Cruce + Mutación)
            while len(new_population) < self.population_size:
                p1 = self.tournament_selection(evaluated_pop)
                p2 = self.tournament_selection(evaluated_pop)

                c1, c2 = self.crossover(p1, p2)

                c1 = self.mutate(c1)
                c2 = self.mutate(c2)

                new_population.append(c1)
                if len(new_population) < self.population_size:
                    new_population.append(c2)

            self.population = new_population

        return pd.DataFrame(history), initial_eval, evaluated_pop


# ==========================================
# 4. DESPLIEGUE VISUAL Y EXPORTACIÓN EN CARPETA
# ==========================================

def plot_and_export_results(df_results, initial_eval, final_eval, output_folder):
    # 1. Cálculo del Análisis Comparativo de Mejora
    initial_best_fitness = max([e['metrics']['fitness'] for e in initial_eval])
    initial_avg_fitness = np.mean([e['metrics']['fitness'] for e in initial_eval])
    
    final_best_fitness = df_results['best_fitness'].iloc[-1]
    final_avg_fitness = df_results['avg_fitness'].iloc[-1]

    mejora_best = ((final_best_fitness - initial_best_fitness) / initial_best_fitness) * 100
    mejora_avg = ((final_avg_fitness - initial_avg_fitness) / initial_avg_fitness) * 100

    print("\n" + "="*50)
    print("      ANÁLISIS COMPARATIVO DE MEJORA")
    print("="*50)
    print(f"Fitness Máximo Inicial:   {initial_best_fitness:.4f}")
    print(f"Fitness Máximo Final:     {final_best_fitness:.4f}")
    print(f"Mejora Fitness Máximo:    {mejora_best:+.2f}%\n")
    print(f"Fitness Promedio Inicial: {initial_avg_fitness:.4f}")
    print(f"Fitness Promedio Final:   {final_avg_fitness:.4f}")
    print(f"Mejora Fitness Promedio:  {mejora_avg:+.2f}%")
    print("="*50)

    # 2. Exportar Excel dentro de la carpeta resultados/
    excel_path = os.path.join(output_folder, 'resultados_optimizacion_prompts.xlsx')
    
    df_comparison = pd.DataFrame({
        'Prompt_Original': [e['prompt'] for e in initial_eval],
        'Fitness_Original': [e['metrics']['fitness'] for e in initial_eval],
        'Prompt_Evolucionado': [e['prompt'] for e in final_eval],
        'Fitness_Evolucionado': [e['metrics']['fitness'] for e in final_eval],
    })
    df_comparison['Mejora_Porcentual (%)'] = ((df_comparison['Fitness_Evolucionado'] - df_comparison['Fitness_Original']) / df_comparison['Fitness_Original']) * 100

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df_results.to_excel(writer, sheet_name='Evolucion_Por_Generacion', index=False)
        df_comparison.to_excel(writer, sheet_name='Analisis_Comparativo', index=False)

    print(f"[+] Excel guardado en: '{excel_path}'\n")

    # 3. Muestra la gráfica en pantalla (Sin Guardar en Disco)
    plt.figure(figsize=(9, 5))
    plt.plot(df_results['generation'], df_results['best_fitness'], marker='o', label='Mejor Fitness', color='#1f77b4', linewidth=2)
    plt.plot(df_results['generation'], df_results['avg_fitness'], marker='s', label='Fitness Promedio', color='#ff7f0e', linestyle='--', linewidth=2)
    
    plt.title('Evolución del Fitness por Generación (Algoritmo Genético)', fontsize=12, fontweight='bold')
    plt.xlabel('Generación', fontsize=10)
    plt.ylabel('Score de Fitness Total', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower right')
    plt.tight_layout()
    
    plt.show()


# ==========================================
# 5. EJECUCIÓN PRINCIPAL CON BÚSQUEDA AUTOMÁTICA
# ==========================================

if __name__ == "__main__":
    def find_file(filename, search_directory):
        for root, dirs, files in os.walk(search_directory):
            if filename in files:
                return os.path.join(root, filename)
        return None

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(script_dir, ".."))

    # Crear la carpeta 'resultados' en el directorio del script si no existe
    results_dir = os.path.join(script_dir, "resultados")
    os.makedirs(results_dir, exist_ok=True)

    train_path = find_file('train.csv', base_dir)
    test_path = find_file('test.csv', base_dir)

    if not train_path or not test_path:
        raise FileNotFoundError(
            f"No se encontraron 'train.csv' o 'test.csv' dentro de '{base_dir}'."
        )

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    all_prompts = pd.concat([df_train['Prompt'], df_test['Prompt']]).dropna().str.strip().tolist()

    optimizer = GeneticPromptOptimizer(
        seed_prompts=all_prompts,
        population_size=15,
        mutation_rate=0.3,
        crossover_rate=0.8
    )

    df_results, initial_eval, final_eval = optimizer.run(num_generations=5)
    
    # Exportar Excel a 'resultados/' y mostrar gráfico en pantalla
    plot_and_export_results(df_results, initial_eval, final_eval, results_dir)