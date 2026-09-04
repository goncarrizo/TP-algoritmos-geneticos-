import random
import numpy as np
import pandas as pd
import cv2

# --- OPCIONAL: Importaciones para CLIP, Diffusers y Torch (si se corre en entorno con GPU) ---
# import torch
# from transformers import CLIPProcessor, CLIPModel
# from diffusers import StableDiffusionPipeline

# ==========================================
# 1. MÓDULO FRACTAL: BOX-COUNTING Y GAUSSIANA
# ==========================================

def calculate_fractal_dimension(image_array):
    """
    Calcula la dimensión fractal D mediante el método de Box-Counting
    sobre una imagen binarizada con detector de bordes Canny.
    """
    # 1. Grayscale
    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_array

    # 2. Detección de bordes (Canny)
    edges = cv2.Canny(gray, 100, 200)
    binary = edges > 0

    # 3. Box Counting
    min_dim = min(binary.shape)
    # Hallar mayor potencia de 2 menor o igual a min_dim
    p = int(np.floor(np.log2(min_dim)))
    sizes = 2**np.arange(p, 1, -1)

    counts = []
    for size in sizes:
        # Redimensionar / contar cajas ocupadas
        shape = (binary.shape[0] // size, size, binary.shape[1] // size, size)
        counts.append(np.sum(binary[:shape[0]*size, :shape[1]*size].reshape(shape).any(axis=(1, 3))))

    if len(counts) < 2 or np.sum(counts) == 0:
        return 1.0

    # Regresión lineal log-log
    coeffs = np.polyfit(np.log(1.0 / sizes), np.log(counts), 1)
    return float(coeffs[0])

def gaussian_fractal_score(D, D_opt=1.4, sigma=0.3):
    """
    Transforma la dimensión fractal D en un score normalizado [0, 1]
    basado en el rango estético preferido (Taylor et al., 2001).
    """
    return np.exp(-((D - D_opt) ** 2) / (2 * (sigma ** 2)))


# ==========================================
# 2. MÓDULO DE EVALUACIÓN Y FITNEES
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
        # clip_score = clip_model(...)
        return random.uniform(0.6, 0.95)

    def get_laion_aesthetic_score(self, image):
        """Sustituir con la llamada real al modelo LAION Aesthetic Predictor."""
        # Retorna un valor en rango 0.0 - 1.0 (normalizado de 1-10)
        return random.uniform(0.5, 0.9)

    def evaluate(self, prompt):
        if prompt in self.cache:
            return self.cache[prompt]

        # 1. Generar imagen (512x512)
        image = self.generate_image_mock(prompt)

        # 2. Métricas
        c_p = self.get_clip_score(prompt, image)
        e_p = self.get_laion_aesthetic_score(image)
        
        D = calculate_fractal_dimension(image)
        f_fractal = gaussian_fractal_score(D)

        # 3. Fitness Global Integrado
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
        
        # Banco de modificadores estéticos y descriptores para mutaciones
        self.modifiers_pool = [
            "artstation trending", "8k resolution", "unreal engine 5",
            "octane render", "dramatic lighting", "hyperrealistic",
            "cinematic composition", "golden hour", "intricate details",
            "photorealistic", "soft focus", "vibrant colors", "volumetric fog"
        ]

        # Inicialización de población usando el dataset proporcionado
        self.population = random.sample(seed_prompts, min(population_size, len(seed_prompts)))

    def crossover(self, parent1, parent2):
        """Cruzamiento en nivel de frases/descriptores (tokens separados por coma)."""
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
        """Operador de mutación: sustitución, eliminación o adición de descriptores."""
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
        """Selección por Torneo."""
        selected = random.sample(evaluated_pop, k)
        selected.sort(key=lambda x: x["metrics"]["fitness"], reverse=True)
        return selected[0]["prompt"]

    def run(self, num_generations=10):
        print(f"--- Iniciando Bucle Evolutivo ({num_generations} Generaciones) ---\n")

        history = []

        for gen in range(num_generations):
            # 1. Evaluación de la población
            evaluated_pop = []
            for prompt in self.population:
                metrics = self.evaluator.evaluate(prompt)
                evaluated_pop.append({"prompt": prompt, "metrics": metrics})

            # Ordenar por Fitness descendente
            evaluated_pop.sort(key=lambda x: x["metrics"]["fitness"], reverse=True)
            
            best_ind = evaluated_pop[0]
            avg_fitness = np.mean([ind["metrics"]["fitness"] for ind in evaluated_pop])

            print(f"Gen [{gen+1}/{num_generations}] | Mejor Fitness: {best_ind['metrics']['fitness']:.4f} | Promedio: {avg_fitness:.4f}")
            print(f"  -> Mejor Prompt: '{best_ind['prompt'][:80]}...'")
            print(f"  -> [CLIP: {best_ind['metrics']['clip_score']:.2f} | Estética: {best_ind['metrics']['aesthetic_score']:.2f} | D Fractal: {best_ind['metrics']['fractal_D']:.2f}]\n")

            history.append({
                "generation": gen + 1,
                "best_prompt": best_ind["prompt"],
                "best_fitness": best_ind["metrics"]["fitness"],
                "avg_fitness": avg_fitness
            })

            # 2. Elitismo: Conservar los 2 mejores
            new_population = [evaluated_pop[0]["prompt"], evaluated_pop[1]["prompt"]]

            # 3. Reproducción (Cruce + Mutación)
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

        return pd.DataFrame(history)

# ==========================================
# 4. EJECUCIÓN PRINCIPAL CON RUTAS RELATIVAS
# ==========================================

if __name__ == "__main__":
    # Rutas relativas a la carpeta actual donde se ubican los datasets
    train_path = "./train.csv"
    test_path = "./test.csv"

    # Si los CSV están en una subcarpeta llamada 'TPIntegrador', usa:
    # train_path = "TPIntegrador/train.csv"
    # test_path = "TPIntegrador/test.csv"

    # Cargar datasets
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # Combinar y limpiar los prompts para formar la población inicial
    all_prompts = pd.concat([df_train['Prompt'], df_test['Prompt']]).dropna().str.strip().tolist()

    print(f"Población inicial cargada correctamente desde los datasets ({len(all_prompts)} prompts totales).")

    # Instanciar y ejecutar el optimizador evolutivo
    optimizer = GeneticPromptOptimizer(
        seed_prompts=all_prompts,
        population_size=15,
        mutation_rate=0.3,
        crossover_rate=0.8
    )

    df_results = optimizer.run(num_generations=5)
    
    # Guardar reporte final con ruta relativa
    output_path = "./resultados_optimizacion_prompts.csv"
    df_results.to_csv(output_path, index=False)
    print(f"Optimización completada. Resultados guardados en '{output_path}'.")