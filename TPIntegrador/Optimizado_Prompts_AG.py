import os
import zipfile
import math
import random
from io import BytesIO
import numpy as np
import cv2
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

# ==========================================
# 1. EVALUADOR DE APTITUD (FITNESS) Y MÉTRICAS
# ==========================================

class EvaluadorAptitud:
    """
    Clase encargada de calcular las métricas estéticas, semánticas y fractales
    para evaluar el desempeño de cada individuo (prompt e imagen).
    """
    def __init__(self, dispositivo="cuda" if torch.cuda.is_available() else "cpu"):
        self.dispositivo = dispositivo
        
        # Cargar modelo CLIP para la Coherencia Semántica y extracción de Embeddings
        self.modelo_clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.dispositivo)
        self.procesador_clip = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Simulación de pesos del predictor estético LAION sobre los embeddings de CLIP
        np.random.seed(42)
        self.pesos_esteticos = np.random.randn(512)

    def calcular_coherencia_clip(self, imagen: Image.Image, texto_prompt: str) -> float:
        """
        Calcula C(p): Coherencia semántica entre el texto del prompt y la imagen generada (CLIP Score).
        """
        entradas = self.procesador_clip(text=[texto_prompt], images=imagen, return_tensors="pt", padding=True).to(self.dispositivo)
        salidas = self.modelo_clip(**entradas)
        
        # Normalización vectorial y cálculo del coseno de similitud
        embeds_imagen = salidas.image_embeds / salidas.image_embeds.norm(dim=-1, keepdim=True)
        embeds_texto = salidas.text_embeds / salidas.text_embeds.norm(dim=-1, keepdim=True)
        
        similitud = (embeds_imagen @ embeds_texto.T).item()
        return max(0.0, float(similitud))

    def calcular_estetica_laion(self, imagen: Image.Image) -> float:
        """
        Calcula E(p): Calidad estética percibida proyectando los embeddings en el espacio del predictor LAION.
        """
        entradas = self.procesador_clip(images=imagen, return_tensors="pt").to(self.dispositivo)
        with torch.no_grad():
            embeds_imagen = self.modelo_clip.get_image_features(**entradas)
            embeds_imagen = embeds_imagen / embeds_imagen.norm(dim=-1, keepdim=True)
            
        vector_np = embeds_imagen.cpu().numpy().flatten()
        # Proyección lineal normalizada en el rango [0, 1] mediante sigmoide
        puntaje_bruto = np.dot(vector_np, self.pesos_esteticos)
        puntaje_normalizado = 1 / (1 + np.exp(-puntaje_bruto))
        return float(puntaje_normalizado)

    @staticmethod
    def calcular_dimension_box_counting(imagen: Image.Image) -> float:
        """
        Calcula D(p): Dimensión fractal utilizando el método de conteo de cajas (Box-Counting)
        sobre el mapa de bordes binario (Canny).
        """
        # Preprocesamiento: Conversión a escala de grises y detección de bordes mediante Canny
        imagen_grises = np.array(imagen.convert('L'))
        imagen_estandar = cv2.resize(imagen_grises, (512, 512))  # Estandarización a 512x512 píxeles
        bordes = cv2.Canny(imagen_estandar, 100, 200)
        bordes_binarios = bordes > 0

        # Tamaños de caja en potencias de 2
        tamanos_caja = [2, 4, 8, 16, 32, 64, 128, 256]
        conteos = []

        for tamano in tamanos_caja:
            # Conteo de sub-bloques/cajas que contienen al menos un píxel del mapa de bordes
            cajas_ocupadas = np.add.reduceat(
                np.add.reduceat(bordes_binarios, np.arange(0, 512, tamano), axis=0),
                np.arange(0, 512, tamano), axis=1
            )
            conteos.append(np.count_nonzero(cajas_ocupadas))

        # Ajuste de regresión lineal en escala logarítmica (log-log)
        log_tamanos = np.log(1.0 / np.array(tamanos_caja))
        log_conteos = np.log(np.array(conteos) + 1e-10)  # Se añade un valor pequeño para evitar log(0)
        
        polinomio = np.polyfit(log_tamanos, log_conteos, 1)
        dimension_D = float(polinomio[0])  # La pendiente representa la dimensión fractal D
        return dimension_D

    @staticmethod
    def funcion_gaussiana_fractal(dimension_D: float, D_optimo: float = 1.4, sigma: float = 0.3) -> float:
        """
        Transforma el valor bruto de D en un score normalizado [0, 1] mediante una curva gaussiana
        centrada en el rango de preferencia estética humana (1.3 - 1.5).
        """
        return math.exp(-((dimension_D - D_optimo) ** 2) / (2 * (sigma ** 2)))

    def evaluar_individuo(self, imagen: Image.Image, texto_prompt: str, w1=1/3, w2=1/3, w3=1/3) -> dict:
        """
        Calcula la función de aptitud global F(p) combinando los 3 pilares del marco teórico:
        F(p) = w1 * C(p) + w2 * E(p) + w3 * F_fractal(D(p))
        """
        coherencia = self.calcular_coherencia_clip(imagen, texto_prompt)
        estetica = self.calcular_estetica_laion(imagen)
        dimension_fractal = self.calcular_dimension_box_counting(imagen)
        puntaje_fractal = self.funcion_gaussiana_fractal(dimension_fractal)

        aptitud_total = (w1 * coherencia) + (w2 * estetica) + (w3 * puntaje_fractal)
        
        return {
            "aptitud": aptitud_total,
            "coherencia_clip": coherencia,
            "puntaje_estetico": estetica,
            "dimension_fractal": dimension_fractal,
            "puntaje_fractal": puntaje_fractal
        }

# ==========================================
# 2. ESTRUCTURA DEL GENOTIPO Y MOTOR DEL AG
# ==========================================

class PromptEstructurado:
    """
    Representa el cromosoma/genotipo del individuo, segmentado por categorías
    o genes funcionalmente diferenciados.
    """
    def __init__(self, sujeto: str, estilo: str, iluminacion: str, escenario: str):
        self.genes = {
            "sujeto": sujeto,
            "estilo": estilo,
            "iluminacion": iluminacion,
            "escenario": escenario
        }

    def a_texto(self) -> str:
        """Convierte los genes del cromosoma en la cadena final del prompt."""
        return f"{self.genes['sujeto']}, {self.genes['estilo']}, {self.genes['iluminacion']}, {self.genes['escenario']}"

class OptimizadorGeneticoPrompts:
    """
    Motor principal que administra la población, la caché de la base de datos local
    y la evolución intergeneracional.
    """
    def __init__(self, ruta_zip: str, tamano_poblacion: int = 10, generaciones: int = 5):
        self.ruta_zip = ruta_zip
        self.tamano_poblacion = tamano_poblacion
        self.generaciones = generaciones
        self.evaluador = EvaluadorAptitud()
        self.cache_base_datos = {}
        
        # Banco de alelos/genes disponibles para mutación y diversificación
        self.banco_genes = {
            "sujeto": ["A cat", "A surreal cat", "A detailed cat portrait", "A cyberpunk cat"],
            "estilo": ["realistic", "cyberpunk", "surrealist", "hyperrealistic art"],
            "iluminacion": ["sunset", "neon lights", "cinematic lighting", "dramatic shading"],
            "escenario": ["forest", "futuristic city", "colorful landscape", "jungle"]
        }

    def cargar_base_datos_desde_zip(self):
        """Extrae las imágenes y prompts almacenados en el archivo ZIP."""
        print(f"--> Extrayendo y procesando base de datos desde: {self.ruta_zip}")
        with zipfile.ZipFile(self.ruta_zip, 'r') as z:
            for nombre_archivo in z.namelist():
                if nombre_archivo.endswith(('.png', '.jpg', '.jpeg')):
                    # Carga de imagen
                    datos_imagen = z.read(nombre_archivo)
                    imagen = Image.open(BytesIO(datos_imagen)).convert('RGB')
                    
                    # Carga o deducción del texto del prompt
                    archivo_texto = os.path.splitext(nombre_archivo)[0] + '.txt'
                    if archivo_texto in z.namelist():
                        texto_prompt = z.read(archivo_texto).decode('utf-8').strip()
                    else:
                        texto_prompt = os.path.splitext(os.path.basename(nombre_archivo))[0].replace('_', ' ')

                    # Evaluación inicial y guardado en la caché
                    metricas = self.evaluador.evaluar_individuo(imagen, texto_prompt)
                    self.cache_base_datos[texto_prompt] = {
                        "imagen": imagen,
                        "metricas": metricas
                    }
        print(f"--> Base de datos cargada exitosamente. Total de registros: {len(self.cache_base_datos)}")

    def obtener_aptitud(self, prompt: PromptEstructurado) -> dict:
        """Recupera el fitness desde la caché local o busca el par más cercano."""
        texto = prompt.a_texto()
        if texto in self.cache_base_datos:
            return self.cache_base_datos[texto]["metricas"]
        
        # Simulación fuera de línea: si el prompt es nuevo, se usa un registro de la base
        prompt_cercano = random.choice(list(self.cache_base_datos.keys()))
        imagen = self.cache_base_datos[prompt_cercano]["imagen"]
        
        metricas = self.evaluador.evaluar_individuo(imagen, texto)
        self.cache_base_datos[texto] = {"imagen": imagen, "metricas": metricas}
        return metricas

    def cruzamiento(self, padre1: PromptEstructurado, padre2: PromptEstructurado) -> PromptEstructurado:
        """Operador de cruce uniforme sobre la estructura de genes."""
        genes_hijo = {}
        for categoria in padre1.genes.keys():
            genes_hijo[categoria] = padre1.genes[categoria] if random.random() > 0.5 else padre2.genes[categoria]
        return PromptEstructurado(**genes_hijo)

    def mutacion(self, prompt: PromptEstructurado, tasa_mutacion: float = 0.3) -> PromptEstructurado:
        """Operador de mutación categórica."""
        for categoria in prompt.genes.keys():
            if random.random() < tasa_mutacion:
                prompt.genes[categoria] = random.choice(self.banco_genes[categoria])
        return prompt

    def ejecutar_evolucion(self):
        """Ejecuta el bucle de optimización evolutiva durante N generaciones."""
        self.cargar_base_datos_desde_zip()
        
        # 1. Sembrado de la Población Inicial (Seeding con los mejores individuos de la BD)
        cache_ordenada = sorted(self.cache_base_datos.items(), key=lambda x: x[1]["metricas"]["aptitud"], reverse=True)
        poblacion = []
        
        for texto, datos in cache_ordenada[:self.tamano_poblacion]:
            partes = [p.strip() for p in texto.split(',')]
            sujeto = partes[0] if len(partes) > 0 else "A cat"
            estilo = partes[1] if len(partes) > 1 else "realistic"
            iluminacion = partes[2] if len(partes) > 2 else "sunset"
            escenario = partes[3] if len(partes) > 3 else "forest"
            poblacion.append(PromptEstructurado(sujeto, estilo, iluminacion, escenario))

        print("\n==========================================")
        print("   INICIANDO BUCLE EVOLUTIVO DE PROMPTS   ")
        print("==========================================")
        
        for gen in range(1, self.generaciones + 1):
            evaluaciones = [(ind, self.obtener_aptitud(ind)) for ind in poblacion]
            evaluaciones.sort(key=lambda x: x[1]["aptitud"], reverse=True)

            mejor_ind, mejores_metricas = evaluaciones[0]
            aptitud_promedio = np.mean([e[1]["aptitud"] for e in evaluaciones])

            print(f"\n--- GENERACIÓN {gen} ---")
            print(f"Mejor Aptitud Global: {mejores_metricas['aptitud']:.4f} | Aptitud Promedio: {aptitud_promedio:.4f}")
            print(f"Mejor Prompt: '{mejor_ind.a_texto()}'")
            print(f" [CLIP: {mejores_metricas['coherencia_clip']:.3f} | Estética: {mejores_metricas['puntaje_estetico']:.3f} | Dim. Fractal D: {mejores_metricas['dimension_fractal']:.3f} (Score Gaussiano: {mejores_metricas['puntaje_fractal']:.3f})]")

            # Selección por Torneo y construcción de la siguiente generación
            siguiente_generacion = [evaluaciones[0][0]]  # Elitismo: Conservar el mejor individuo
            
            while len(siguiente_generacion) < self.tamano_poblacion:
                # Selección por torneo
                torneo = random.sample(evaluaciones, 3)
                padre1 = max(torneo, key=lambda x: x[1]["aptitud"])[0]
                torneo = random.sample(evaluaciones, 3)
                padre2 = max(torneo, key=lambda x: x[1]["aptitud"])[0]

                # Aplicación de operadores evolutivos
                hijo = self.cruzamiento(padre1, padre2)
                hijo = self.mutacion(hijo)
                siguiente_generacion.append(hijo)

            poblacion = siguiente_generacion

# ==========================================
# 3. EJECUCIÓN DEL PROGRAMA
# ==========================================
if __name__ == "__main__":
    # Ruta absoluta configurada hacia la carpeta del TP Integrador.
    # Reemplaza 'base_de_datos.zip' por el nombre real de tu archivo comprimido.
    RUTA_ZIP_BASE_DATOS = r"C:\Users\Usuario\Desktop\facultad\algoritmos geneticos\TP-algoritmos-geneticos-\TPIntegrador\base_de_datos.zip"
    
    if os.path.exists(RUTA_ZIP_BASE_DATOS):
        optimizador = OptimizadorGeneticoPrompts(
            ruta_zip=RUTA_ZIP_BASE_DATOS, 
            tamano_poblacion=5, 
            generaciones=3
        )
        optimizador.ejecutar_evolucion()
    else:
        print(f"Error: No se encontró el archivo en la ruta especificada:\n{RUTA_ZIP_BASE_DATOS}")