# app.py - Optimización de rutas con PSO para transporte de peces

import numpy as np
import matplotlib.pyplot as plt
import random
import streamlit as st

# ============================================
# 1. FUNCIÓN DE DISTANCIA
# ============================================

def distancia_euclidiana(p1, p2):
    """Calcula distancia entre dos puntos (x,y)"""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def distancia_total(ruta, puntos):
    """Calcula distancia total de una ruta (incluyendo regreso a compañía)"""
    # Desde compañía (0,0) al primer cliente
    dist = distancia_euclidiana((0,0), puntos[ruta[0]])
    
    # Entre clientes consecutivos
    for i in range(len(ruta) - 1):
        dist += distancia_euclidiana(puntos[ruta[i]], puntos[ruta[i+1]])
    
    # Desde el último cliente de regreso a compañía
    dist += distancia_euclidiana(puntos[ruta[-1]], (0,0))
    
    return dist

# ============================================
# 2. GENERAR PUNTOS DE ENTREGA (CLIENTES)
# ============================================

def generar_puntos(n):
    """Genera n puntos aleatorios de entrega (coordenadas entre 1 y 10)"""
    random.seed(42)  # Para que siempre salgan los mismos (reproducible)
    return [(random.randint(1,10), random.randint(1,10)) for _ in range(n)]

# ============================================
# 3. ALGORITMO PSO
# ============================================

def crossover(padre1, padre2):
    """Operador de cruce para PSO en rutas (order crossover)"""
    size = len(padre1)
    start, end = sorted(random.sample(range(size), 2))
    
    hijo = [-1] * size
    for i in range(start, end + 1):
        hijo[i] = padre1[i]
    
    pos = 0
    for i in range(size):
        if padre2[i] not in hijo:
            while hijo[pos] != -1:
                pos += 1
            hijo[pos] = padre2[i]
    return hijo

def pso(puntos, num_particulas=20, iteraciones=200, c1=1.5, c2=1.5):
    """
    Algoritmo de Optimización por Enjambre de Partículas (PSO)
    Adaptado para problema de rutas (TSP)
    """
    num_clientes = len(puntos)
    
    # Inicializar partículas (rutas aleatorias)
    particulas = []
    for _ in range(num_particulas):
        ruta = list(range(num_clientes))
        random.shuffle(ruta)
        particulas.append(ruta)
    
    # Mejor posición personal (pbest)
    pbest = particulas.copy()
    fitness_pbest = [distancia_total(ruta, puntos) for ruta in pbest]
    
    # Mejor posición global (gbest)
    gbest_idx = np.argmin(fitness_pbest)
    gbest = pbest[gbest_idx].copy()
    fitness_gbest = fitness_pbest[gbest_idx]
    
    # Historial de mejora
    historial_fitness = [fitness_gbest]
    
    # Bucle principal del algoritmo
    for iteracion in range(iteraciones):
        nuevas_particulas = []
        
        for i, particula in enumerate(particulas):
            nueva_ruta = particula.copy()
            
            # Componente cognitiva (influencia de pbest)
            if random.random() < c1/3:
                nueva_ruta = crossover(nueva_ruta, pbest[i])
            
            # Componente social (influencia de gbest)
            if random.random() < c2/3:
                nueva_ruta = crossover(nueva_ruta, gbest)
            
            # Evaluar nueva ruta
            fitness_nueva = distancia_total(nueva_ruta, puntos)
            fitness_actual = distancia_total(particula, puntos)
            
            if fitness_nueva < fitness_actual:
                particula = nueva_ruta
                fitness_actual = fitness_nueva
                
                # Actualizar pbest
                if fitness_actual < fitness_pbest[i]:
                    pbest[i] = particula.copy()
                    fitness_pbest[i] = fitness_actual
                    
                    # Actualizar gbest
                    if fitness_actual < fitness_gbest:
                        gbest = particula.copy()
                        fitness_gbest = fitness_actual
        
        nuevas_particulas.append(particula)
        historial_fitness.append(fitness_gbest)
    
    return gbest, fitness_gbest, historial_fitness

# ============================================
# 4. VISUALIZACIÓN DE LA RUTA
# ============================================

def graficar_ruta(ruta, puntos, titulo="Ruta óptima"):
    """Genera gráfico de la ruta"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Coordenadas de la ruta (compañía + clientes + regreso)
    coords = [(0, 0)] + [puntos[i] for i in ruta] + [(0, 0)]
    xs, ys = zip(*coords)
    
    # Dibujar ruta
    ax.plot(xs, ys, 'o-', linewidth=2, markersize=8, color='blue')
    
    # Marcar compañía
    ax.scatter(0, 0, c='red', s=200, marker='s', label='Compañía (0,0)')
    
    # Marcar clientes
    for i, (x, y) in enumerate(puntos):
        ax.annotate(f'C{i+1}', (x, y), fontsize=9, ha='center', va='bottom')
        ax.scatter(x, y, c='green', s=100)
    
    ax.set_title(titulo)
    ax.set_xlabel("Coordenada X")
    ax.set_ylabel("Coordenada Y")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

# ============================================
# 5. INTERFAZ WEB CON STREAMLIT
# ============================================

st.set_page_config(page_title="Optimización de Rutas - PSO", layout="wide")

st.title("🐟 Optimización de Rutas para Transporte de Peces")
st.markdown("### Algoritmo bioinspirado en cardúmenes (PSO)")

# Sidebar con parámetros
st.sidebar.header("⚙️ Parámetros del algoritmo")

num_clientes = st.sidebar.slider("Número de puntos de entrega", 5, 20, 12)
num_particulas = st.sidebar.slider("Número de partículas", 10, 50, 20)
iteraciones = st.sidebar.slider("Número de iteraciones", 50, 500, 200)
c1 = st.sidebar.slider("Coeficiente cognitivo (c1)", 0.5, 2.5, 1.5)
c2 = st.sidebar.slider("Coeficiente social (c2)", 0.5, 2.5, 1.5)

# Botón para ejecutar
if st.button("🚀 Ejecutar simulación", type="primary"):
    with st.spinner("🐟 Los peces están encontrando la mejor ruta..."):
        # Generar puntos
        puntos = generar_puntos(num_clientes)
        
        # Calcular distancia inicial (ruta aleatoria)
        ruta_inicial = list(range(num_clientes))
        random.shuffle(ruta_inicial)
        distancia_inicial = distancia_total(ruta_inicial, puntos)
        
        # Ejecutar PSO
        mejor_ruta, distancia_final, historial = pso(
            puntos, 
            num_particulas=num_particulas, 
            iteraciones=iteraciones,
            c1=c1, 
            c2=c2
        )
        
        # Calcular mejora
        mejora = ((distancia_inicial - distancia_final) / distancia_inicial) * 100
        
        # Mostrar resultados
        col1, col2, col3 = st.columns(3)
        col1.metric("📏 Distancia inicial", f"{distancia_inicial:.1f}")
        col2.metric("✅ Distancia optimizada", f"{distancia_final:.1f}")
        col3.metric("📈 Mejora", f"{mejora:.1f}%", delta=f"{-mejora:.1f}%")
        
        # Mostrar gráfico de la ruta
        st.subheader("🗺️ Ruta óptima encontrada")
        fig_ruta = graficar_ruta(mejor_ruta, puntos, f"Ruta optimizada - {mejora:.1f}% de mejora")
        st.pyplot(fig_ruta)
        
        # Mostrar gráfico de convergencia
        st.subheader("📉 Evolución del algoritmo")
        fig_conv, ax = plt.subplots(figsize=(10, 4))
        ax.plot(historial, color='green', linewidth=2)
        ax.set_xlabel("Iteración")
        ax.set_ylabel("Distancia total")
        ax.set_title("Convergencia del PSO")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig_conv)
        
        st.success("✨ Simulación completada con éxito")

st.markdown("---")
st.caption("Proyecto de Computación Bioinspirada - Optimización de rutas para transporte de peces")