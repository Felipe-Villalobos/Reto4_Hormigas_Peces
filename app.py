# app.py - Optimización de rutas con PSO para transporte de peces
# Incluye simulación de restricciones (punto bloqueado)

import numpy as np
import matplotlib.pyplot as plt
import random
import streamlit as st

# ============================================
# 1. FUNCIONES DE DISTANCIA
# ============================================

def distancia_euclidiana(p1, p2):
    """Calcula distancia entre dos puntos (x,y)"""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def distancia_total(ruta, puntos, punto_bloqueado=None):
    """
    Calcula distancia total de una ruta
    Si punto_bloqueado no es None, lo excluye de la ruta (restricción)
    """
    # Filtrar punto bloqueado si existe
    if punto_bloqueado is not None:
        ruta_filtrada = [p for p in ruta if p != punto_bloqueado]
    else:
        ruta_filtrada = ruta
    
    if len(ruta_filtrada) == 0:
        return float('inf')
    
    # Desde compañía (0,0) al primer cliente
    dist = distancia_euclidiana((0,0), puntos[ruta_filtrada[0]])
    
    # Entre clientes consecutivos
    for i in range(len(ruta_filtrada) - 1):
        dist += distancia_euclidiana(puntos[ruta_filtrada[i]], puntos[ruta_filtrada[i+1]])
    
    # Desde el último cliente de regreso a compañía
    dist += distancia_euclidiana(puntos[ruta_filtrada[-1]], (0,0))
    
    return dist

# ============================================
# 2. GENERAR PUNTOS DE ENTREGA
# ============================================

def generar_puntos(n, semilla=42):
    """Genera n puntos aleatorios de entrega (coordenadas entre 1 y 10)"""
    random.seed(semilla)
    return [(random.randint(1,10), random.randint(1,10)) for _ in range(n)]

# ============================================
# 3. OPERADOR DE CRUCE (CROSSOVER)
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

# ============================================
# 4. ALGORITMO PSO (con soporte para restricciones)
# ============================================

def pso(puntos, punto_bloqueado=None, num_particulas=20, iteraciones=200, c1=1.5, c2=1.5):
    """
    Algoritmo de Optimización por Enjambre de Partículas (PSO)
    Adaptado para problema de rutas (TSP) con opción de restricción
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
    fitness_pbest = [distancia_total(ruta, puntos, punto_bloqueado) for ruta in pbest]
    
    # Mejor posición global (gbest)
    gbest_idx = np.argmin(fitness_pbest)
    gbest = pbest[gbest_idx].copy()
    fitness_gbest = fitness_pbest[gbest_idx]
    
    # Historial de mejora
    historial_fitness = [fitness_gbest]
    
    # Bucle principal del algoritmo
    for iteracion in range(iteraciones):
        for i, particula in enumerate(particulas):
            nueva_ruta = particula.copy()
            
            # Componente cognitiva (influencia de pbest)
            if random.random() < c1/3:
                nueva_ruta = crossover(nueva_ruta, pbest[i])
            
            # Componente social (influencia de gbest)
            if random.random() < c2/3:
                nueva_ruta = crossover(nueva_ruta, gbest)
            
            # Evaluar nueva ruta
            fitness_nueva = distancia_total(nueva_ruta, puntos, punto_bloqueado)
            fitness_actual = distancia_total(particula, puntos, punto_bloqueado)
            
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
        
        historial_fitness.append(fitness_gbest)
    
    return gbest, fitness_gbest, historial_fitness

# ============================================
# 5. VISUALIZACIÓN DE LA RUTA
# ============================================

def graficar_ruta(ruta, puntos, punto_bloqueado=None, titulo="Ruta óptima"):
    """Genera gráfico de la ruta (marcando punto bloqueado si existe)"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Filtrar punto bloqueado para la ruta
    if punto_bloqueado is not None:
        ruta_filtrada = [p for p in ruta if p != punto_bloqueado]
    else:
        ruta_filtrada = ruta
    
    # Coordenadas de la ruta (compañía + clientes + regreso)
    coords = [(0, 0)] + [puntos[i] for i in ruta_filtrada] + [(0, 0)]
    xs, ys = zip(*coords)
    
    # Dibujar ruta
    ax.plot(xs, ys, 'o-', linewidth=2, markersize=8, color='blue', label='Ruta')
    
    # Marcar compañía
    ax.scatter(0, 0, c='red', s=200, marker='s', label='Compañía (0,0)')
    
    # Marcar clientes
    for i, (x, y) in enumerate(puntos):
        color = 'gray' if punto_bloqueado is not None and i == punto_bloqueado else 'green'
        marker = 'x' if punto_bloqueado is not None and i == punto_bloqueado else 'o'
        size = 150 if punto_bloqueado is not None and i == punto_bloqueado else 100
        ax.scatter(x, y, c=color, s=size, marker=marker)
        ax.annotate(f'C{i+1}', (x, y), fontsize=9, ha='center', va='bottom')
    
    # Leyenda adicional si hay bloqueado
    if punto_bloqueado is not None:
        ax.scatter([], [], c='gray', s=100, marker='x', label=f'Cliente {punto_bloqueado+1} (BLOQUEADO)')
    
    ax.set_title(titulo)
    ax.set_xlabel("Coordenada X")
    ax.set_ylabel("Coordenada Y")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

# ============================================
# 6. INTERFAZ WEB CON STREAMLIT
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

# ============================================
# SECCIÓN DE RESTRICCIONES (NUEVO)
# ============================================
st.sidebar.markdown("---")
st.sidebar.subheader("🚧 Simulación de restricciones")

modo_restriccion = st.sidebar.radio(
    "Tipo de simulación",
    ["Sin restricción", "Bloquear un cliente"]
)

punto_bloqueado = None
if modo_restriccion == "Bloquear un cliente":
    # Generar puntos temporalmente para saber cuántos clientes hay
    puntos_temp = generar_puntos(num_clientes, semilla=42)
    cliente_bloqueado = st.sidebar.selectbox(
        "Seleccione el cliente a bloquear",
        options=list(range(1, num_clientes + 1)),
        format_func=lambda x: f"Cliente {x}"
    )
    punto_bloqueado = cliente_bloqueado - 1  # Convertir a índice 0-based
    st.sidebar.info(f"🚫 Cliente {cliente_bloqueado} será excluido de la ruta")

# Botón para ejecutar
if st.button("🚀 Ejecutar simulación", type="primary"):
    with st.spinner("🐟 Los peces están encontrando la mejor ruta..."):
        # Generar puntos
        puntos = generar_puntos(num_clientes, semilla=42)
        
        # Calcular distancia inicial (ruta aleatoria SIN considerar bloqueado para comparación justa)
        ruta_inicial = list(range(num_clientes))
        random.shuffle(ruta_inicial)
        
        # Para la distancia inicial, si hay bloqueado, calculamos la ruta sin ese punto
        if punto_bloqueado is not None:
            ruta_inicial_sin_bloqueado = [p for p in ruta_inicial if p != punto_bloqueado]
            distancia_inicial = distancia_total(ruta_inicial_sin_bloqueado, puntos, punto_bloqueado)
        else:
            distancia_inicial = distancia_total(ruta_inicial, puntos, punto_bloqueado)
        
        # Ejecutar PSO
        mejor_ruta, distancia_final, historial = pso(
            puntos, 
            punto_bloqueado=punto_bloqueado,
            num_particulas=num_particulas, 
            iteraciones=iteraciones,
            c1=c1, 
            c2=c2
        )
        
        # Calcular mejora
        mejora = ((distancia_inicial - distancia_final) / distancia_inicial) * 100 if distancia_inicial > 0 else 0
        
        # Mostrar resultados
        col1, col2, col3 = st.columns(3)
        col1.metric("📏 Distancia inicial", f"{distancia_inicial:.1f}")
        col2.metric("✅ Distancia optimizada", f"{distancia_final:.1f}")
        col3.metric("📈 Mejora", f"{mejora:.1f}%", delta=f"{-mejora:.1f}%")
        
        # Mostrar información de restricción si aplica
        if punto_bloqueado is not None:
            st.warning(f"🚫 Modo restricción activo: Cliente {punto_bloqueado+1} ha sido bloqueado y excluido de la ruta")
        
        # Mostrar gráfico de la ruta
        titulo_ruta = f"Ruta optimizada - {mejora:.1f}% de mejora"
        if punto_bloqueado is not None:
            titulo_ruta += f" (Cliente {punto_bloqueado+1} bloqueado)"
        fig_ruta = graficar_ruta(mejor_ruta, puntos, punto_bloqueado, titulo_ruta)
        st.pyplot(fig_ruta)
        
        # Mostrar gráfico de convergencia
        st.subheader("📉 Evolución del algoritmo")
        fig_conv, ax = plt.subplots(figsize=(10, 4))
        ax.plot(historial, color='green', linewidth=2)
        ax.set_xlabel("Iteración")
        ax.set_ylabel("Distancia total")
        ax.set_title("Convergencia del PSO" + (" con restricción" if punto_bloqueado else ""))
        ax.grid(True, alpha=0.3)
        st.pyplot(fig_conv)
        
        st.success("✨ Simulación completada con éxito")

st.markdown("---")
st.caption("Proyecto de Computación Bioinspirada - Optimización de rutas para transporte de peces")
st.caption("Incluye simulación de restricciones (bloqueo de clientes)")