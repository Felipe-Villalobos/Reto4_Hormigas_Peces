# app.py - Optimización de rutas con PSO para transporte de peces
# Incluye simulación de restricciones (punto bloqueado)
# Incluye visualización de ESTIGMERMIA (señal ambiental)

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
    
    REGLAS DE ASOCIACIÓN IMPLEMENTADAS:
    - ESTIGMERMIA (interacción indirecta): gbest actúa como "señal ambiental"
    - INTERACCIÓN SOCIAL DIRECTA: componente social (c2) acerca partículas a gbest
    - MEMORIA INDIVIDUAL: componente cognitiva (c1) y pbest
    """
    num_clientes = len(puntos)
    
    # INICIALIZACIÓN: exploración inicial (sin asociaciones previas)
    particulas = []
    for _ in range(num_particulas):
        ruta = list(range(num_clientes))
        random.shuffle(ruta)
        particulas.append(ruta)
    
    # MEMORIA INDIVIDUAL: cada partícula guarda su mejor experiencia (pbest)
    pbest = particulas.copy()
    fitness_pbest = [distancia_total(ruta, puntos, punto_bloqueado) for ruta in pbest]
    
    # ESTIGMERMIA: la mejor ruta global (gbest) es la "señal ambiental"
    gbest_idx = np.argmin(fitness_pbest)
    gbest = pbest[gbest_idx].copy()
    fitness_gbest = fitness_pbest[gbest_idx]
    
    # Historial de mejora (para el gráfico de convergencia)
    historial_fitness = [fitness_gbest]
    
    # Historial de pbest para visualizar ESTIGMERMIA (señal ambiental)
    historial_pbest = [pbest[i].copy() for i in range(num_particulas)]
    
    # Bucle principal del algoritmo
    for iteracion in range(iteraciones):
        for i, particula in enumerate(particulas):
            nueva_ruta = particula.copy()
            
            # INTERACCIÓN SOCIAL DIRECTA: la partícula sigue a la mejor del grupo (gbest)
            # En la naturaleza: los peces se alinean con sus vecinos
            if random.random() < c2/3:
                nueva_ruta = crossover(nueva_ruta, gbest)
            
            # MEMORIA INDIVIDUAL: la partícula recuerda su mejor experiencia (pbest)
            # En la naturaleza: un pez recuerda dónde encontró comida
            if random.random() < c1/3:
                nueva_ruta = crossover(nueva_ruta, pbest[i])
            
            # Evaluar nueva ruta
            fitness_nueva = distancia_total(nueva_ruta, puntos, punto_bloqueado)
            fitness_actual = distancia_total(particula, puntos, punto_bloqueado)
            
            if fitness_nueva < fitness_actual:
                particula = nueva_ruta
                fitness_actual = fitness_nueva
                
                # Actualizar MEMORIA INDIVIDUAL (pbest)
                if fitness_actual < fitness_pbest[i]:
                    pbest[i] = particula.copy()
                    fitness_pbest[i] = fitness_actual
                    
                    # Actualizar ESTIGMERMIA (gbest es la señal ambiental)
                    if fitness_actual < fitness_gbest:
                        gbest = particula.copy()
                        fitness_gbest = fitness_actual
        
        # Guardar el estado actual de pbest para visualizar estigmergia
        historial_pbest.append([pbest[i].copy() for i in range(num_particulas)])
        historial_fitness.append(fitness_gbest)
    
    return gbest, fitness_gbest, historial_fitness, historial_pbest

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
# 6. VISUALIZACIÓN DE ESTIGMERMIA (CALOR DE FEROMONAS VIRTUAL)
# ============================================

def graficar_calor_estigmergia(historial_pbest, puntos, iteracion_actual):
    """
    Visualiza la "estigmergia" del sistema: cómo las mejores rutas individuales
    (pbest) crean una "señal ambiental" que guía al enjambre.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Crear un mapa de calor basado en la frecuencia con que cada cliente
    # aparece en buenas posiciones dentro de las rutas pbest
    calor = np.zeros(len(puntos))
    
    # Recorrer todas las partículas en la iteración actual
    for ruta in historial_pbest:
        for i, cliente in enumerate(ruta):
            # Los clientes al inicio de la ruta (cerca de la compañía) tienen más influencia
            peso = 1.0 / (i + 1)
            calor[cliente] += peso
    
    # Normalizar
    if np.max(calor) > 0:
        calor = calor / np.max(calor)
    
    # Dibujar puntos con tamaño proporcional a la "intensidad de estigmergia"
    for i, (x, y) in enumerate(puntos):
        size = 50 + (calor[i] * 200)
        color = plt.cm.hot(calor[i])
        ax.scatter(x, y, c=[color], s=size, edgecolors='black', linewidth=1)
        ax.annotate(f'C{i+1}', (x, y), fontsize=9, ha='center', va='bottom')
    
    # Marcar compañía
    ax.scatter(0, 0, c='blue', s=200, marker='s', label='Compañía (0,0)')
    
    ax.set_title(f"Estigmergia - Intensidad de señal ambiental (Iteración {iteracion_actual})")
    ax.set_xlabel("Coordenada X")
    ax.set_ylabel("Coordenada Y")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

# ============================================
# 7. INTERFAZ WEB CON STREAMLIT
# ============================================

st.set_page_config(page_title="Optimización de Rutas - PSO", layout="wide")

st.title("🐟 Optimización de Rutas para Transporte de Peces")
st.markdown("### Algoritmo bioinspirado en cardúmenes (PSO)")
st.markdown("**Reglas de asociación implementadas:** Estigmergia (señal ambiental) + Interacción social directa + Memoria individual")

# Sidebar con parámetros
st.sidebar.header("⚙️ Parámetros del algoritmo")

num_clientes = st.sidebar.slider("Número de puntos de entrega", 5, 20, 12)
num_particulas = st.sidebar.slider("Número de partículas", 10, 50, 20)
iteraciones = st.sidebar.slider("Número de iteraciones", 50, 500, 200)
c1 = st.sidebar.slider("Coeficiente cognitivo (c1) - Memoria individual", 0.5, 2.5, 1.5)
c2 = st.sidebar.slider("Coeficiente social (c2) - Interacción social", 0.5, 2.5, 1.5)

# ============================================
# SECCIÓN DE RESTRICCIONES
# ============================================
st.sidebar.markdown("---")
st.sidebar.subheader("🚧 Simulación de restricciones")

modo_restriccion = st.sidebar.radio(
    "Tipo de simulación",
    ["Sin restricción", "Bloquear un cliente"]
)

punto_bloqueado = None
if modo_restriccion == "Bloquear un cliente":
    puntos_temp = generar_puntos(num_clientes, semilla=42)
    cliente_bloqueado = st.sidebar.selectbox(
        "Seleccione el cliente a bloquear",
        options=list(range(1, num_clientes + 1)),
        format_func=lambda x: f"Cliente {x}"
    )
    punto_bloqueado = cliente_bloqueado - 1
    st.sidebar.info(f"🚫 Cliente {cliente_bloqueado} será excluido de la ruta")

# Botón para ejecutar
if st.button("🚀 Ejecutar simulación", type="primary"):
    with st.spinner("🐟 Los peces están encontrando la mejor ruta..."):
        # Generar puntos
        puntos = generar_puntos(num_clientes, semilla=42)
        
        # Calcular distancia inicial
        ruta_inicial = list(range(num_clientes))
        random.shuffle(ruta_inicial)
        
        if punto_bloqueado is not None:
            ruta_inicial_sin_bloqueado = [p for p in ruta_inicial if p != punto_bloqueado]
            distancia_inicial = distancia_total(ruta_inicial_sin_bloqueado, puntos, punto_bloqueado)
        else:
            distancia_inicial = distancia_total(ruta_inicial, puntos, punto_bloqueado)
        
        # Ejecutar PSO (ahora devuelve 4 valores)
        mejor_ruta, distancia_final, historial_fitness, historial_pbest = pso(
            puntos, 
            punto_bloqueado=punto_bloqueado,
            num_particulas=num_particulas, 
            iteraciones=iteraciones,
            c1=c1, 
            c2=c2
        )
        
        # Calcular mejora
        mejora = ((distancia_inicial - distancia_final) / distancia_inicial) * 100 if distancia_inicial > 0 else 0
        
        # ============================================
        # MOSTRAR RESULTADOS
        # ============================================
        st.subheader("📊 Resultados de la simulación")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("📏 Distancia inicial", f"{distancia_inicial:.1f}")
        col2.metric("✅ Distancia optimizada", f"{distancia_final:.1f}")
        col3.metric("📈 Mejora", f"{mejora:.1f}%", delta=f"{-mejora:.1f}%")
        
        if punto_bloqueado is not None:
            st.warning(f"🚫 Modo restricción activo: Cliente {punto_bloqueado+1} ha sido bloqueado y excluido de la ruta")
        
        # ============================================
        # GRÁFICO DE LA RUTA ÓPTIMA
        # ============================================
        st.subheader("🗺️ Ruta óptima encontrada")
        titulo_ruta = f"Ruta optimizada - {mejora:.1f}% de mejora"
        if punto_bloqueado is not None:
            titulo_ruta += f" (Cliente {punto_bloqueado+1} bloqueado)"
        fig_ruta = graficar_ruta(mejor_ruta, puntos, punto_bloqueado, titulo_ruta)
        st.pyplot(fig_ruta)
        
        # ============================================
        # GRÁFICO DE CONVERGENCIA
        # ============================================
        st.subheader("📉 Evolución del algoritmo (convergencia)")
        fig_conv, ax = plt.subplots(figsize=(10, 4))
        ax.plot(historial_fitness, color='green', linewidth=2)
        ax.set_xlabel("Iteración")
        ax.set_ylabel("Distancia total")
        ax.set_title("Convergencia del PSO" + (" con restricción" if punto_bloqueado else ""))
        ax.grid(True, alpha=0.3)
        st.pyplot(fig_conv)
        
        # ============================================
        # VISUALIZACIÓN DE ESTIGMERMIA (NUEVO)
        # ============================================
        st.subheader("🧬 Estigmergia - Señal ambiental del enjambre")
        st.caption("""
        **¿Qué muestra este gráfico?**  
        Los clientes con colores **más cálidos (rojo/naranja)** y **círculos más grandes** son aquellos que aparecen con mayor frecuencia en las mejores rutas individuales (pbest).  
        Esto representa la **señal ambiental** (estigmergia) que emerge del enjambre y guía a las partículas hacia soluciones óptimas.
        """)
        
        # Usar el último historial de pbest (después de todas las iteraciones)
        ultimo_historial_pbest = historial_pbest[-1]
        fig_estigmergia = graficar_calor_estigmergia(ultimo_historial_pbest, puntos, iteraciones)
        st.pyplot(fig_estigmergia)
        
        st.success("✨ Simulación completada con éxito")

st.markdown("---")
st.caption("Proyecto de Computación Bioinspirada - Optimización de rutas para transporte de peces")
st.caption("Incluye simulación de restricciones (bloqueo de clientes) y visualización de ESTIGMERMIA")