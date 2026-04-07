# 🐟 Optimización de Rutas para Transporte de Peces

## Algoritmo bioinspirado en enjambres (PSO) con simulación de restricciones

## Desarrollado por:

**Jeison Estick Avila Guzmán**   Modelador del sistema / Desarrollador web

**Luis Felipe Villalobos Montañez**   Documentador / Validador de soluciones


## 📌 Descripción del proyecto

Este proyecto fue desarrollado para la asignatura **Computación Bioinspirada** (Semana 4). Implementa un agente inteligente basado en el algoritmo de **Optimización por Enjambre de Partículas (PSO)** para optimizar las rutas de transporte de una empresa comercializadora de peces.

### Problemática que resuelve

Los peces se mueren durante el transporte porque las rutas son muy largas. Nuestro algoritmo encuentra rutas más cortas para que los peces viajen menos tiempo y lleguen en mejores condiciones.

### Sistema biológico que inspira el algoritmo

El PSO se inspira en el comportamiento de los **cardúmenes de peces**:

- Cada pez (partícula) recuerda su mejor experiencia (pbest)
- El grupo comparte la mejor experiencia encontrada (gbest)
- Con el tiempo, todos se mueven hacia la mejor ruta

## 🚀 Cómo ejecutar la aplicación

### 1. Clonar el repositorio

git clone https://github.com/Felipe-Villalobos/Reto4_Hormigas_Peces.git
cd Reto4_Hormigas_Peces

### 2. Clonar el repositorio

pip install streamlit numpy matplotlib

### 3. Ejecutar la aplicación

streamlit run app.py

## 📚 Referencias

Kennedy, J. & Eberhart, R. (1995). Particle Swarm Optimization. Proceedings of ICNN'95.

Cimmino, A. y Corchuelo, R. (2020). Enterprise information integration: on discovering links using genetic programming. Dykinson.

Streamlit Documentation: https://docs.streamlit.io/

## 📄 Licencia
Proyecto académico desarrollado para la Corporación Universitaria Uniminuto.

## 🙏 Agradecimientos

A nuestro docente Geovanny Alberto Catamuscay Medina por la guía durante el desarrollo del proyecto.