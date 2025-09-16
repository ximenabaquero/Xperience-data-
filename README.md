# 🎮 Análisis de Ventas de Videojuegos

## 📋 Descripción del Proyecto

Este proyecto analiza los factores que determinan el éxito comercial de los videojuegos utilizando técnicas de ciencia de datos y machine learning. El análisis se basa en datos históricos de ventas de más de 16,000 videojuegos desde 1980 hasta 2020.

## 🎯 Objetivos

### Pregunta Principal
**¿Qué determina el éxito comercial de un videojuego?**

### Preguntas de Apoyo
- ¿Qué plataformas generan mayores ventas?
- ¿Existe algún género que sea consistentemente más exitoso?
- ¿Cómo han evolucionado las ventas por año?
- ¿Qué regiones representan los mercados más importantes?
- ¿La edad del juego afecta las ventas?

## 📊 Dataset

**Fuente:** [Video Game Sales Dataset - Kaggle](https://www.kaggle.com/datasets/gregorut/videogamesales)

**Características del Dataset:**
- **Filas:** 16,598 videojuegos
- **Periodo:** 1980-2020
- **Variables:** 11 columnas principales
- **Regiones:** Norte América, Europa, Japón, Otros
- **Calidad:** Solo 1.6% valores faltantes

### Variables Principales
- `Name`: Nombre del videojuego
- `Platform`: Plataforma de lanzamiento
- `Year`: Año de lanzamiento
- `Genre`: Género del videojuego
- `Publisher`: Empresa desarrolladora
- `NA_Sales`: Ventas en Norte América (millones)
- `EU_Sales`: Ventas en Europa (millones)
- `JP_Sales`: Ventas en Japón (millones)
- `Other_Sales`: Ventas en otras regiones (millones)
- `Global_Sales`: Ventas globales totales (millones)

## 🔧 Metodología

### 1. Exploración y Limpieza de Datos
- Análisis de valores faltantes (1.6% en Year, 0.3% en Publisher)
- Imputación de valores nulos con mediana/categoría "Unknown"
- Detección y análisis de outliers (1,893 juegos con ventas >1.1M)
- Creación de variables derivadas (décadas, categorías de éxito)

### 2. Ingeniería de Características
- **Variables temporales:** Edad del juego, era de consola, indicadores modernos
- **Variables de popularidad:** Estadísticas por plataforma/género/publisher
- **Variables regionales:** Dominancia por mercado, diversidad geográfica
- **Variables de competencia:** Juegos lanzados por año/plataforma

### 3. Análisis Exploratorio
- Distribución de ventas globales
- Análisis por plataformas, géneros y décadas
- Correlaciones entre variables numéricas
- Patrones regionales de preferencias

### 4. Modelado Predictivo
- **Problema de regresión:** Predicción de ventas exactas
- **Problema de clasificación:** Predicción de éxito alto/bajo
- **Algoritmo:** Random Forest (100 árboles)
- **Validación:** Train/test split (80/20) + validación cruzada (5-fold)

### 5. Corrección de Data Leakage
- Eliminación de variable "Rank" (correlación circular)
- Exclusión de ventas regionales (componentes del target)
- Uso exclusivo de características disponibles pre-lanzamiento

## 📈 Resultados Principales

### Métricas del Modelo Final
- **R² = 0.3385** (explica 33.85% de la varianza en ventas)
- **RMSE = 1.0031** millones de unidades
- **Accuracy = 88.01%** para clasificación de éxito
- **CV R² = 0.2461 ± 0.1604** (validación cruzada)

### Hallazgos Clave

#### 🏆 Top Performers
- **Juego más exitoso:** Wii Sports (82.74M ventas)
- **Plataforma líder:** PS2 (1,256M ventas totales)
- **Género dominante:** Action (1,751M ventas totales)
- **Año pico:** 2007 (711M ventas)
- **Mercado principal:** Norte América (49.3% del total)

#### 📊 Distribución del Éxito
- **Mega-éxito (>10M):** 62 juegos (0.4%)
- **Alto éxito (2-10M):** 3,101 juegos (18.7%)
- **Éxito medio (0.5-2M):** 6,470 juegos (39.0%)
- **Bajo rendimiento (<0.5M):** 6,965 juegos (42.0%)

#### 🌍 Análisis Regional
**Norte América (4,393M ventas):**
- Géneros favoritos: Action, Sports, Shooter

**Europa (2,434M ventas):**
- Géneros favoritos: Action, Sports, Shooter

**Japón (1,291M ventas):**
- Géneros favoritos: Role-Playing, Action, Sports

### Factores Críticos de Éxito
1. **Selección de plataforma:** PlayStation, Xbox, Nintendo dominan
2. **Elección de género:** Action, Sports, Shooter tienen mayor potencial
3. **Timing de lanzamiento:** Años de nuevas consolas son clave
4. **Mercado objetivo:** Norte América es el mercado más grande
5. **Publisher establecido:** Empresas con historial tienen ventaja

## 🛠️ Tecnologías Utilizadas

### Lenguajes y Librerías
```python
# Análisis de datos
pandas==1.5.3
numpy==1.24.3

# Visualización
matplotlib==3.7.1
seaborn==0.12.2

# Machine Learning
scikit-learn==1.3.0

# Descarga de datos
kaggle==1.5.16
```

### Herramientas
- **Google Colab / Jupyter Notebook:** Desarrollo y análisis
- **Kaggle API:** Descarga automatizada de datos
- **GitHub:** Control de versiones y documentación

## 📁 Estructura del Proyecto

```
videogame-sales-analysis/
│
├── data/
│   ├── vgsales.csv                 # Dataset original de Kaggle
│   └── videojuegos_clean.csv       # Dataset procesado
│
├── notebooks/
│   ├── 01_exploracion_inicial.ipynb
│   ├── 02_limpieza_datos.ipynb
│   ├── 03_analisis_exploratorio.ipynb
│   ├── 04_ingenieria_caracteristicas.ipynb
│   ├── 05_modelado_predictivo.ipynb
│   └── videogame_analysis_complete.ipynb  # Notebook completo
│
├── src/
│   ├── data_processing.py          # Funciones de limpieza
│   ├── feature_engineering.py     # Creación de características
│   ├── visualization.py           # Funciones de visualización
│   └── modeling.py                # Modelos predictivos
│
├── results/
│   ├── figures/                   # Gráficos generados
│   ├── modelo_ventas_videojuegos.pkl  # Modelo entrenado
│   └── feature_importance.csv     # Importancia de características
│
├── docs/
│   ├── informe_ciencia_datos.pdf  # Informe técnico completo
│   └── presentacion_resultados.pdf # Presentación ejecutiva
│
├── README.md                      # Este archivo
├── requirements.txt               # Dependencias
└── .gitignore                     # Archivos excluidos de Git
```

## 🚀 Instalación y Uso

### 1. Clonar el Repositorio
```bash
git clone https://github.com/tu-usuario/videogame-sales-analysis.git
cd videogame-sales-analysis
```

### 2. Instalar Dependencias
```bash
pip install -r requirements.txt
```

### 3. Configurar Kaggle API
```bash
# Descargar kaggle.json desde tu perfil de Kaggle
mkdir ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 4. Ejecutar el Análisis
```python
# Opción 1: Notebook completo
jupyter notebook notebooks/videogame_analysis_complete.ipynb

# Opción 2: Scripts individuales
python src/data_processing.py
python src/modeling.py
```

### 5. Descarga Automática de Datos
```python
import kaggle
kaggle.api.dataset_download_files(
    'gregorut/videogamesales', 
    path='./data', 
    unzip=True
)
```

## 📊 Principales Visualizaciones

1. **Distribución de Ventas Globales:** Histograma con línea de mediana
2. **Top 10 Plataformas:** Gráfico de barras con valores
3. **Ventas por Género:** Análisis comparativo
4. **Evolución Temporal:** Serie de tiempo 1980-2020
5. **Distribución Regional:** Gráfico circular con porcentajes
6. **Matriz de Correlación:** Heatmap de variables numéricas
7. **Tendencias por Década:** Análisis multivariado
8. **Importancia de Características:** Ranking de predictores



## 💡 Recomendaciones Estratégicas

### Para Desarrolladores
1. **Priorizar plataformas principales** (PlayStation, Xbox, Nintendo)
2. **Apostar por géneros probados** (Action, Sports, Shooter)
3. **Timing estratégico** evitando períodos saturados
4. **Enfoque regional** priorizando Norte América

### Para Publishers
1. **Portfolio diversificado** combinando géneros seguros y nichos
2. **Análisis de mercado** monitoreando tendencias emergentes
3. **Marketing dirigido** adaptado por región
4. **Partnerships estratégicos** con desarrolladores establecidos

### Para Futuros Análisis
1. **Datos adicionales:** Presupuestos, ratings, ventas digitales
2. **Modelos avanzados:** Análisis de sentimientos, series temporales
3. **Monitoreo continuo:** Actualización con datos recientes

## ⚠️ Limitaciones

### Datos
- Periodo limitado (hasta 2016 principalmente)
- Sesgo hacia ventas físicas vs digitales
- Posible subrepresentación de juegos independientes
- No incluye mobile gaming moderno

### Modelo
- 66% de varianza no explicada (factores impredecibles)
- Basado en patrones históricos
- No considera factores cualitativos (calidad, marketing)
- Requiere actualización con datos recientes

### Contexto
- Industria ha cambiado significativamente post-2016
- Nuevas plataformas y modelos de negocio
- Impacto de streaming y servicios de suscripción

## 📚 Referencias

1. [Video Game Sales Dataset - Kaggle](https://www.kaggle.com/datasets/gregorut/videogamesales)
2. Entertainment Software Association. (2023). Essential Facts About the Video Game Industry
3. Newzoo. (2023). Global Games Market Report
4. Scikit-learn Documentation. (2023). Random Forest Classifier/Regressor

## 👥 Contribuidores

- **Autores:** Ximena Gil, Juliana Galindo, Kevin Carvajal, Edwin Gonzalez
- **Dataset:** Gregory Smith (Kaggle)


## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver `LICENSE` para más detalles.


---

**Última actualización:** Diciembre 2024

**Versión:** 1.0.0
