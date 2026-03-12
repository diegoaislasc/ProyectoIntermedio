# Predicción de Victorias en la NBA

Clasificación binaria (victoria/derrota) de partidos de la NBA usando tres algoritmos de aprendizaje supervisado.

**Asignatura:** Introducción al Aprendizaje Automático 
**Integrantes:**
- Diego Andre Islas Cadillo 2132810
- Mayela Mayte Lopez Cerino 1953581
- Luis Daniel Ruiz Dorador 2132809 

**Dataset:** [NBA Dataset – Kaggle](https://www.kaggle.com/datasets/brandonrollins/nba-dataset) (102,551 registros, 18 características)

---

## Objetivo

Predecir si un equipo ganó o perdió un partido a partir de estadísticas de rendimiento en cancha (tiros de campo, triples, tiros libres, rebotes, asistencias, robos, bloqueos, pérdidas y faltas).

## Modelos Comparados

| Modelo | Accuracy | Recall | F1-Score | Tiempo Entrenamiento |
|---|:---:|:---:|:---:|:---:|
| KNN (k=5) | 0.7847 | 0.7798 | 0.7833 | 0.004 s |
| Árbol de Decisión (depth=5) | 0.7436 | 0.7366 | 0.7413 | 0.175 s |
| **Random Forest (100 árboles)** | **0.8232** | **0.8153** | **0.8214** | 10.18 s |

**Random Forest** obtuvo el mejor rendimiento en todas las métricas. La característica más influyente fue `fieldGoalsPercentage` (importancia: 0.1657).

## Estructura del Proyecto

```
├── Proyecto_Intermedio_ML.ipynb   # Notebook con el código completo
├── Reporte.md                     # Reporte detallado del proyecto
├── TeamStatistics.csv             # Dataset utilizado
├── requirements.txt               # Dependencias de Python
└── README.md
```

## Instalación y Ejecución

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Abrir `Proyecto_Intermedio_ML.ipynb` y seleccionar el kernel del venv.

## Herramientas

Python 3.10 · scikit-learn · pandas · matplotlib · seaborn
