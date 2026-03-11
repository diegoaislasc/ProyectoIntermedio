# Proyecto Intermedio de Introducción al Aprendizaje Automático

## Comparación de Algoritmos: KNN, Árboles de Decisión y Random Forest

**Asignatura:** Introducción al Aprendizaje Automático  
**Dataset:** [NBA Dataset – Kaggle (brandonrollins)](https://www.kaggle.com/datasets/brandonrollins/nba-dataset)  
**Herramientas:** Python 3.10 · scikit-learn · pandas · matplotlib · seaborn

---

## i. Introducción

El presente proyecto aborda un **problema de clasificación binaria** en el dominio del baloncesto profesional (NBA). El objetivo consiste en predecir si un equipo **ganó o perdió** un partido a partir de sus estadísticas de rendimiento registradas durante el juego.

El conjunto de datos utilizado proviene de la plataforma Kaggle y contiene más de **144,000 registros** históricos de equipos de la NBA, cada uno descrito por 48 variables originales. Para el modelado se seleccionaron **18 características numéricas** representativas del rendimiento en cancha:

| Categoría | Características |
|---|---|
| **Localía** | `home` (1 = local, 0 = visitante) |
| **Tiros de campo** | `fieldGoalsAttempted`, `fieldGoalsMade`, `fieldGoalsPercentage` |
| **Triples** | `threePointersAttempted`, `threePointersMade`, `threePointersPercentage` |
| **Tiros libres** | `freeThrowsAttempted`, `freeThrowsMade`, `freeThrowsPercentage` |
| **Rebotes** | `reboundsDefensive`, `reboundsOffensive`, `reboundsTotal` |
| **Otros** | `assists`, `blocks`, `steals`, `foulsPersonal`, `turnovers` |

Tras la limpieza de valores nulos, el dataset resultante cuenta con **102,551 registros** perfectamente balanceados (50% victorias, 50% derrotas), lo cual elimina la necesidad de técnicas de re-muestreo y permite una evaluación directa y justa de los modelos.

La variable objetivo es `win` (0 = derrota, 1 = victoria). Se busca identificar qué estadísticas de juego son los mejores predictores de victoria y comparar el rendimiento de tres algoritmos clásicos de aprendizaje supervisado.

---

## ii. Descripción de los Modelos

### K-Nearest Neighbors (KNN)

KNN es un algoritmo de **aprendizaje supervisado** clasificado como **"perezoso" (lazy learner)**. A diferencia de otros métodos, KNN no construye un modelo interno durante la fase de entrenamiento; simplemente **almacena el conjunto de datos completo** en memoria. La clasificación real ocurre en el momento de la predicción.

**Funcionamiento:** Cuando se presenta una nueva observación, KNN calcula la **distancia** entre ese punto y todos los puntos del conjunto de entrenamiento. Las métricas de distancia más comunes son:

- **Distancia Euclidiana:** $d(p, q) = \sqrt{\sum_{i=1}^{n}(p_i - q_i)^2}$
- **Distancia de Manhattan:** $d(p, q) = \sum_{i=1}^{n}|p_i - q_i|$

Luego selecciona los **k vecinos más cercanos** y asigna la clase mayoritaria entre ellos.

**Efecto del hiperparámetro k:**
- Un valor de **k pequeño** (ej. k=1 o k=3) genera fronteras de decisión muy complejas e irregulares, lo que hace al modelo altamente sensible al ruido y propenso al **sobreajuste (overfitting)**.
- Un valor de **k grande** produce fronteras más suaves pero puede llevar al **subajuste (underfitting)**, donde el modelo es incapaz de capturar patrones locales relevantes.

**Ventajas:**
- Fácil de implementar y entender.
- Versátil: funciona tanto para clasificación como para regresión.
- No requiere fase de entrenamiento explícita.

**Desventajas:**
- **Lento con datos masivos** en tiempo de predicción, ya que debe calcular distancias contra todo el dataset.
- Sufre severamente con **muchas dimensiones** (*maldición de la dimensionalidad*): en espacios de alta dimensión, las distancias tienden a uniformarse, perdiendo su capacidad discriminativa.
- Sensible a la escala de las variables (requiere normalización).

---

### Árbol de Decisión

Los Árboles de Decisión son modelos que construyen una **estructura jerárquica tipo árbol**, dividiendo recursivamente los datos en subconjuntos cada vez más homogéneos, basándose en los valores de las características.

**Criterios de división:** En cada nodo, el algoritmo selecciona la característica y el umbral que mejor separa las clases. Los criterios más utilizados son:

- **Índice Gini:** Mide la **impureza** de un nodo. Un Gini de 0 significa que el nodo es completamente puro (una sola clase). Se calcula como: $Gini = 1 - \sum_{i=1}^{c}p_i^2$, donde $p_i$ es la proporción de cada clase.
- **Ganancia de Información (Entropía):** Mide la reducción de incertidumbre al dividir. La entropía se calcula como: $H = -\sum_{i=1}^{c}p_i \log_2(p_i)$. La división que maximiza la ganancia de información es la seleccionada.

**Sobreajuste y Poda (Pruning):** Los árboles de decisión son **altamente propensos al sobreajuste** si se les permite crecer sin restricciones. Un árbol sin límites memoriza el conjunto de entrenamiento, incluyendo su ruido. Para combatir esto, se aplican técnicas de poda como limitar la **profundidad máxima (`max_depth`)**, establecer un número mínimo de muestras por hoja o un número mínimo de muestras para dividir.

**Ventajas:**
- **Fáciles de interpretar y visualizar**, lo que los hace ideales para comunicar resultados a stakeholders no técnicos.
- Baja necesidad de preprocesamiento: no requieren normalización ni estandarización.
- Manejan naturalmente variables categóricas y numéricas.

**Desventajas:**
- **Alta tendencia al sobreajuste** si no se controla la profundidad.
- Inestabilidad: pequeños cambios en los datos pueden producir árboles completamente diferentes.
- Fronteras de decisión ortogonales (paralelas a los ejes), lo que limita su flexibilidad.

---

### Random Forest

Random Forest es un método de **ensamble basado en Bagging (Bootstrap Aggregating)** que combina las predicciones de múltiples árboles de decisión para obtener un resultado más robusto y preciso.

**Funcionamiento:**
1. Se generan **múltiples subconjuntos aleatorios** del dataset de entrenamiento mediante **muestreo con reemplazo (bootstrap)**.
2. Cada subconjunto se utiliza para entrenar un árbol de decisión independiente.
3. En cada nodo de cada árbol, solo se considera un **subconjunto aleatorio de características** (no todas), lo que introduce diversidad adicional entre los árboles.
4. La predicción final se determina por **votación mayoritaria** (en clasificación): cada árbol emite un voto y la clase con más votos gana.

Este doble mecanismo de aleatoriedad (en datos y en características) es clave para reducir la varianza del modelo.

**Ventajas:**
- **Reduce enormemente el riesgo de sobreajuste** de los árboles individuales al promediar sus predicciones.
- Alta precisión en la mayoría de problemas sin requerir un ajuste extensivo de hiperparámetros.
- **Robusto ante datos faltantes** y valores atípicos.
- Proporciona una medida de importancia de las características.

**Desventajas:**
- **Computacionalmente costoso:** requiere entrenar cientos de árboles, lo que implica mayor uso de memoria y tiempo.
- **Mayor tiempo de entrenamiento** comparado con modelos individuales.
- **Menor interpretabilidad** ("caja negra"): a diferencia de un solo árbol, no es posible visualizar fácilmente la lógica de decisión del bosque completo.

---

## iii. Implementación y Resultados

### Hiperparámetros Utilizados

| Modelo | Hiperparámetro | Valor |
|---|---|---|
| **KNN** | `n_neighbors` | 5 |
| | `metric` | `minkowski` (p=2, equivalente a Euclidiana) |
| **Árbol de Decisión** | `max_depth` | 5 |
| | `criterion` | `gini` |
| | `random_state` | 42 |
| **Random Forest** | `n_estimators` | 100 |
| | `random_state` | 42 |

**Configuración de la división de datos:**
- Entrenamiento: **80%** (82,040 registros)
- Prueba: **20%** (20,511 registros)
- `random_state=42`, con estratificación (`stratify=y`)

Se aplicó **StandardScaler** para normalizar las características, lo cual es esencial para el correcto funcionamiento de KNN (algoritmo sensible a la escala de las variables).

---

### Tabla Comparativa de Resultados

| Modelo | Accuracy | Recall | F1-Score | Tiempo de Entrenamiento (s) |
|---|:---:|:---:|:---:|:---:|
| **KNN (k=5, Minkowski)** | 0.7847 | 0.7798 | 0.7833 | 0.003540 |
| **Árbol de Decisión (depth=5, Gini)** | 0.7436 | 0.7366 | 0.7413 | 0.174936 |
| **Random Forest (100 árboles)** | **0.8232** | **0.8153** | **0.8214** | 10.176229 |

---

### Matrices de Confusión

#### KNN (k=5, Minkowski)

| | Predicción: Derrota | Predicción: Victoria |
|---|:---:|:---:|
| **Real: Derrota** | 8,118 (TN) | 2,162 (FP) |
| **Real: Victoria** | 2,253 (FN) | 7,978 (TP) |

#### Árbol de Decisión (depth=5, Gini)

| | Predicción: Derrota | Predicción: Victoria |
|---|:---:|:---:|
| **Real: Derrota** | 7,715 (TN) | 2,565 (FP) |
| **Real: Victoria** | 2,695 (FN) | 7,536 (TP) |

#### Random Forest (100 árboles)

| | Predicción: Derrota | Predicción: Victoria |
|---|:---:|:---:|
| **Real: Derrota** | 8,544 (TN) | 1,736 (FP) |
| **Real: Victoria** | 1,890 (FN) | 8,341 (TP) |

---

### Top 5 — Características más Importantes (Random Forest)

| Característica | Importancia |
|---|:---:|
| `fieldGoalsPercentage` | 0.1657 |
| `reboundsDefensive` | 0.0904 |
| `reboundsTotal` | 0.0759 |
| `threePointersPercentage` | 0.0665 |
| `fieldGoalsMade` | 0.0589 |

El porcentaje de tiros de campo (`fieldGoalsPercentage`) es, con amplio margen, la característica más influyente para predecir victorias, seguido de los rebotes defensivos y la efectividad en triples.

---

## iv. Análisis y Conclusiones

### Comparación de Desempeño

Basándonos estrictamente en los resultados obtenidos:

**1. Precisión (Accuracy):**
- **Random Forest** obtuvo la mejor precisión global con **82.32%**, superando a KNN (78.47%) y al Árbol de Decisión (74.36%). Esto confirma que el método de ensamble logra generalizar mejor al combinar las predicciones de 100 árboles independientes.
- **KNN** se posiciona como el segundo mejor modelo, demostrando que la clasificación basada en similitud (vecinos cercanos) captura patrones relevantes en las estadísticas de juego.
- El **Árbol de Decisión** presenta el rendimiento más bajo, lo cual es esperado dada su limitación a una profundidad máxima de 5 niveles, que restringe su capacidad de capturar interacciones complejas entre las 18 características.

**2. Recall:**
- Random Forest también lidera en recall (**81.53%**), lo que indica que identifica correctamente una mayor proporción de victorias reales. Esto es relevante porque minimiza los falsos negativos (victorias clasificadas como derrotas).
- KNN logró un recall de **77.98%**, mientras que el Árbol de Decisión alcanzó **73.66%**.

**3. F1-Score:**
- El patrón se mantiene consistente: Random Forest (**0.8214**) > KNN (**0.7833**) > Árbol de Decisión (**0.7413**). El F1-score, al ser la media armónica entre precisión y recall, confirma que Random Forest ofrece el mejor equilibrio entre ambas métricas.

**4. Rapidez de Entrenamiento:**
- **KNN fue extremadamente rápido en entrenar** (0.0035 s), lo cual es coherente con su naturaleza de "lazy learner": no realiza ningún cómputo durante el entrenamiento, solo almacena los datos. Sin embargo, es importante notar que KNN **transfiere su costo computacional a la predicción**, donde debe calcular distancias contra todo el dataset para cada nueva observación.
- El **Árbol de Decisión** entrenó en 0.1749 s, un tiempo moderado que refleja la construcción de la estructura jerárquica.
- **Random Forest** tardó significativamente más (**10.18 s**), ya que debe construir y entrenar 100 árboles de decisión de forma independiente. Este mayor costo computacional es el precio que paga por su superior capacidad predictiva.

### Análisis de las Matrices de Confusión

Las matrices de confusión revelan que:
- Random Forest tiene la menor cantidad de errores totales (3,626 errores vs. 4,415 de KNN y 5,260 del Árbol de Decisión).
- Los tres modelos muestran un comportamiento relativamente simétrico entre falsos positivos y falsos negativos, lo cual es consistente con el balance perfecto del dataset.

### Aplicabilidad en Situaciones Reales

| Criterio | KNN | Árbol de Decisión | Random Forest |
|---|---|---|---|
| **Precisión** | Media-Alta | Media | Alta |
| **Rapidez (entrenamiento)** | Muy rápida | Rápida | Lenta |
| **Rapidez (predicción)** | Lenta (escala mal) | Muy rápida | Rápida |
| **Interpretabilidad** | Baja | Muy alta | Baja |
| **Escalabilidad** | Baja | Alta | Media |

**KNN es más adecuado cuando:**
- El dataset es de tamaño pequeño o mediano.
- Se necesita un prototipo rápido sin mucha configuración.
- El problema tiene pocas dimensiones (evita la maldición de la dimensionalidad).
- *Ejemplo de negocio:* Sistemas de recomendación en tiendas pequeñas donde se buscan productos "similares" a los que el cliente ya compró.

**Árboles de Decisión son más adecuados cuando:**
- La **interpretabilidad** es crítica y los resultados deben ser explicados a stakeholders no técnicos (gerentes, médicos, reguladores).
- Se trabaja con datos mixtos (numéricos y categóricos) sin necesidad de preprocesamiento extensivo.
- Se requiere una solución rápida tanto en entrenamiento como en predicción.
- *Ejemplo de negocio:* Aprobación de créditos bancarios, donde las regulaciones exigen que las decisiones del modelo sean explicables y auditables.

**Random Forest es más adecuado cuando:**
- La **precisión máxima** es la prioridad principal y se dispone de recursos computacionales suficientes.
- El dataset es grande y tiene muchas características.
- Se sospecha que existen interacciones complejas entre variables.
- Se necesita robustez ante datos ruidosos, valores atípicos o datos faltantes.
- *Ejemplo de negocio:* Predicción de resultados deportivos para casas de apuestas, detección de fraude financiero, o diagnóstico médico asistido donde la precisión puede salvar vidas.

### Conclusión Final

Para el problema específico de predecir victorias en la NBA a partir de estadísticas de juego, **Random Forest es la opción óptima**, logrando la mejor precisión (82.32%) a costa de un mayor tiempo de entrenamiento. Sin embargo, si el tiempo de respuesta en predicción fuera crítico (por ejemplo, en un sistema en tiempo real durante un juego), podría considerarse **KNN como alternativa viable**, siempre que se apliquen técnicas de optimización como KD-Trees o Ball-Trees para acelerar la búsqueda de vecinos. El Árbol de Decisión, aunque menos preciso, sería la elección ideal si se necesitara **explicar las reglas de decisión** a un cuerpo técnico o analistas deportivos de manera transparente.

---

*Reporte generado como parte del Proyecto Intermedio de Introducción al Aprendizaje Automático.*
