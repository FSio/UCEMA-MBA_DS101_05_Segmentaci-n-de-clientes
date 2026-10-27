# Anexo: prompts para reproducir el análisis detallado

Listado ordenado de las consignas necesarias para cubrir cada sección de [Analisis detallado de Segmentacion de Clientes Mall.md](Analisis%20detallado%20de%20Segmentacion%20de%20Clientes%20Mall.md). Cada prompt indica qué produce y cómo se verifica de forma independiente, que es lo que pide la consigna del TP.

**Aviso sobre este anexo:** es un guion de prompts reconstruido a partir del análisis ya hecho, ordenado como se daría la conversación ideal. **No es la transcripción literal** de la conversación con la IA (Claude). Para la entrega, adjuntar además la exportación real del chat (texto o capturas).

**Contexto fijo** (pegar al inicio de la conversación o en el primer prompt):

> Actuá como analista de datos. Trabajamos en un TP de MBA (Data Sciences 101). Dataset: `fuentes/Mall_Customers.csv` (200 clientes: CustomerID, Gender, Age, Annual Income (k$), Spending Score (1-100)). Objetivo: agrupar clientes para inferir conductas de compra y dirigir campañas de marketing. Usá Python (pandas, scikit-learn, scipy), semilla 42, guardá el código en `scripts/`, las tablas en `outputs/tablas/` y las figuras en `outputs/figuras/`. Si algo no sale concluyente, decilo; no muestres solo lo que confirma tu hipótesis.

---

## Resumen de prompts

| N.º | Sección del análisis | Objetivo | Salida principal |
|---|---|---|---|
| 1 | 2. Datos y calidad | Revisar el dataset | Tabla de controles, descriptivos |
| 2 | 3.1 a 3.3 | Relaciones entre variables | Correlaciones, tramos de edad, género |
| 3 | 3.4 | Elegir variables para agrupar | Comparación de conjuntos A, B, C, D |
| 4 | 4 | Elegir k | Métricas por k, codo, silueta, BIC, gap |
| 5 | 4 (contraste) | Probar otros métodos | Ward, mezcla gaussiana, DBSCAN |
| 6 | 5 | Modelo final y perfiles | Segmentos, perfiles, z-scores |
| 7 | 5.2 y 5.3 | Variables no usadas y calidad por cliente | Edad y género por segmento, silueta |
| 8 | 5.4 | Peso relativo de cada segmento | Indicador ilustrativo |
| 9 | 6 | Acciones de marketing | Frase, acción y métrica por segmento |
| 10 | 7 | Variante con edad | Conjunto B, k = 6 |
| 11 | 8 | Verificación independiente | K-means a mano, estabilidad |
| 12 | 9 | Auditoría de criterios | Criterios usados y reservas |
| 13 | 10 | Limitaciones y próximos pasos | Lista de límites |
| 14 | Cierre | Informe breve y anexo | Informe de 1 a 2 páginas |

---

## Prompt 1: Calidad de los datos (sección 2)

> Cargá `fuentes/Mall_Customers.csv`. Informá filas, columnas, tipos, valores faltantes y duplicados (con y sin CustomerID). Calculá estadísticos descriptivos de edad, ingreso y score, y la proporción por género. Hacé histogramas de las tres variables numéricas. Marcá cualquier anomalía.

- **Produce:** tabla de controles, `01_descriptivos.csv`, figura `01_distribuciones.png`.
- **Verificar:** abrir el CSV en una hoja de cálculo y comprobar a mano filas (200), conteo por género (112 y 88) y media de ingreso (60.56).

## Prompt 2: Relaciones entre variables (secciones 3.1 a 3.3)

> Calculá la correlación de Pearson y Spearman entre edad, ingreso y score, con p-valores. Hacé una matriz de dispersión coloreada por género. Resumí edad por tramos (18-29, 30-39, 40-49, 50-59, 60+) con score e ingreso medios. Compará género en las tres variables con Welch y Mann-Whitney. Explicá por qué una correlación cercana a cero entre ingreso y score no descarta que haya grupos.

- **Produce:** `02_correlaciones.csv`, `03_tramos_edad.csv`, figuras 02 y 03.
- **Verificar:** recalcular una correlación con `=COEF.DE.CORREL` en la hoja de cálculo (edad-score: -0.327).

## Prompt 3: Elección de variables (sección 3.4)

> Compará cuatro conjuntos de variables estandarizadas: A (ingreso + score), B (edad + ingreso + score), C (género + edad + ingreso + score) y D (edad + score). Para cada uno calculá el estadístico de Hopkins y la mejor silueta con su k. Advertí qué comparaciones no son estrictamente válidas (distinta dimensión). Recomendá un conjunto y justificá.

- **Produce:** `04_conjuntos_variables.csv`.
- **Verificar:** pedir a la IA (Claude) que explique por qué la silueta no es comparable entre dimensiones distintas y contrastar con una fuente del curso.

## Prompt 4: Número de grupos (sección 4)

> Sobre el conjunto A, corré K-means con 50 inicios para k de 2 a 10. Para cada k reportá inercia, silueta, Calinski-Harabasz y Davies-Bouldin. Sumá BIC con mezcla gaussiana y gap statistic. Hacé el gráfico del codo y de silueta. Decime criterio por criterio qué k apoya cada uno, **incluidos los que no apoyan k = 5 o no son concluyentes**. Calculá también la caída porcentual de inercia entre k consecutivos.

- **Produce:** `05_metricas_k_A.csv`, `07_gap_A.csv`, figuras 04 y 10.
- **Verificar:** cambiar la semilla y confirmar que el máximo de silueta sigue en k = 5.

## Prompt 5: Otros métodos como contraste (sección 4)

> Repetí el agrupamiento con k = 5 usando clustering jerárquico (Ward) y mezcla gaussiana. Compará con K-means usando silueta y ARI. Probá además DBSCAN con varias combinaciones de `eps` y `min_samples` y decime cuántos grupos y cuánto "ruido" resulta. Explicá si DBSCAN sirve como método principal acá. Hacé también la comparación visual de k = 3, 4, 5 y 6.

- **Produce:** figuras 09 y 11.
- **Verificar:** revisar que Ward y la mezcla gaussiana den ARI alto (0.94 y 0.96) contra K-means.

## Prompt 6: Modelo final y perfiles (sección 5)

> Ajustá el modelo final: K-means, k = 5, 100 inicios, semilla 42, sobre ingreso y score estandarizados. Numerá los segmentos de menor a mayor ingreso (y por score en caso de empate) para que la numeración sea repetible. Armá una tabla de perfil (n, %, edad, ingreso y score con media y rango, % de mujeres), el gráfico de dispersión coloreado por segmento y el perfil en z-scores.

- **Produce:** `08_perfil_segmentos.csv`, `09_clientes_con_segmento.csv`, `10_perfil_z.csv`, figuras 06 y 08.
- **Verificar:** filtrar en la hoja de cálculo por segmento y comprobar n y medias.

## Prompt 7: Variables no usadas y calidad por cliente (secciones 5.2 y 5.3)

> Mostrá cómo se distribuyen edad (por tramos) y género entre los segmentos. Aplicá Kruskal-Wallis para edad y chi-cuadrado para género. Calculá la silueta de cada cliente, cuántos tienen silueta negativa y cuáles son los de menor silueta. Explicá qué significa estar en la frontera entre dos segmentos.

- **Produce:** `11_tramo_edad_por_segmento.csv`, figura 07.
- **Verificar:** ubicar a mano los clientes 147 y 43 en el gráfico de dispersión.

## Prompt 8: Peso relativo de cada segmento (sección 5.4)

> El dataset no trae gasto en dinero. Proponé un indicador simple e ilustrativo para priorizar segmentos (por ejemplo ingreso × score / 100, sumado por segmento) y dejá escrito claramente que **no es facturación**. Calculá el % del total por segmento.

- **Produce:** `13_peso_economico_indicativo.csv`.
- **Verificar:** repetir la cuenta de un segmento con la hoja de cálculo y revisar que el aviso esté visible.

## Prompt 9: Acciones de marketing (sección 6)

> Para cada segmento escribí una frase que lo describa, una acción de marketing concreta y una métrica para saber si funcionó. Tené en cuenta las restricciones del segmento (por ejemplo, ingreso bajo no admite empujar crédito agresivo). Aclará que son hipótesis y proponé un piloto A/B con grupo de control para los segmentos de mayor potencial.

- **Produce:** tabla de la sección 6.
- **Verificar:** que cada acción sea distinta, medible y coherente con el perfil. Idealmente, contrastar con alguien que conozca el negocio.

## Prompt 10: Variante con edad (sección 7)

> Repetí el análisis con edad, ingreso y score (conjunto B). Reportá el mejor k, el perfil de cada grupo, un gráfico 3D y el ARI contra la solución final. Decime qué cambia respecto del conjunto A, si conviene mantenerlo y por qué.

- **Produce:** `06_metricas_k_B.csv`, `12_perfil_B.csv`, figuras 05 y 12.
- **Verificar:** que los cinco segmentos originales sean reconocibles dentro de la variante (ARI = 0.69).

## Prompt 11: Verificación independiente (sección 8)

> No te fíes de la librería. Verificá el resultado por otros caminos: (1) implementá K-means a mano (Lloyd con k-means++) y compará la inercia y el ARI con scikit-learn; (2) calculá la silueta con distancias punto a punto; (3) recalculá los centroides con promedios por grupo; (4) medí estabilidad con 200 remuestreos del 80% de los clientes; (5) medí cuántas semillas con `n_init = 1` reproducen la partición final. Informá todo, incluso si algo contradice un supuesto.

- **Produce:** tabla de la sección 8 y el hallazgo sobre mínimos locales (57.5% de coincidencia con una sola inicialización).
- **Verificar:** correr el script con otra semilla y comparar resultados.

## Prompt 12: Auditoría de criterios (sección 9)

> Respondé la pregunta de auditoría del enunciado: ¿qué criterio usaste para decidir que los grupos tienen sentido? Listá los criterios (separación, estabilidad, interpretabilidad, tamaño, coherencia externa). Después sé crítica con tu propio trabajo: ¿dónde son débiles esos criterios? Incluí que el dataset es didáctico, que k = 5 y k = 6 son casi equivalentes, que el score es un índice y no conducta observada y que los nombres de los segmentos son una interpretación tuya.

- **Produce:** secciones 9.1 a 9.3.
- **Verificar:** el estudiante decide si está de acuerdo y lo escribe con sus propias palabras (sección 9.4 y respuesta final del informe).

## Prompt 13: Limitaciones y próximos pasos (sección 10)

> Listá las limitaciones del análisis (tamaño, foto única en el tiempo, score en lugar de gasto, peso igual de las variables, indicador no validado, fuentes no verificadas) y proponé tres próximos pasos concretos.

- **Verificar:** que cada limitación no verificada quede declarada como tal.

## Prompt 14: Informe breve y anexo (cierre)

> Con el análisis completo, redactá el informe breve (una a dos páginas) respondiendo las cuatro consignas: exploración, clustering, segmentos con acción de marketing y auditoría. Hablá de la IA en tercera persona ("la IA (Claude)"). Mantené el vínculo al análisis detallado y no agregues datos que no estén en él.

- **Produce:** [Respuestas_concretas_TP_Segmentacion.md](Respuestas_concretas_TP_Segmentacion.md).
- **Verificar:** que cada número del informe coincida con el análisis detallado.

---

## Prompts transversales (usar cuando haga falta)

| Situación | Prompt sugerido |
|---|---|
| La IA presenta un resultado sin respaldo | "¿Cómo lo verificaste? Mostrame el cálculo, no solo la conclusión." |
| Un criterio no coincide con el resto | "Mostrá también los criterios que no apoyan tu conclusión y decí qué significa que no coincidan." |
| Dudas sobre reproducibilidad | "Corré todo de nuevo con otra semilla y decime qué cambia." |
| Sospecha de sobreafirmación | "Separá lo que está medido de lo que es interpretación tuya." |
| Cita de una fuente externa | "Indicá si pudiste abrir la fuente. Si no, marcala como no verificada." |
