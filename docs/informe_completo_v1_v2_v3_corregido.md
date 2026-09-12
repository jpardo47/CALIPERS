# Informe Completo: Auditoría de V1, V2 y V3 del Pipeline de Inpainting BUS
### Limpieza de calipers y anotaciones en ecografía mamaria sin firmas explotables por IA
*(Versión corregida tras verificación visual directa de `bus_0008-l.png`)*

---

## 0. Resumen ejecutivo

| Versión | Método de reconstrucción | Estado de verificación | Veredicto |
|---|---|---|---|
| **V1** | Trazo delgado + ruido blanco gaussiano i.i.d. | ⚠️ **No verificable** — el CSV recibido es idéntico al de otro método | El argumento matemático es correcto; el dato de respaldo está en duda |
| **V2** | Descomposición acústica $I=L+S$ + isófotas + calibración de speckle por anillo | ✅ Verificado (métricas + inspección visual directa) | **Detector más completo y motor de reconstrucción tan bueno o mejor que V3.** Sigue teniendo falsos positivos conocidos en algún caso puntual |
| **V3** | Mismo motor de V2 + detector "mejorado" (rechazo de núcleo sólido, cruces direccionales, fusión de texto) | ✅ Verificado (métricas + inspección visual directa) | El detector es más estricto pero **no estrictamente mejor**: elimina algunos falsos positivos a costa de introducir falsos negativos (calipers reales que deja sin limpiar) |

**Cambio de conclusión respecto a la versión anterior de este informe:** con evidencia visual directa (no solo las tablas de los reportes de auditoría), confirmé que en `bus_0008-l.png` **V2 detecta y elimina las 4 marcas de caliper correctamente, mientras que V3 deja una marca completa sin detectar y visible en la imagen final "limpia".** Esto no estaba documentado en ninguno de los dos reportes de auditoría (ni V2 ni V3 mencionan este archivo como problemático). Este hallazgo cambia la recomendación de la sección 6: **el detector base a mejorar debe ser el de V2, no adoptar el de V3 en bloque.**

---

## 1. Versión 1 (V1) — estado: pendiente de aclaración

### 1.1 Lo que el reporte afirma
Un prototipo que extrae el trazo delgado del caliper/texto y, para evitar el alisamiento del inpainting convencional, **inyecta ruido gaussiano blanco i.i.d.** ($\eta \sim \mathcal{N}(0,\sigma^2)$) sobre el parche reconstruido. El argumento matemático — que el ruido blanco tiene autocorrelación delta de Dirac (densidad espectral plana) mientras que el speckle acústico real tiene una longitud de correlación finita (~2–3 px), y que esa discrepancia dispara el gradiente de Sobel en el borde — es **correcto y bien fundamentado** como argumento físico general.

### 1.2 El problema de procedencia
Verifiqué el archivo `inpaint_quantitative_qc_benchmark_v1.csv` contra el CSV que ya habíamos analizado en un turno anterior de esta conversación (el que acompañaba a `walkthrough.md`, ahí descrito como método de **síntesis de speckle con isófotas**, no ruido blanco). **Son el mismo archivo, byte por byte** (mismo hash MD5), y la columna `method` dentro del CSV dice literalmente `"3. Híbrido Propuesto"`, sin mención de ruido blanco. Me confirmaste que V1 es la misma que se compartió al inicio — con eso entiendo que el CSV de V1 en efecto es ese, y por tanto el "Hallazgo Crítico de la Auditoría V1" (78.37 de discontinuidad de borde atribuida al ruido blanco) describe en realidad los resultados del método híbrido de speckle, no de un experimento de ruido blanco independiente.

### 1.3 Qué conservar de V1
El argumento teórico es válido y reutilizable como fundamentación (por qué el ruido blanco i.i.d. sería una mala idea), pero para el texto final del manuscrito hay que presentarlo como **justificación de diseño previa**, no como resultado experimental auditado — ya que no hay un experimento independiente que lo respalde con datos propios.

---

## 2. Versión 2 (V2) — el detector más confiable de los dos verificados

### 2.1 Método
$$I = L + S, \qquad L = G_{\sigma=1.2} * I, \qquad S = I - L$$
- $L$ (macroestructura) se reconstruye por transporte de isófotas (Telea) sobre la máscara.
- $S$ (speckle real, extraído del propio tejido) se reconstruye igual, centrado en 128.
- Energía calibrada por componente conectado: $\beta_k = \min(1.5,\ \sigma_{\text{anillo}}/\sigma_{\text{parche}})$.
- Fusión final con feathering gaussiano suave.

### 2.2 Métricas verificadas (10 imágenes BUS_BRA)

| Métrica | Baseline BBox | Trazo Simple | **Híbrido V2** |
|---|---:|---:|---:|
| Cobertura de máscara (media) | 3.32% | 1.86% | 1.86% |
| Ganancia de tejido preservado (media, relativa a su propio bbox) | 0.00% | 44.72% | 44.72% |
| Ratio varianza speckle (media / mediana) | 0.808 / 0.840 | 0.764 / 0.773 | **0.808 / 0.831** |
| Discontinuidad de borde Sobel (media / mediana) | 42.12 / 39.70 | 52.15 / 53.60 | **56.58 / 57.45** |
| SSIM fuera de máscara (media) | 0.9953 | 0.9895 | 0.9889 |
| GLCM contrast diff (mediana) | 0.435 | 0.420 | **0.416** |

### 2.3 Confirmación visual directa: detección completa en `bus_0008-l.png`

Revisé el panel de máscara y el resultado final de esta imagen a resolución ampliada:
- **Máscara V2:** cubre las 4 marcas del caliper (izquierda "1+", superior "2+", y las dos cruces centrales) más el texto quemado.
- **Resultado final V2:** las 4 marcas quedan completamente eliminadas — no queda ningún caliper visible en la imagen limpia.

### 2.4 Falsos positivos y errores conocidos (del propio reporte de V2 — pendientes de verificación visual independiente)

| Imagen | Problema reportado | Causa técnica | Impacto |
|---|---|---|---|
| `bus_0019-r.png` | Falso positivo en parche de tejido hiperecoico brillante sano (569 px, $y=158$) | Umbral de saturación sin filtro de grosor 2D | Tejido sano alterado innecesariamente; ganancia cayó a 12.81% |
| `bus_0019-r.png` (mismo archivo) | Omisión del caliper superior en $y=36$ | Atenuación por debajo del umbral | Caliper real quedó sin limpiar |
| `bus_0019-l.png` | Omisión del segundo caliper (superior derecho) | Menor contraste local | Segmentación incompleta |
| `bus_0038-s.png` | Caliper truncado en $x=0$ no detectado | Filtro de simetría descarta cruces cortadas por el borde | Caliper visible sin limpiar |
| `bus_0018-s.png` | Texto médico cortado a la mitad | Kernel de agrupamiento $13\times3$ insuficiente | Firma tipográfica parcial residual |

**Nota importante de rigor:** estos 5 problemas vienen del texto del reporte de V2, no los verifiqué yo mismo con las imágenes todavía (a diferencia de `bus_0008-l.png`, que sí verifiqué directamente). Dado que acabamos de descubrir que el reporte de V3 omitió un problema real y visible de V3 mismo, **conviene verificar visualmente estos 5 casos también antes de darlos por ciertos al 100%** — especialmente el de `bus_0019-r.png`, que es el argumento central usado para justificar el rediseño del detector en V3.

---

## 3. Versión 3 (V3) — detector más estricto, no necesariamente mejor

### 3.1 Qué añade sobre V2
1. **Rechazo de núcleo sólido**: $\text{Core}(I) = \mathcal{E}_{B_{3\times3}}(I \ge 235)$; componentes con área erosionada $\ge 6$ px se excluyen por ser "demasiado gruesos" para un trazo real.
2. **Detección direccional de cruces**: aperturas $5\times1$, $1\times5$, $5\times5$.
3. **Recuperación de calipers truncados en el borde** ($x \le 2$ o $x \ge w-2$).
4. **Fusión colineal de texto** con histéresis de semillas ($235 \to 175$).

### 3.2 Métricas verificadas (mismas 10 imágenes)

| Métrica | Baseline BBox | Trazo Simple | **Híbrido V3** |
|---|---:|---:|---:|
| Cobertura de máscara (media) | 6.34% | 2.18% | 2.18% |
| Ganancia de tejido preservado (media, relativa a su propio bbox) | 0.00% | 64.40% | 64.40% |
| Ratio varianza speckle (media / mediana) | 0.820 / 0.800 | 0.820 / 0.850 | **0.870 / 0.895** |
| Discontinuidad de borde Sobel (media / mediana) | 44.60 / 43.68 | 54.14 / 51.75 | **58.12 / 56.68** |
| SSIM fuera de máscara (media) | 0.9969 | 0.9892 | 0.9887 |
| GLCM contrast diff (mediana) | 0.227 | 0.205 | **0.203** |

### 3.3 ⚠️ Hallazgo nuevo, verificado visualmente: falso negativo en `bus_0008-l.png`

Comparé pixel a pixel las dos imágenes de comparación (`comparacion_bus_0008-l_VERSION2.png` vs `_VERSION3.png`, distintas en 6,151 píxeles, concentradas en los paneles de máscara y resultado):

- **Máscara V3:** de las 4 marcas de caliper, la cruz central-derecha **queda completamente sin cubrir** (se ve blanca, no roja).
- **Resultado final V3:** esa misma cruz **sigue completamente visible** en la imagen "limpia" — el pipeline no la tocó en absoluto.

Esto es un **falso negativo** (no un falso positivo): V3 no marcó como anotación algo que sí lo era. Es congruente con los números del CSV: en esta imagen V3 cubre *menos* área que V2 (1.97% vs 2.06%) — antes yo había interpretado esa diferencia como ruido estadístico; ahora sabemos que es directamente la marca perdida.

**Ningún reporte de auditoría (ni V2 ni V3) menciona este caso.** Es evidencia de que el rechazo de núcleo sólido de V3 (diseñado para eliminar falsos positivos como el de `bus_0019-r.png`) puede estar **sobre-corrigiendo** y empezando a rechazar también marcas de caliper reales y válidas — un trade-off clásico de precisión vs. recall, no una mejora estricta.

### 3.4 Advertencia sobre "tejido preservado": no es directamente comparable entre versiones
El $64.4\%$ de V3 vs. $44.7\%$ de V2 es relativo al bbox inicial de cada versión, que no son comparables (V3 parte de un bbox que cubre $6.34\%$ de la imagen vs. $3.32\%$ en V2). En área **final absoluta** realmente alterada:

| | Cobertura final absoluta |
|---|---:|
| V2 | **1.86%** |
| V3 | **2.18%** |

V3 modifica ligeramente *más* imagen en total, no menos — y ahora sabemos que, en al menos un caso verificado, deja *sin modificar* una marca real que debía limpiar.

### 3.5 Regresión adicional ya documentada: costura en calipers truncados en el borde

| Imagen | Discontinuidad borde V2 | Discontinuidad borde V3 | Diferencia |
|---|---:|---:|---:|
| `bus_0038-s.png` | 63.81 | **82.13** | **+18.32** (mucho peor) |

Coincide con el caliper recién recuperado en el borde de la imagen ($x=0$) — menos contexto de tejido sano alrededor para el algoritmo de isófotas.

### 3.6 Nota de rigor sobre "0% falsos positivos / 100% recuperados"
Las reglas de V3 se ajustaron específicamente para resolver los 4 casos que V2 falló en este mismo conjunto de 10 imágenes. No es una validación independiente. Y ahora, con el hallazgo de `bus_0008-l.png`, sabemos que ni siquiera es cierto que V3 recuperó todo dentro de ese mismo conjunto de ajuste — hay al menos un caliper que V3 pierde y V2 sí detecta.

---

## 4. Comparación consolidada V2 vs. V3 (datos + evidencia visual verificada)

| Imagen | Grad. borde V2 | Grad. borde V3 | Δ (V3−V2) | Ganancia tejido V2 | Ganancia tejido V3 | Verificación visual |
|---|---:|---:|---:|---:|---:|---|
| bus_0002-l | 51.34 | 50.90 | −0.44 | 55.92% | 70.14% | no revisado |
| bus_0003-l | 55.37 | 56.31 | +0.94 | 54.78% | 70.28% | no revisado |
| bus_0003-r | 58.01 | 51.31 | −6.70 | 65.48% | 56.65% | no revisado |
| **bus_0008-l** | 60.32 | 62.09 | +1.77 | 49.16% | 75.92% | **✅ V3 pierde 1 caliper (falso negativo)** |
| bus_0018-s | 48.65 | 46.94 | −1.71 | 33.14% | 54.90% | no revisado |
| bus_0019-l | 56.88 | 62.45 | +5.57 | 40.24% | 54.75% | no revisado |
| bus_0019-r | 62.67 | 65.03 | +2.36 | 12.81% | 49.40% | no revisado — aquí V2 reporta su propio falso positivo, pendiente de confirmar visualmente |
| bus_0020-l | 58.43 | 57.05 | −1.38 | 42.93% | 71.53% | no revisado |
| bus_0027-l | 50.34 | 46.96 | −3.38 | 38.82% | 65.66% | no revisado |
| bus_0038-s | 63.81 | 82.13 | **+18.32** | 53.89% | 74.72% | no revisado — coincide con caliper truncado en borde |

**V3 tiene peor discontinuidad de borde que V2 en 5 de 10 imágenes**, y en al menos 1 de las 10 (verificada visualmente) tiene además un falso negativo de detección completo. Con esta evidencia, la afirmación "V3 tiene mejor detector que V2" del reporte original ya no se sostiene sin matices — es más preciso decir que **V3 cambia el perfil de errores de V2 (menos falsos positivos reportados, pero con nueva evidencia de falsos negativos), no que lo mejora de forma estricta.**

---

## 5. Conclusión: qué conservar de cada versión

- **De V1:** el argumento teórico sobre el ruido blanco — útil como fundamentación, sin experimento propio que lo respalde.
- **De V2:** el motor de reconstrucción completo, y **el detector como base principal** — es el que demostradamente encontró y limpió las 4 marcas en `bus_0008-l.png` sin fallar.
- **De V3:** ideas puntuales del detector (rechazo de núcleo sólido, cruces direccionales, fusión de texto) **a evaluar una por una, no a adoptar en bloque** — al menos el filtro de núcleo sólido parece tener un costo de recall que hay que acotar antes de incorporarlo.

---

## 6. Cómo mejorar V2 (el detector base) para resolver sus falsos positivos conocidos

### 6.1 Antes de nada: verificar visualmente el falso positivo de `bus_0019-r.png`
Dado que acabamos de encontrar un error no reportado en V3, no des por sentado el reporte de V2 sobre su propio falso positivo sin mirarlo tú mismo. Súbeme `comparacion_bus_0019-r_V2.png` (con nombre distintivo) y lo reviso igual de a fondo que `bus_0008-l.png`.

### 6.2 Si se confirma el falso positivo de tejido brillante
Incorpora el filtro de rechazo de núcleo sólido de V3, **pero con un umbral de área más permisivo que el de V3** (por ejemplo, probar con 8–10 px en vez de 6 px) y valídalo específicamente contra `bus_0008-l.png` antes y después del cambio — ese es ahora tu caso de control para asegurarte de que el filtro no empieza a rechazar calipers reales como le pasó a V3.

### 6.3 Para calipers tenues o de bajo contraste (como el que V3 perdió en `bus_0008-l.png` y el que V2 reporta perder en `bus_0019-l.png`)
Añade la detección direccional de cruces (aperturas $5\times1$, $1\times5$) **como una segunda pasada de rescate**, no como reemplazo del umbral principal — es decir: primero corre el detector actual de V2 (que ya funciona bien), y solo sobre los píxeles que quedaron *justo* por debajo del umbral, aplica la búsqueda de geometría de cruz perpendicular para rescatar casos tenues sin arriesgar los que ya detectas bien.

### 6.4 Para texto fragmentado
Adopta la fusión colineal con histéresis de V3 ($235 \to 175$) en vez del kernel fijo $13\times3$.

### 6.5 Para calipers truncados en el borde
Añade la búsqueda en $x\le2$ / $x\ge w-2$, pero con **mirror padding** de 15–20 px antes de la reconstrucción (no el pipeline de V3 tal cual, que es el que produjo el salto de +18.32 en discontinuidad de borde en `bus_0038-s.png`).

### 6.6 Plan de validación
1. Implementa 6.2–6.5 sobre la base de código de **V2**, no de V3.
2. Vuelve a correr las 10 imágenes y confirma en cada una, con inspección visual directa (no solo el CSV):
   - `bus_0008-l.png` sigue con las 4 marcas completamente limpias (tu caso de control de "no regresión").
   - `bus_0019-r.png`, `bus_0019-l.png`, `bus_0038-s.png`, `bus_0018-s.png` mejoran sin introducir nuevos falsos negativos.
3. Corre la versión combinada sobre imágenes que nunca se usaron para ajustar ningún umbral (ni de V2 ni de V3) antes de reportar cualquier cifra de "% recuperado" o "% falsos positivos" para el jurado.
4. Considera la métrica perceptual complementaria (LPIPS o detector adversarial parche-real-vs-reconstruido) que discutimos antes, para tener una segunda señal más allá de Sobel y GLCM.
