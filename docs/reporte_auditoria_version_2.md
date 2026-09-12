# Informe de Auditoría y Validación Cuantitativa: Versión 2 (V2)
## Inpainting por Descomposición Residual Acústica y Agrupamiento Morfológico Inicial en BUS_BRA

**Autor:** Equipo de Investigación en Visión Artificial y Ultrasonido  
**Fecha de Evaluación:** 6 de Septiembre de 2026  
**Dataset de Auditoría V2:** 10 Imágenes representativas curadas de `BUS_BRA` (BUSBRA)  
**Archivos Auditables Asociados:**
- Archivo de métricas brutas: [`metricas_cuantitativas_bus_bra.csv`](file:///c:/Users/sebas/OneDrive/Escritorio/PRUEBAINPAINTING/resultados_version_2/metricas_cuantitativas_bus_bra.csv)
- Mosaico visual compuesto (10x4): [`mosaico_10_imagenes_bus_bra.png`](file:///c:/Users/sebas/OneDrive/Escritorio/PRUEBAINPAINTING/resultados_version_2/mosaico_10_imagenes_bus_bra.png)
- Comparativas de 10 imágenes: `comparacion_bus_0002-l.png` a `comparacion_bus_0038-s.png`

---

## 1. Resumen Ejecutivo para el Jurado

La **Versión 2 (V2)** representó un salto metodológico fundamental respecto a la Versión 1, diseñado para resolver de raíz el problema de la discontinuidad de borde ocasionada por el ruido blanco sintético.

### Principales Avances Introducidos en V2:
1. **Descomposición Residual Acústica ($I = L + S$)**: En lugar de inyectar ruido sintético i.i.d., se descompone la imagen ecográfica en macroestructura de baja frecuencia ($L$) y microestructura acústica real ($S = I - L$). El residuo $S$ transporta la respuesta de la Función de Dispersión de Punto (PSF) del transductor, propagándose a través del trazo mediante isófotas continuas y calibración local de varianza.
2. **Caída Radical de la Discontinuidad de Borde**: La discontinuidad de Sobel se redujo de **$78.37$** (V1) a **$57.45$** (mediana en V2, media **$56.58$**), situándose con total naturalidad dentro del rango intrínseco del tejido mamario sano de BUS_BRA (**mediana natural: $49.82$, media natural: $68.41$**).
3. **Preservación Anatómica Mejorada**: La ganancia de preservación de parénquima respecto a la caja envolvente (BBox) aumentó de un modesto $12.5\%$ en V1 a un promedio del **$44.72\% \pm 14.8\%$** (mediana **$46.05\%$**), sin presentar ninguna imagen con ganancia nula ($0\%$).

---

## 2. Fundamentación Físico-Matemática de la Solución Acústica

El algoritmo implementado en V2 formula la reconstrucción como la suma desacoplada de dos escalas espaciales:

$$I(x, y) = L(x, y) + S(x, y)$$

1. **Macroestructura de Fondo ($L$)**:
   $$L = G_{\sigma=1.2} * I$$
   Representa las variaciones anatómicas lentas (ecogenicidad de estroma, grasa y márgenes lesionales). Se interpola en la máscara $M$ mediante el transporte de isófotas de Telea:
   $$L_{\text{inp}} = \text{Inpaint}(L, M)$$

2. **Residuo Acústico Microestructural ($S$)**:
   $$S = I - L$$
   Captura las fluctuaciones de interferencia granular (speckle). Al centrarse en $128$ para evitar truncamiento negativo:
   $$S_{\text{shift}} = \text{clip}(S + 128, 0, 255)$$
   $$S_{\text{inp}} = \text{Inpaint}(S_{\text{shift}}, M) - 128$$

3. **Calibración Dinámica de Energía**:
   Para cada componente conectado $C_k \subseteq M$, se calcula la desviación estándar del residuo en el anillo exterior perilesional $\sigma_{\text{ring}}$ y en el interior del parche $\sigma_{\text{patch}}$, modulando la amplitud:
   $$\beta_k = \min\left(1.5, \frac{\sigma_{\text{ring}}}{\sigma_{\text{patch}}}\right)$$
   $$S_{\text{calibrado}}(x, y) = \beta_k \cdot S_{\text{inp}}(x, y) \quad \forall (x, y) \in C_k$$

4. **Fusión Suave (Feathering Gaussian Boundary)**:
   $$\alpha = G_{\sigma=0.5} * \left(\frac{M}{255}\right)$$
   $$I_{\text{final}} = \alpha \cdot \left[ L_{\text{inp}} + S_{\text{calibrado}} \right] + (1 - \alpha) \cdot I$$

---

## 3. Hallazgos Críticos de la Auditoría V2 (Casos Límite y Falsos Positivos)

A pesar del éxito en la eliminación de la costura de ruido blanco, la auditoría clínica detallada de las imágenes de BUS_BRA reveló que el detector morfológico de V2 aún presentaba deficiencias en casos complejos:

| Imagen Auditada | Comportamiento Observado en V2 | Diagnóstico Técnico de la Falla | Impacto Clínico / Algorítmico |
| :--- | :--- | :--- | :--- |
| **`bus_0038-s.png`** | Se detectó el texto inferior pero **se omitió el caliper en el margen izquierdo**. | El caliper cruciforme toca el límite del transductor ($x=0$). El filtro de simetría de cruces ($0.55 \le \text{aspect} \le 1.8$) lo descartó por truncamiento. | El caliper permaneció visible en la imagen final. |
| **`bus_0019-r.png`** | **Falso positivo masivo** en un parche brillante a $y=158$ ($569$ px) y omisión del caliper superior ($y=36$). | Una masa de tejido hiperecoico parenquimatoso normal saturaba a $I \ge 242$. El detector carecía de filtro de grosor bidimensional. El caliper superior se omitió por atenuación. | Se modificó innecesariamente tejido sano ($1.41\%$ cobertura); la ganancia de preservación cayó a solo $12.81\%$. |
| **`bus_0019-l.png`** | Se detectó el caliper izquierdo pero **se omitió el segundo caliper** (superior derecho en $(250, 88)$). | El caliper superior derecho presentaba menor contraste local. Se generaron además specks menores en crestas de tejido normal. | Caliper residual visible; segmentación incompleta de marcas clínicas. |
| **`bus_0018-s.png`** | Texto médico del encabezado y mediciones centrales **cortadas parcialmente a la mitad**. | El kernel de agrupamiento ($13\times 3$) no logró conectar letras separadas por espacios amplios ni números con comas decimales. | Caracteres mutilados que dejan firmas tipográficas parciales. |

---

## 4. Tabla Cuantitativa Completa de Datos Auditables (V2 - 10 Imágenes BUS_BRA)

Datos brutos extraídos directamente de [`resultados_version_2/metricas_cuantitativas_bus_bra.csv`](file:///c:/Users/sebas/OneDrive/Escritorio/PRUEBAINPAINTING/resultados_version_2/metricas_cuantitativas_bus_bra.csv):

| Archivo de Imagen (`BUS_BRA`) | Cobertura Máscara (%) | Ganancia Preservación Tejido (%) | Ratio Varianza Speckle | Discontinuidad Borde Sobel | SSIM Tejido Exterior | GLCM Contrast Diff | Método Evaluado |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `bus_0002-l.png` | $3.22\%$ | $0.00\%$ | $0.7239$ | $39.07$ | $0.9958$ | $0.4802$ | 1. Baseline BBox |
| `bus_0002-l.png` | $1.42\%$ | $55.92\%$ | $0.7842$ | $45.72$ | $0.9894$ | $0.4703$ | 2. Trazo Simple |
| `bus_0002-l.png` | $1.42\%$ | **$55.92\%$** | **$0.8638$** | **$51.34$** | $0.9888$ | $0.4680$ | 3. Híbrido Propuesto (V2) |
| `bus_0003-l.png` | $4.47\%$ | $0.00\%$ | $0.8520$ | $30.66$ | $0.9960$ | $0.4238$ | 1. Baseline BBox |
| `bus_0003-l.png` | $2.02\%$ | $54.78\%$ | $0.7866$ | $48.19$ | $0.9841$ | $0.4125$ | 2. Trazo Simple |
| `bus_0003-l.png` | $2.02\%$ | **$54.78\%$** | **$0.8705$** | **$55.37$** | $0.9832$ | $0.4113$ | 3. Híbrido Propuesto (V2) |
| `bus_0003-r.png` | $0.41\%$ | $0.00\%$ | $0.4409$ | $62.38$ | $0.9997$ | $1.2209$ | 1. Baseline BBox |
| `bus_0003-r.png` | $0.14\%$ | $65.48\%$ | $0.5097$ | $55.38$ | $0.9991$ | $1.3636$ | 2. Trazo Simple |
| `bus_0003-r.png` | $0.14\%$ | **$65.48\%$** | **$0.5602$** | **$58.01$** | $0.9991$ | $1.3680$ | 3. Híbrido Propuesto (V2) |
| `bus_0008-l.png` | $4.06\%$ | $0.00\%$ | $0.8402$ | $40.06$ | $0.9915$ | $0.4642$ | 1. Baseline BBox |
| `bus_0008-l.png` | $2.06\%$ | $49.16\%$ | $0.6757$ | $54.40$ | $0.9833$ | $0.4533$ | 2. Trazo Simple |
| `bus_0008-l.png` | $2.06\%$ | **$49.16\%$** | **$0.7024$** | **$60.32$** | $0.9829$ | $0.4507$ | 3. Híbrido Propuesto (V2) |
| `bus_0018-s.png` | $4.96\%$ | $0.00\%$ | $0.9307$ | $39.34$ | $0.9926$ | $0.0997$ | 1. Baseline BBox |
| `bus_0018-s.png` | $3.32\%$ | $33.14\%$ | $0.8357$ | $44.54$ | $0.9862$ | $0.0776$ | 2. Trazo Simple |
| `bus_0018-s.png` | $3.32\%$ | **$33.14\%$** | **$0.8605$** | **$48.65$** | $0.9853$ | $0.0767$ | 3. Híbrido Propuesto (V2) |
| `bus_0019-l.png` | $5.32\%$ | $0.00\%$ | $0.9043$ | $48.13$ | $0.9938$ | $0.3085$ | 1. Baseline BBox |
| `bus_0019-l.png` | $3.18\%$ | $40.24\%$ | $0.8864$ | $53.38$ | $0.9862$ | $0.2867$ | 2. Trazo Simple |
| `bus_0019-l.png` | $3.18\%$ | **$40.24\%$** | **$0.9336$** | **$56.88$** | $0.9853$ | $0.2846$ | 3. Híbrido Propuesto (V2) |
| `bus_0019-r.png` | $1.62\%$ | $0.00\%$ | $0.7345$ | $62.02$ | $0.9992$ | $0.1638$ | 1. Baseline BBox |
| `bus_0019-r.png` | $1.41\%$ | $12.81\%$ | $0.7489$ | $60.57$ | $0.9989$ | $0.1197$ | 2. Trazo Simple |
| `bus_0019-r.png` | $1.41\%$ | **$12.81\%$** | **$0.7758$** | **$62.67$** | $0.9989$ | $0.1060$ | 3. Híbrido Propuesto (V2) |
| `bus_0020-l.png` | $2.63\%$ | $0.00\%$ | $0.8398$ | $42.40$ | $0.9945$ | $0.4467$ | 1. Baseline BBox |
| `bus_0020-l.png` | $1.50\%$ | $42.93\%$ | $0.7609$ | $53.81$ | $0.9900$ | $0.4265$ | 2. Trazo Simple |
| `bus_0020-l.png` | $1.50\%$ | **$42.93\%$** | **$0.8016$** | **$58.43$** | $0.9892$ | $0.4206$ | 3. Híbrido Propuesto (V2) |
| `bus_0027-l.png` | $3.83\%$ | $0.00\%$ | $0.9929$ | $36.93$ | $0.9920$ | $0.2038$ | 1. Baseline BBox |
| `bus_0027-l.png` | $2.34\%$ | $38.82\%$ | $0.9669$ | $48.38$ | $0.9874$ | $0.1881$ | 2. Trazo Simple |
| `bus_0027-l.png` | $2.34\%$ | **$38.82\%$** | **$1.0024$** | **$50.34$** | $0.9868$ | $0.1873$ | 3. Híbrido Propuesto (V2) |
| `bus_0038-s.png` | $2.69\%$ | $0.00\%$ | $0.8177$ | $20.18$ | $0.9975$ | $5.2629$ | 1. Baseline BBox |
| `bus_0038-s.png` | $1.24\%$ | $53.89\%$ | $0.6856$ | $57.09$ | $0.9899$ | $6.4701$ | 2. Trazo Simple |
| `bus_0038-s.png` | $1.24\%$ | **$53.89\%$** | **$0.7091$** | **$63.81$** | $0.9897$ | $6.4347$ | 3. Híbrido Propuesto (V2) |

---

### Resumen Estadístico Agregado (V2)

| Métrica Auditada | 1. Baseline BBox | 2. Trazo Simple (Liso) | 3. Híbrido Acústico Propuesto (V2) | Estado Metodológico |
| :--- | :---: | :---: | :---: | :--- |
| **Discontinuidad de Borde (Mediana)** | $39.70$ (liso) | $53.60$ | **$57.45$** | **ÉXITO: Normalizado dentro del tejido sano (~50-68)** |
| **Discontinuidad de Borde (Media $\pm$ Std)** | $42.12 \pm 12.9$ | $52.15 \pm 5.2$ | **$56.58 \pm 5.16$** | Eliminada la costura artificial (antes hasta 144) |
| **Ganancia Preservación Tejido (Mediana)** | $0.00\%$ | $46.05\%$ | **$46.05\%$** | Mejora sustancial respecto a V1 ($9.4\%$) |
| **Ganancia Preservación Tejido (Media $\pm$ Std)**| $0.00\%$ | $44.72\% \pm 14.8\%$ | **$44.72\% \pm 14.8\%$** | Cero casos con ganancia nula |
| **Ratio Varianza Speckle (Mediana)** | $0.8400$ | $0.7726$ | **$0.8310$** | Granularidad acústica balanceada |
| **Ratio Varianza Speckle (Media $\pm$ Std)** | $0.8077 \pm 0.15$ | $0.7641 \pm 0.13$ | **$0.8080 \pm 0.13$** | Sin parches homogéneos artificiales |
| **SSIM Tejido Intacto Exterior (Media)** | $0.9953$ | $0.9895$ | **$0.9889$** | Preservación estricta de anatomía exterior |
| **Diferencia de Contraste GLCM (Mediana)** | $0.4352$ | $0.4195$ | **$0.4160$** | Continuidad de textura de co-ocurrencia |

---

## 5. Conclusiones y Necesidad de la Versión 3 (V3)

La **Versión 2** validó con éxito la teoría de la **Descomposición Residual Acústica ($I = L + S$)**, resolviendo de manera categórica el problema del gradiente de borde. Sin embargo, la auditoría identificó la necesidad ineludible de evolucionar el subsistema de detección hacia la **Versión 3 (V3)** para resolver:

1. **Rechazo de Núcleos Sólidos**: Discriminar tejido biológico hiperecoico normal mediante erosión morfológica ($\text{Core} \ge 6$ px) para erradicar falsos positivos como el de `bus_0019-r`.
2. **Aperturas Direccionales Ortogonales y Diagonales**: Detectar cruces de calipers tenues mediante aperturas $5\times 1$, $1\times 5$ y $5\times 5$ para resolver `bus_0019-l` y `bus_0019-r`.
3. **Calipers en Frontera**: Recuperar calipers truncados en márgenes de adquisición ($x \le 2$ o $x \ge w-2$) para resolver `bus_0038-s`.
4. **Fusión Colineal de Texto**: Evitar caracteres cortados a la mitad en encabezados médicos para resolver `bus_0018-s`.
