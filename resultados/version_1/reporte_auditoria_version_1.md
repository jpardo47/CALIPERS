# Informe de Auditoría y Validación Cuantitativa: Versión 1 (V1)
## Línea Base Exploratoria: Bounding Box Convencional e Inyección de Ruido Blanco

**Autor:** Equipo de Investigación en Visión Artificial y Ultrasonido  
**Fecha de Evaluación Inicial:** 6 de Septiembre de 2026  
**Dataset de Auditoría V1:** 14 Imágenes ecográficas de prueba (Banco Benchmark Inicial)  
**Archivos Auditables Asociados:**
- Archivo de métricas brutas: [`inpaint_quantitative_qc_benchmark_v1.csv`](file:///c:/Users/sebas/OneDrive/Escritorio/PRUEBAINPAINTING/resultados_version_1/inpaint_quantitative_qc_benchmark_v1.csv)
- Registro visual compuesto: [`PRUEBA-BUS_BRA_v1.png`](file:///c:/Users/sebas/OneDrive/Escritorio/PRUEBAINPAINTING/resultados_version_1/PRUEBA-BUS_BRA_v1.png)
- Comparativas de 10 imágenes: `comparacion_bus_0002-l.png` a `comparacion_bus_0038-s.png`

---

## 1. Resumen Ejecutivo para el Jurado

La **Versión 1 (V1)** representó el prototipo inicial del proyecto orientado a eliminar artefactos (calipers y texto) en imágenes de ecografía mamaria para evitar el aprendizaje de atajos (*shortcut learning*).

En esta primera etapa se contrastaron dos enfoques elementales:
1. **Línea Base Convencional (Bounding Box + Inpainting Telea)**: Restauración de regiones rectangulares completas delimitadas por cajas envolventes.
2. **Prototipo Híbrido V1 (Trazo Delgado + Ruido Blanco Gaussiano Sintético)**: Extracción de trazo fino sumando ruido gaussiano i.i.d. $\eta \sim \mathcal{N}(0, \sigma^2)$ sobre el parche reconstruido para mitigar el alisamiento artificial.

### Hallazgo Crítico de la Auditoría V1
La evaluación estadística automatizada reveló una anomalía determinante: **el prototipo con ruido blanco presentó MAYOR discontinuidad de borde que la línea base lisa en 13 de las 14 imágenes evaluadas**, alcanzando una mediana de discontinuidad de Sobel de **$78.37$** (frente a **$50.14$** del baseline), y picos de hasta **$113$ a $144$** en ecografías de BUS_BRA. 

Este resultado evidenció que la adición ingenua de ruido blanco genera una "costura" de alta frecuencia fácilmente explotable por redes convolucionales como firma sintética.

---

## 2. Fundamentación Físico-Matemática del Fallo en V1

### 2.1 La Naturaleza del Ruido Blanco vs. Speckle Ultrasónico Real
* **Ruido Blanco Gaussiano**: Las muestras sintéticas inyectadas son estadísticamente independientes e idénticamente distribuidas (i.i.d.):
  $$\mathbb{E}[\eta(x, y) \cdot \eta(x + \Delta x, y + \Delta y)] = \sigma^2 \delta(\Delta x, \Delta y)$$
  Su función de autocorrelación espacial es una delta de Dirac ($\rho(r) = 0$ para $r \ne 0$), lo que implica densidad espectral de potencia constante en todas las frecuencias espaciales.
* **Speckle Acústico Real**: El speckle en ultrasonido médico no es ruido aditivo independiente, sino el patrón de interferencia determinista de ecos dispersos sub-resolución convolucionados con la **Función de Dispersión de Punto (PSF)** del transductor piezoeléctrico. Por tanto, posee una longitud de correlación espacial finita no nula:
  $$r_{\text{corr}} \approx 2 \text{ a } 3 \text{ píxeles}$$

### 2.2 Respuesta del Operador de Gradiente (Sobel)
El operador de Sobel $S_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}$ actúa matemáticamente como un filtro pasa-altas. 

En la frontera entre el tejido biológico real (suavemente correlacionado) y el parche restaurado con ruido blanco (oscilaciones no correlacionadas píxel a píxel), el gradiente medio de Sobel se dispara artificialmente:

$$\|\nabla I_{\text{V1}}\|_{\text{borde}} = \sqrt{(\partial_x I)^2 + (\partial_y I)^2} \longrightarrow \mathbf{78.37} \text{ (Mediana V1)}$$

Lejos de camuflar el inpainting, el ruido blanco creó una firma estática ("ruido de sintonizador de televisión") en los bordes de cada caliper y carácter.

---

## 3. Limitaciones del Módulo de Detección en V1

1. **Invasividad del Bounding Box**:
   - Al emplear cajas envolventes rectangulares completas, se destruyó parénquima sano en un área hasta 4 veces superior al grosor real del caliper.
   - En casos como `malignant (101).png` o `benign (102).png`, la ganancia de preservación de tejido fue del **$0.00\%$**.
2. **Umbral Global No Adaptativo**:
   - El detector V1 aplicaba un umbral de corte estático ($I \ge 220$), careciendo de filtros morfológicos de espesor, lo que generaba detecciones accidentales en tejido hiperecoico normal.
3. **Fragmentación de Cadenas de Texto**:
   - Sin un mecanismo de agrupamiento colineal, los números y letras de tamaño pequeño eran cortados a la mitad o descartados como ruido puntual.

---

## 4. Tabla Cuantitativa Completa de Datos Auditables (V1)

Datos brutos extraídos directamente de [`inpaint_quantitative_qc_benchmark_v1.csv`](file:///c:/Users/sebas/OneDrive/Escritorio/PRUEBAINPAINTING/resultados_version_1/inpaint_quantitative_qc_benchmark_v1.csv):

| Archivo de Imagen | Cobertura Máscara (%) | Ganancia Preservación Tejido (%) | Ratio Varianza Speckle | Discontinuidad Borde Sobel | SSIM Tejido Exterior | Método Evaluado |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `benign (1).png` | $3.70\%$ | $0.00\%$ | $1.0607$ | $50.36$ | $0.9896$ | 1. Baseline BBox |
| `benign (1).png` | $3.34\%$ | $9.63\%$ | $1.1152$ | $46.87$ | $0.9880$ | 2. Trazo Simple |
| `benign (1).png` | $3.34\%$ | $9.63\%$ | **$1.1786$** | **$77.33$** | $0.9859$ | 3. Híbrido V1 (Ruido Blanco) |
| `benign (10).png` | $0.46\%$ | $0.00\%$ | $0.6769$ | $54.17$ | $0.9988$ | 1. Baseline BBox |
| `benign (10).png` | $0.44\%$ | $3.86\%$ | $0.6605$ | $46.82$ | $0.9986$ | 2. Trazo Simple |
| `benign (10).png` | $0.44\%$ | $3.86\%$ | $0.8543$ | **$80.44$** | $0.9981$ | 3. Híbrido V1 (Ruido Blanco) |
| `benign (100).png` | $1.28\%$ | $0.00\%$ | $0.8106$ | $37.95$ | $0.9976$ | 1. Baseline BBox |
| `benign (100).png` | $1.13\%$ | $11.50\%$ | $0.8529$ | $35.60$ | $0.9969$ | 2. Trazo Simple |
| `benign (100).png` | $1.13\%$ | $11.50\%$ | $0.9743$ | **$51.88$** | $0.9960$ | 3. Híbrido V1 (Ruido Blanco) |
| `benign (101).png` | $1.30\%$ | $0.00\%$ | $0.7050$ | $60.17$ | $0.9961$ | 1. Baseline BBox |
| `benign (101).png` | $1.22\%$ | $6.28\%$ | $0.7779$ | $52.77$ | $0.9957$ | 2. Trazo Simple |
| `benign (101).png` | $1.22\%$ | $6.28\%$ | $0.8989$ | **$81.50$** | $0.9949$ | 3. Híbrido V1 (Ruido Blanco) |
| `benign (102).png` | $0.01\%$ | $0.00\%$ | $0.1946$ | $135.91$ | $0.9999$ | 1. Baseline BBox |
| `benign (102).png` | $0.01\%$ | $0.00\%$ | $0.2297$ | $99.94$ | $0.9999$ | 2. Trazo Simple |
| `benign (102).png` | $0.01\%$ | $0.00\%$ | $0.9076$ | **$181.71$** | $0.9999$ | 3. Híbrido V1 (Ruido Blanco) |
| `benign (103).png` | $0.73\%$ | $0.00\%$ | $1.1405$ | $48.59$ | $0.9992$ | 1. Baseline BBox |
| `benign (103).png` | $0.49\%$ | $33.10\%$ | $0.6134$ | $58.56$ | $0.9989$ | 2. Trazo Simple |
| `benign (103).png` | $0.49\%$ | $33.10\%$ | $0.9679$ | **$109.09$** | $0.9983$ | 3. Híbrido V1 (Ruido Blanco) |
| `benign (104).png` | $1.33\%$ | $0.00\%$ | $0.9989$ | $45.42$ | $0.9979$ | 1. Baseline BBox |
| `benign (104).png` | $1.03\%$ | $22.96\%$ | $0.7086$ | $44.72$ | $0.9974$ | 2. Trazo Simple |
| `benign (104).png` | $1.03\%$ | $22.96\%$ | $0.9397$ | **$79.41$** | $0.9962$ | 3. Híbrido V1 (Ruido Blanco) |
| `benign (105).png` | $0.79\%$ | $0.00\%$ | $0.6655$ | $44.56$ | $0.9987$ | 1. Baseline BBox |
| `benign (105).png` | $0.62\%$ | $21.33\%$ | $0.6329$ | $43.51$ | $0.9983$ | 2. Trazo Simple |
| `benign (105).png` | $0.62\%$ | $21.33\%$ | $0.8567$ | **$69.42$** | $0.9978$ | 3. Híbrido V1 (Ruido Blanco) |
| `malignant (1).png` | $0.37\%$ | $0.00\%$ | $1.1447$ | $83.11$ | $0.9986$ | 1. Baseline BBox |
| `malignant (1).png` | $0.37\%$ | $1.99\%$ | $0.7332$ | $41.21$ | $0.9990$ | 2. Trazo Simple |
| `malignant (1).png` | $0.37\%$ | $1.99\%$ | $0.9407$ | **$72.54$** | $0.9986$ | 3. Híbrido V1 (Ruido Blanco) |
| `malignant (10).png` | $1.78\%$ | $0.00\%$ | $0.7281$ | $102.40$ | $0.9941$ | 1. Baseline BBox |
| `malignant (10).png` | $1.61\%$ | $9.12\%$ | $0.8589$ | $88.95$ | $0.9921$ | 2. Trazo Simple |
| `malignant (10).png` | $1.61\%$ | $9.12\%$ | $0.9766$ | **$118.68$** | $0.9918$ | 3. Híbrido V1 (Ruido Blanco) |
| `malignant (100).png`| $2.70\%$ | $0.00\%$ | $0.9018$ | $49.93$ | $0.9967$ | 1. Baseline BBox |
| `malignant (100).png`| $1.70\%$ | $36.84\%$ | $0.8950$ | $44.71$ | $0.9947$ | 2. Trazo Simple |
| `malignant (100).png`| $1.70\%$ | $36.84\%$ | $1.0510$ | **$67.27$** | $0.9941$ | 3. Híbrido V1 (Ruido Blanco) |
| `malignant (101).png`| $0.06\%$ | $0.00\%$ | $0.5770$ | $117.46$ | $0.9997$ | 1. Baseline BBox |
| `malignant (101).png`| $0.07\%$ | $0.00\%$ | $0.7377$ | $91.77$ | $0.9996$ | 2. Trazo Simple |
| `malignant (101).png`| $0.07\%$ | $0.00\%$ | $0.9647$ | **$126.66$** | $0.9996$ | 3. Híbrido V1 (Ruido Blanco) |
| `malignant (103).png`| $2.83\%$ | $0.00\%$ | $0.7880$ | $35.95$ | $0.9948$ | 1. Baseline BBox |
| `malignant (103).png`| $2.42\%$ | $14.61\%$ | $0.8089$ | $33.21$ | $0.9940$ | 2. Trazo Simple |
| `malignant (103).png`| $2.42\%$ | $14.61\%$ | $0.9479$ | **$49.06$** | $0.9924$ | 3. Híbrido V1 (Ruido Blanco) |
| `malignant (104).png`| $1.18\%$ | $0.00\%$ | $0.8209$ | $42.63$ | $0.9969$ | 1. Baseline BBox |
| `malignant (104).png`| $1.13\%$ | $4.32\%$ | $0.9440$ | $37.89$ | $0.9963$ | 2. Trazo Simple |
| `malignant (104).png`| $1.13\%$ | $4.32\%$ | $1.0000$ | **$52.74$** | $0.9960$ | 3. Híbrido V1 (Ruido Blanco) |

---

### Resumen Estadístico Agregado (V1)

| Métrica Auditada | 1. Baseline BBox | 2. Trazo Simple (Liso) | 3. Híbrido V1 (Ruido Blanco) | Evaluación de Impacto |
| :--- | :---: | :---: | :---: | :--- |
| **Discontinuidad de Borde (Mediana)** | **$50.14$** | **$45.77$** | **$78.37$** | **RECHAZADO: Empeora $+56.3\%$ por ruido blanco** |
| **Discontinuidad de Borde (Media $\pm$ Std)** | $64.90 \pm 32.5$ | $54.75 \pm 22.1$ | **$86.98 \pm 36.4$** | Picos severos en imágenes de alta atenuación |
| **Ganancia de Tejido Preservado (Mediana)** | $0.00\%$ | $9.38\%$ | $9.38\%$ | Ganancia baja debido a segmentación imprecisa |
| **Ganancia de Tejido Preservado (Media)** | $0.00\%$ | $12.54\%$ | $12.54\%$ | Casos frecuentes con ganancia nula ($0.0\%$) |
| **Ratio de Varianza de Speckle (Mediana)** | $0.7993$ | $0.7578$ | $0.9563$ | Falsa mejora: la varianza subió a costa de estática |
| **SSIM Tejido Intacto Exterior (Media)** | $0.9970$ | $0.9964$ | $0.9957$ | Preservación anatómica exterior aceptable |

---

## 5. Conclusiones de la Auditoría V1 y Lecciones Aprendidas

1. **Inviabilidad del Ruido Blanco Sintético**: Inyectar ruido gaussiano $\mathcal{N}(0, \sigma^2)$ destruye la continuidad espectral y la correlación física del transductor ecográfico. Provoca un aumento masivo de la discontinuidad de Sobel en frontera ($78.37$ vs. $50.14$), creando una costura artificial que actúa como atajo en modelos profundos.
2. **Necesidad de Descomposición Multiescala**: Para preservar el speckle sin saltos de frontera, el residuo acústico debe extraerse del propio tejido circundante (donde ya incorpora la PSF del transductor) en lugar de ser generado aleatoriamente.
3. **Necesidad de Agrupamiento Morfológico**: El detector debe incorporar filtros morfológicos direccionales para conectar palabras y números, además de distinguir cruces de calipers de ecos parenquimatosos normales.

*Estas lecciones motivaron el desarrollo de la **Versión 2 (V2)** mediante Descomposición Residual Acústica ($I = L + S$).*
