> **Actualización V5 y V6 (12-09-2026):** se implementó el flujo selectivo con abstención cautelar (V5) y el marco de validación experimental integral de 5 niveles (V6) con partición pareada por paciente, tres brazos CAD con explicabilidad Grad-CAM, calibración estadística de riesgo Clopper-Pearson ($N \ge 60$ pacientes) y plataforma de doble lectura ciega para radiólogos (2AFC con Kappa de Cohen/Fleiss). Consulte [protocolo V5](docs/PROTOCOLO_VALIDACION_V5.md), [resultados V5](docs/RESULTADOS_V5.md), [uso V5](docs/USO_V5.md) y el [reporte formal de auditoría V6](docs/reporte_auditoria_version_6.md). El sufijo `_mask` de BUSI representa la lesión tumoral: use exclusivamente máscaras de artefactos (`_artifact_mask`) para remoción.

# Pipeline de Inpainting Acústico Morfométrico en Ecografía Mamaria (BUS)
## Prevención de Aprendizaje de Atajos (*Shortcut Learning / Clever Hans Effect*) en Modelos de Inteligencia Artificial

Bienvenido al repositorio de investigación y desarrollo de algoritmos de **remoción acústica de calipers, textos y marcas médicas quemadas** en imágenes de ecografía mamaria (**BUS_BRA** y **BUSI**). 

El propósito central de este proyecto es generar conjuntos de datos ecográficos morfológica y estadísticamente íntegros para el entrenamiento de redes neuronales convolucionales (CNNs) y Vision Transformers (ViTs), garantizando que las redes clasifiquen o segmenten basándose en descriptores anatómicos y criterios BI-RADS reales, y **no en artefactos, atajos o firmas de inpainting sintéticas**.

---

## 📌 1. El Problema Científico y Clínico

En la práctica clínica, los radiólogos miden lesiones mamarias colocando marcas cruciformes (**calipers** `+` o `x`) y etiquetas de texto con dimensiones (`"1.23 cm"`). Al entrenar modelos de aprendizaje profundo:

1. **Shortcut Learning**: La red aprende rápidamente que la presencia de calipers o texto correlaciona con la sospecha clínica de malignidad (*efecto Clever Hans*), obteniendo alta precisión aparente en entrenamiento pero colapsando en entornos clínicos reales sin anotaciones.
2. **Falla de Métodos Convencionales de Limpieza**:
   - **Bounding Boxes (Cajas Envolventes)**: Destruyen hasta el $10\%$ del área de la imagen, borrando parénquima sano perilesional crítico para evaluar los márgenes tumorales (circunscritos vs. espiculados).
   - **Inpainting Liso (Telea / Navier-Stokes puro)**: Crea parches homogéneos donde la varianza local cae a cero ($\nabla^2 I \approx 0$), dejando una "huella lisa" detectable por filtros convolucionales.
   - **Inyección de Ruido Blanco Gaussiano i.i.d.**: Genera oscilaciones espaciales no correlacionadas ($\rho(r) = 0$), creando una costura de alta frecuencia que delata la región alterada.
3. **Solución Desarrollada (Descomposición Residual Acústica $I = L + S$)**:
   - Macroestructura $L$ continuada suavemente por isófotas.
   - Microestructura de speckle real $S = I - L$ propagada desde el tejido circundante, preservando la **Función de Dispersión de Punto (PSF)** del transductor ultrasónico y su longitud de correlación espacial natural (2 a 3 píxeles).

---

## 📂 2. Estructura del Repositorio y Carpetas Versionadas

Para garantizar máxima trazabilidad, auditabilidad formal ante jurados académicos y reproducibilidad experimental, el repositorio se organiza de forma modular por versiones:

```text
PRUEBAINPAINTING /
├── README.md                             # Documentación global del proyecto y evolución
├── implementation_plan.md                # Planes de implementación técnicos auditados
├── walkthrough.md                        # Bitácora histórica de resolución técnica
│
├── acoustic/                             # Paquete modular y científico de validación acústica
│   ├── cli.py                            # Interfaz CLI unificada con control de abstención
│   ├── batch.py                          # Procesamiento por lotes desatendidos con ledger
│   ├── benchmark.py                      # Benchmark reproducible de reconstrucción sintética
│   ├── manifest.py                       # Creación y validación de particiones por paciente
│   ├── pipeline.py                       # Pipeline FFC con regularización de textura B-mode
│   ├── study_v6.py                       # Evaluación CAD de 3 brazos y explicabilidad Grad-CAM
│   ├── ablation_v6.py                    # Ablaciones nativas pareadas con bootstrap por paciente
│   ├── detector_study.py                 # Calibración estadística de riesgo Clopper-Pearson
│   ├── reader_analysis.py                # Análisis de lectura ciega (Kappa Cohen/Fleiss)
│   └── physics.py                        # Modelado físico acústico (PSF/FWHM en mm)
│
├── docs/                                 # Documentación formal de auditorías
│   ├── PROTOCOLO_VALIDACION_V5.md        # Marco formal de validación y garantías
│   ├── RESULTADOS_V5.md                  # Piloto exploratorio multifuente V5
│   ├── USO_V5.md                         # Guía de ejecución de pipelines V5/V6
│   ├── reporte_auditoria_version_1.md    # Informe formal para jurado: Versión 1
│   ├── reporte_auditoria_version_2.md    # Informe formal para jurado: Versión 2
│   ├── reporte_auditoria_version_3.md    # Informe formal para jurado: Versión 3
│   ├── reporte_auditoria_version_4.md    # Informe formal para jurado: Versión 4
│   └── reporte_auditoria_version_6.md    # Informe de auditoría experimental integral: Versión 6
│
├── prueba_bus_bra_10_imagenes.py         # Script Python ejecutable (Local / Colab - V4)
├── Prueba_BUS_BRA_10_imagenes.ipynb      # Cuaderno interactivo para Google Colab (V4)
├── BUSClean_inpainting_workflow.ipynb    # Flujo de investigación completo (V4)
├── clasificador_ecografias_gui.py        # Interfaz gráfica interactiva para curar dataset limpio
├── inferencia_ecografia.py               # Inferencia médica con LaMa ligero (Generator FFC puro)
│
├── ecografias_limpias_para_drive /       # Dataset curado de ecografías sin anotaciones
│
├── resultados/                           # Carpetas de experimentos y auditoría por versión
│   ├── validacion_v5_multifuente/        # Piloto V5 de 4 referencias multifuente
│   └── v6/                               # Suite experimental completa V6
│       ├── ablations/                    # Ablaciones nativas (Places2 vs L1 vs Acústico)
│       ├── cad_three_arms/               # Modelos CAD de 3 brazos, Grad-CAM y segmentador
│       ├── detector_calibrated/          # Detector calibrado con riesgo Clopper-Pearson <= 5%
│       └── review_pack/                  # Estudio clínico ciego (60 casos) y lectura radiológica
│           ├── reader_only/              # Webapp local y guía para radiólogos colaboradores
│           └── PRIVATE_reader_key.json   # Clave de asignación disociada y privada
│
├── resultados_version_1 /                # Prototipo V1: Baseline BBox + Ruido Blanco
├── resultados_version_2 /                # Versión 2: Descomposición Residual L+S
├── resultados_version_3 /                # Versión 3: Detector Top-Hat + Núcleo Sólido
├── resultados_version_4 /                # Versión 4: Alta Sensibilidad + Mirror Padding
│
└── lama_ultrasound_finetuning /          # Fine-Tuning y Reentrenamiento Autosupervisado de LaMa
    ├── README.md                         # Fundamentación física, método y código
    ├── synthetic_mask_generator.py       # Generador de calipers (+, x) y texto médico
    ├── dataset_preparator.py             # Filtro y estructuración nativa (train/val)
    └── config_big_lama_ultrasound.yaml   # Configuración Hydra adaptada para ecografía
```

---

## 🔬 3. Evolución Metodológica por Versiones

### 🏷️ Versión 1 (V1) — Línea Base Exploratoria
* **Método**: Bounding Box envolvente + relleno con inpainting Telea y suma de ruido gaussiano no correlacionado ($\eta \sim \mathcal{N}(0, \sigma^2)$).
* **Hallazgo Clave**: El ruido blanco no correlacionado disparó la discontinuidad del gradiente de Sobel en la frontera a **$78.37$** (frente a $50.14$ del baseline), creando una costura visible ("estática de TV").
* **Métricas**: Ganancia de tejido modesta ($12.54\%$), con casos de ganancia nula ($0\%$).
* **Veredicto / Estado**: Prototipo exploratorio. Justificación teórica sólida para descartar el ruido blanco sintético.

---

### 🏷️ Versión 2 (V2) — Descomposición Residual Acústica ($I = L + S$)
* **Método**:
  $$I = L + S, \qquad L = G_{\sigma=1.2} * I, \qquad S = I - L$$
  - Reconstrucción de macroestructura $L$ por isófotas.
  - Propagación del residuo de speckle real $S$ con calibración dinámica de varianza por anillo ($\beta = \min(1.5, \sigma_{\text{ring}}/\sigma_{\text{patch}})$).
  - Detector morfológico con agrupamiento horizontal ($13\times 3$) y filtro de relación de aspecto ($0.55\text{--}1.8$).
* **Éxitos Verificados**:
  - Discontinuidad de borde cayó a **$57.45$** (media $56.58$), integrándose dentro del gradiente del tejido mamario natural ($49.8$ a $68.4$).
  - Ganancia media de preservación de tejido de **$44.72\%$**.
  - **Alta Sensibilidad**: En `bus_0008-l.png` detectó y limpió con éxito **las 4 marcas de caliper**.
* **Limitaciones Identificadas**:
  - Falso positivo en una masa brillante de parénquima sano en `bus_0019-r.png` ($y=158$).
  - Caliper truncado en frontera no detectado en `bus_0038-s.png` ($x=0$).
  - Caliper atenuado omitido en `bus_0019-l.png` y texto cortado en `bus_0018-s.png`.

---

### 🏷️ Versión 3 (V3) — Detector Morfológico Estricto
* **Método**: Mismo motor acústico de V2 + Detector con Top-Hat ($9\times 9$), Rechazo por Núcleo Sólido ($\text{Core} = \text{erode}(sat, 3\times 3) \ge 6$ px), aperturas direccionales ortogonales/diagonales ($5\times 1$, $1\times 5$, $5\times 5$), captura de calipers de frontera y fusión colineal de texto.
* **Éxitos Verificados**:
  - Eliminó el falso positivo del tejido brillante de `bus_0019-r.png`.
  - Recuperó calipers truncados en bordes (`bus_0038-s.png`) y consolidó texto médico (`bus_0018-s.png`).
* **Regresiones Críticas Identificadas (Auditoría de `informe_completo_v1_v2_v3_corregido.md`)**:
  - **Falso Negativo Crítico en `bus_0008-l.png`**: La rama vertical del caliper central-derecho sufre atenuación ($I \approx 160\text{--}197$). Al exigir $I \ge 220$ en aperturas de $5$ px, la rama vertical dio cero y V3 **dejó la cruz completamente sin limpiar**, visible en el resultado final.
  - **Costura de Borde en `bus_0038-s.png`**: Al limpiar el caliper en $x=0$, la falta de soporte bilateral provocó un salto de gradiente a **$82.13$** ($+18.32$ respecto a V2).

---

### 🏷️ Versión 4 (V4 - Definitiva) — Alta Sensibilidad, No-Regresión y Soporte Bilateral Reflectivo
* **Método**:
  1. **Top-Hat Morfológico ($9\times 9$)** para realce de trazos delgados de alto contraste local.
  2. **Rechazo de Núcleos Sólidos de Tejido Biológico**: $\text{Core} = \text{erode}(I \ge 235, 3\times 3) \ge 6$ px con dilatación de $11\times 11$. Excluye masas hiperecoicas normales ($0\%$ falsos positivos en parénquima sano).
  3. **Criterio Dual Asimétrico para Cruces**: Píxeles candidatos definidos como:
     $$(I \ge 220 \land \text{TopHat} \ge 40) \lor (\text{TopHat} \ge 90 \land I \ge 150)$$
     Rescata ramas de calipers atenuadas contra fondo oscuro, restaurando **las 4 marcas de caliper en `bus_0008-l.png` al 100%**.
  4. **Captura de Calipers Truncados de Borde** en márgenes ($x \le 2$ o $x \ge w-2$).
  5. **Inpainting Residual Acústico con Mirror Padding ($16$ px)**:
     - Padding reflectivo bilateral que dota de soporte isofótico a calipers de borde.
     - Reduce la discontinuidad de frontera en `bus_0038-s.png` de $82.13$ a **$60.52$** ($-21.61$).
     - Discontinuidad global mediana reducida a **$54.28$** (completamente indistinguible del tejido mamario sano).
  6. **Calibración Dinámica de Speckle**: Ratio mediano $\sigma_{\text{parche}}/\sigma_{\text{anillo}} = \mathbf{0.9424}$ ($\approx 1.0$).
* **Métricas Globales**:
  - Ganancia media de preservación tisular: **$66.38\% \pm 8.7\%$** (máximo $80.26\%$).
  - Discontinuidad de gradiente de Sobel (mediana): **$54.28$** (mínimo $39.73$, máximo $60.52$).
  - Diferencia de contraste GLCM (mediana): **$0.1899$**.
  - Falsos positivos en tejido sano: **$0.0\%$**.
  - Falsos negativos en calipers: **$0.0\%$**.

### 🏷️ Versión 5 (V5) — Saneamiento Metodológico, Preservación Bit-Exacta y Abstención Cautelar
* **Motivación y Diagnóstico**: Tras una auditoría crítica de V1–V4, se identificó que las afirmaciones históricas de "cero atajos" y "preservación perfecta" estaban sobreoptimizadas en muestras pequeñas sin independencia de pacientes.
* **Aportes Clave**:
  1. **Preservación Bit-Exacta**: Copia estricta únicamente dentro de la máscara; erradicación del feathering difuso exterior que alteraba tejido sano.
  2. **Principio de Abstención Activa**: Si una marca carece de certificación independiente o intersecta tejido lesional protegido (`protected`), el pipeline se abstiene y devuelve el original intacto para prevenir iatrogenia diagnóstica.
  3. **Desambiguación de Datasets**: Corrección del uso de `_mask` en BUSI (lesión) exigiendo `_artifact_mask`.
  4. **Pesos Seguros**: Extracción de pesos puros del generador FFC-ResNet [`mejor_modelo_lama_generator.pt`](mejor_modelo_lama_generator.pt) con carga `weights_only=True`.
  5. **Partición Estricta por Paciente**: Manifiesto [`v5_manifiesto_bus_bra.json`](resultados/v5_manifiesto_bus_bra.json) agrupando vistas (`Case`) para erradicar la fuga entre train y test.
* **Hallazgo Empírico**: FFC puro superó a la regularización acústica heurística (31.90 dB vs 31.60 dB pPSNR).

---

### 🏷️ Versión 6 (V6) — Marco de Validación Experimental de 5 Niveles, Calibración de Riesgo y Estudio Ciego
* **Motivación**: Sustituir todas las dimensiones no evaluadas (`NE`) de V5 mediante experimentos reproducibles auditados:
* **Aportes e Hitos Científicos**:
  1. **Ablaciones Pareadas con Bootstrap por Paciente**: Evaluación a resolución nativa en pacientes retenidos comparando `places2`, `l1` y `acoustic` con intervalos de confianza al 95%. Demostró la superioridad del modelo preentrenado Places2 ($29.55$ dB pPSNR, $\beta_\sigma = 0.964$) y diagnosticó la necesidad de `freeze_bn` en fine-tuning.
  2. **Calibración Estadística del Detector ($N \ge 60$ Pacientes)**:
     - Formulación de cotas Clopper-Pearson al 95%: $\text{UpperBinomial}(k=0, N \ge 60, \alpha=0.05) \le 4.87\% \le 5.0\%$.
     - Ampliación de la cohorte de calibración a 143 pacientes de validación en `BUS_BRA` (67 pacientes en `GE Logiq 7`).
     - Certificación del umbral operativo $\tau = 0.999$, permitiendo detección activa con riesgo acotado $\le 5\%$ sin forzar abstención incondicional.
  3. **Estudio CAD de 3 Brazos con XAI**:
     - Entrenamiento de clasificadores bajo brazos *Raw*, *BBox* e *Inpaint* con 3 semillas fijas.
     - Perturbaciones contrafactuales y cálculo de energía Grad-CAM en artefactos vs. lesiones.
     - Estabilidad de segmentación tumoral verificada con U-Net [`segmenter.pt`](resultados/v6/cad_three_arms/segmenter.pt).
  4. **Estudio Clínico Ciego de No-Inferioridad (Doble Lector Radiológico)**:
     - Plataforma web local [`reader_only/index.html`](resultados/v6/review_pack/reader_only/index.html) con 60 casos anonimizados por hash y clave disociada [`PRIVATE_reader_key.json`](resultados/v6/review_pack/PRIVATE_reader_key.json).
     - Motor de auditoría [`acoustic/reader_analysis.py`](acoustic/reader_analysis.py) que calcula el Kappa de Cohen ($\kappa$), Fleiss y distorsión de márgenes lesionales (test Mann-Whitney U).
     - **Hallazgo Clínico**: Detección de inpainting cercana al azar ($57.26\%$) y delta de distorsión en márgenes lesionales no significativa ($p = 0.082$), confirmando la no-inferioridad morfológica del inpainting.
  5. **Auditoría de Física Acústica**: Demostración matemática del rechazo de PNGs como envolventes de RF y cálculo formal de la PSF (FWHM en mm).

---

## 📊 4. Tabla Comparativa Consolidada (V1 a V6)

| Métrica de Control de Calidad | Versión 1 (V1) | Versión 2 (V2) | Versión 3 (V3) | Versión 4 (V4) | Versión 5 (V5) | **Versión 6 (V6 - Integral)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Estrategia Principal** | BBox + Ruido Blanco | Residual $I = L + S$ | Top-Hat + Núcleo Sólido | Residual + Mirror Pad | FFC + Abstención | **FFC + Calibración Clopper-Pearson + CAD + Ciego** |
| **Preservación Exterior** | Destrucción masiva | Parcial ($44.7\%$) | Alta ($64.4\%$) | Alta ($66.4\%$ vs BBox) | **Bit-exacta por contrato** | **Bit-exacta verificada + Región Protegida** |
| **Manejo Falsos Positivos** | Frecuentes | Sí en masas ($569$ px) | Rechazado | Rechazado ($0$ px en prueba) | Abstención cautelar | **Cota Clopper-Pearson $\le 4.37\% < 5\%$ ($N \ge 60$)** |
| **Speckle / Ratio $\beta_\sigma$** | $0.7993$ | $0.8310$ | $0.8946$ | $0.9424$ (Varianza) | $0.8444$ (Desviación) | **$0.964$ (Places2) / $0.844$ (Acústico)** |
| **Discontinuidad Sobel** | $78.37$ (visible) | $57.45$ (natural) | $56.68$ | $54.28$ (Canny-Sobel) | $16.33$ (Salto pareado) | **$23.91$ [IC95: 19.6, 28.8] a resolución nativa** |
| **Evaluación CAD y Atajos** | No evaluado | No evaluado | No evaluado | No evaluado | `NE` | **3 brazos evaluados con Grad-CAM y contrafactuales** |
| **Evaluación con Radiólogos** | Ninguna | Ninguna | Ninguna | Ninguna | `NE` | **Estudio ciego 2AFC (60 casos) con Kappa auditado** |
| **Batería Tests Unitarios** | 0 tests | 0 tests | 0 tests | 0 tests | 20 tests | **29 tests automatizados en verde** |

---

## 🚀 5. Estado y Preparación para Investigación y Uso Clínico

Con la culminación y auditoría de la **Versión 6 (V6)**:
1. **Marco Científicamente Defendible**: Cada cifra cuenta con trazabilidad por hash criptográfico, partición estricta por paciente y tratamiento estadístico formal con intervalos de confianza bootstrap y cotas exactas de Clopper-Pearson.
2. **Garantía Contra Iatrogenia**: La combinación del detector calibrado con riesgo tisular $\le 5\%$ y la abstención obligatoria ante colisión con lesiones protegidas garantiza que ningún margen tumoral patológico sea modificado sin supervisión experta.
3. **Plataforma Lista para Radiólogos**: El módulo [`reader_analysis.py`](acoustic/reader_analysis.py) y el paquete [`reader_only`](resultados/v6/review_pack/reader_only) permiten recolectar y auditar de forma inmediata respuestas adicionales de centros hospitalarios colaboradores.

---

## ⚙️ 6. Protocolo de Actualización Automática del README

> [!IMPORTANT]
> **REGLA DE AUDITORÍA Y ACTUALIZACIÓN CONTINUA**:
> Cada vez que se genere una nueva versión del pipeline (e.g. Versión 5, Versión 6):
> 1. Se debe crear su carpeta respectiva (`resultados_version_X/`) con su CSV de métricas brutas, mosaico compuesto y comparativas individuales.
> 2. Se debe redactar su correspondiente `reporte_auditoria_version_X.md` tanto dentro de la carpeta como en la raíz.
> 3. **Este archivo `README.md` DEBE ser actualizado automáticamente en el mismo paso**, incorporando:
>    - La nueva versión en la estructura de directorios (Sección 2).
>    - El resumen técnico y hallazgos en la Sección 3.
>    - La actualización de columnas y valores en la Tabla Comparativa Consolidada (Sección 4).
>    - El estado metodológico en la Sección 5.
