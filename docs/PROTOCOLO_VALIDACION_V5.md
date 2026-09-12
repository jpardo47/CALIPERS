# Pipeline selectivo FFC + regularización de textura B-mode

Estado: implementación experimental con abstención. No constituye validación clínica ni demuestra erradicación de shortcut learning. Las cifras de los informes V1–V4 permanecen como antecedentes históricos, no como resultados nuevos certificados. La tabla sugerida por el autor contiene objetivos/ejemplos que deben sustituirse por mediciones.

## Hallazgos corregidos

1. La inferencia anterior inventaba una máscara rectangular cuando faltaba la máscara. Ahora falla explícitamente; el flujo selectivo propone candidatos y se abstiene sin soporte independiente.
2. La salida V4 difundía el feathering fuera de la máscara y convertía toda la imagen a gris. Ahora copia al original únicamente los píxeles enmascarados, conservando el exterior y sus canales.
3. `_mask` en BUSI suele representar la lesión. El lote de inferencia ahora busca `_artifact_mask`; jamás se debe entregar la segmentación tumoral como máscara de borrado.
4. Pesos incompletos se aceptaban mediante `strict=False`, y módulos ausentes se sustituían por funciones vacías. Ahora se cargan tensores con `weights_only=True`, coincidencia estricta y dependencias de entrenamiento importadas solo cuando se usan. Un checkpoint Lightning con objetos no permitidos puede rechazarse: use el generador `.pt` existente.
5. Separar imágenes aleatoriamente puede compartir pacientes entre train/test. El manifiesto usa `Case` en BUS_BRA y verifica duplicados por píxeles decodificados.
6. Brillo bajo no prueba ausencia de marcas. La preparación nueva exige `clean_verified=true` respaldado por revisión independiente y conserva resolución nativa.

## Arquitectura y garantías implementadas

Imagen original → detector de candidatos de trazos → soporte independiente de artefactos y exclusiones anatómicas → candidato FFC → ajuste acotado de residuo local → controles por componente → copia estrictamente dentro de la máscara → PNG y auditoría.

Sin `--weights`, el backend está identificado como Telea acústico, no como FFC. La red existente es big-LaMa con 18 bloques y ramas globales FFC. El regularizador descompone la propuesta en baja frecuencia y residuo; incrementa como máximo 1,5 veces la amplitud del residuo si está deprimida frente al anillo. No añade ruido blanco. Esta modificación es una hipótesis experimental que debe superar a FFC puro en ablación; no se presupone que mejore cada imagen.

El detector morfológico propone cruces, líneas y texto brillante; sus scores no son probabilidades calibradas. No resuelve universalmente texto coloreado, antialiasing débil ni estructuras hiperecoicas parecidas a cruces. Por ello, **no se permite que su salida autorice automáticamente edición anatómica**.

`certified_artifact` constituye una frontera de confianza: es una anotación independiente y exacta del trazo, o una salida de un sistema externo validado. Entregar la misma máscara heurística como “certificada” elude la garantía y no valida el detector. Si se necesita remoción totalmente autónoma sin esa información, el requisito absoluto pFPR=0 no está resuelto. La abstención preserva el original, pero tampoco cuenta como remoción exitosa.

`protected` admite lesión, quiste u otra región protegida, ampliada tres píxeles por defecto. Una intersección rechaza el caso completo. No se recorta silenciosamente una cruz situada en el margen tumoral. Los casos de cruces sobre lesiones solo pueden investigarse en el benchmark sin aprobación de uso clínico.

Garantías verificables: misma geometría, mismo tipo uint8, máscara binaria exacta, ausencia de redimensionamiento implícito, conservación bit a bit del exterior, rechazo de salida no finita, no sobrescritura de entradas, guardado PNG comprobado por relectura. Ninguna de estas garantiza que el interior de una máscara errónea sea tejido no alterado.

## Métricas y correcciones estadísticas

| Métrica | Definición implementada y cautela |
|---|---|
| pPSNR | MSE exclusivamente en el soporte común de la marca sintética; rango 255. Reconstrucción perfecta se registra con bandera, sin Infinity en JSON. |
| pSSIM | Media del mapa SSIM en centros dentro de la máscara; ventana gaussiana sigma=1,5, rango 255. La ventana usa contexto externo y no equivale a SSIM de píxeles aislados. |
| pFPR | Píxeles cambiados en tejido fuera de GT de artefactos / tejido fuera de GT de artefactos. Sin ambas anotaciones es `null`. |
| FPR del detector | Soporte propuesto sobre tejido negativo / tejido negativo. Se informa por separado de pFPR para evitar que la abstención oculte errores de detección. |
| βσ | Desviación estándar del parche / desviación del anillo. Es ratio de desviaciones, no de varianzas; el ratio de varianzas se guarda aparte como βσ². |
| Sobel jump | Media de diferencia absoluta de magnitudes Sobel 3×3 entre pares vecinos que cruzan el borde, sin normalizar, uint8 0–255. No es la magnitud del gradiente tomada en Canny, usada en V4. |
| KS | Se registran D y p convencional como descriptivos. El speckle tiene dependencia espacial y las intensidades uint8 son discretas; el p iid no es una prueba válida de equivalencia acústica. |
| GLCM | 16 niveles, distancias 1,2,3, cuatro direcciones, pares con ambos extremos dentro del soporte. Contraste, homogeneidad, entropía y correlación; sin suficientes pares se informa `null`. |
| ACF | Diferencia de correlación de pares en ambos ejes a 1,2,3 píxeles. Proxy de textura B-mode, no medición de PSF. |
| Edge-L1 | Error de gradiente frente a GT limpio donde la marca cruza la banda del margen de una lesión independiente. |
| Dice / HD95 | Requieren segmentaciones realmente obtenidas antes/después; reutilizar la misma máscara no evalúa preservación. Unidades píxel salvo espaciado físico disponible. |
| LPIPS | Pendiente de ejecución con pesos/backbone identificados y contexto de parche comparable. No se sustituye por SSIM ni se inventa un resultado. |

Un p>0,05 **no acepta ni demuestra identidad de distribuciones**. Para equivalencia predefinir una tolerancia y evaluar su intervalo; usar remuestreo por paciente y, dentro de imágenes, bloques espaciales. Un p no significativo en una muestra pequeña también puede indicar potencia insuficiente. Véase [definición de KS en SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html).

Los valores absolutos Sobel <60 y contraste <0,2 dependen de escala, cuantización, definición y equipo. Son criterios exploratorios, no constantes fisiológicas universales. Los rangos históricos 49,8–68,4 no son directamente comparables con la nueva definición. Los controles de aceptación implementados incluyen cobertura, soporte, región protegida, βσ y Sobel; GLCM/ACF permanecen descriptivos hasta calibrar umbrales independientes.

Rayleigh/Nakagami modelan una envolvente apropiada. No ajustar esos modelos al PNG log-comprimido y llamarlo validación de física sin conocer la transformación de compresión. PSF real exige datos del transductor/adquisición o mediciones con fantoma; la autocorrelación del PNG no basta. No es posible recuperar con certeza el tejido originalmente tapado por una anotación opaca.

## Diseño de validación

### Nivel 1 — Reconstrucción pareada

Revisar imágenes limpias, excluir identificadores y anotaciones remanentes; separar pacientes antes de generar corrupciones. Congelar pacientes de prueba y semillas. Inyectar cruces +/×, líneas y texto variando escala, espesor, opacidad y posición, incluyendo cruces en márgenes de lesión solo para investigación. El generador actual cubre un conjunto inicial de geometrías, no todos los estilos de fabricantes.

Comparar con **misma GT y soporte de evaluación**: Telea BBox, Telea sobre trazo, FFC Places2 si se proporcionan sus pesos, FFC afinado, FFC acústico y detector+reconstructor. Separar máscara oráculo de detección end-to-end. Informar sensibilidad, falsos positivos y tasa de abstención. No contabilizar imágenes intactas por abstención como reconstrucciones perfectas.

La carpeta curada existente y el checkpoint no tienen trazabilidad suficiente para afirmar independencia entre entrenamiento y evaluación. Los experimentos ejecutados aquí son exploratorios; sus GT se refieren únicamente a las marcas nuevas inyectadas, no a una certificación de limpieza de la imagen completa.

### Nivel 2 — Fidelidad de textura

Evaluar por componente y por profundidad/tipo de tejido. El anillo excluye otras marcas conocidas y regiones protegidas. Un anillo heterogéneo puede invalidar la comparación. Reportar distribución de βσ, D de KS, GLCM, ACF y saltos de borde, no solo medias globales. Calibrar en validación y bloquear umbrales antes del test. El ajuste acústico solo se conserva como método propuesto si supera las ablaciones con intervalos por paciente.

### Nivel 3 — Anatomía

Crear GT experto de artefactos, parénquima, lesión, quistes y margen; incluir negativos hiperecoicos y espiculados. Evaluar pFPR observado y su incertidumbre sin tratar millones de píxeles correlacionados como muestras independientes. Cero errores observados no prueba tasa poblacional cero. Como referencia, para cero eventos en N pacientes independientes, el límite superior unilateral binomial 95% de la tasa de pacientes con error es 1−0,05^(1/N); no es un intervalo de pFPR por píxel.

Registrar por separado cambios en área, perímetro, compacidad, ejes y margen tras segmentación ciega antes/después. Convertir HD95 a mm solo con espaciado verificado. No deducir criterios BI-RADS a partir de dos métricas geométricas.

### Nivel 4 — CAD y atajos

Entrenar A=raw, B=BBox y C=inpainting con idénticos pacientes, inicializaciones, aumentos y presupuesto; usar al menos tres semillas. Elegir hiperparámetros únicamente en validación. Evaluar en pacientes limpios externos y en pares contrafactuales de **los mismos pacientes** con y sin marcas. La diferencia AUC train−test no identifica causalmente un atajo: también cambia dominio y sobreajuste.

`acoustic.cad` calcula ΔAUC pareado, cambio absoluto de score e intervalo por bootstrap de pacientes. Para cada brazo, exportar CSV `label,clean_score,marked_score,patient_id`; usar el mismo orden de casos y un modelo congelado. Para segmentación, añadir métricas por lesión. No se han entrenado modelos CAD ni generado predicciones clínicas en esta entrega.

Medir energía Grad-CAM en artefacto, lesión y enriquecimiento frente a fracción de área. Una región pequeña puede contener <1% de energía incluso con un mapa uniforme. Un caliper que cruza un tumor comparte biomarcadores, de modo que energía en su región no equivale por sí sola a un atajo. Añadir pruebas de aleatorización de pesos/etiquetas y cambios de predicción ante intervención; [Sanity Checks for Saliency Maps](https://proceedings.neurips.cc/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html) fundamenta la necesidad de controles de explicabilidad.

Entrenar también un detector real/restaurada en pacientes separados, y probar etiquetas permutadas y máscaras desplazadas. Un rendimiento cercano al azar con intervalo amplio no prueba ausencia de firma. El problema de generalización se formula en [Shortcut learning in deep neural networks](https://doi.org/10.1038/s42256-020-00257-z).

### Nivel 5 — Lectura ciega

Preparar originales y reconstrucciones equilibrados por clase, equipo y dificultad, retirar nombres/metadatos que revelen método, aleatorizar por lector, registrar semilla y respuestas. La tarea “real o manipulada” sobre una sola imagen es clasificación binaria; un verdadero 2AFC presenta dos alternativas emparejadas. Medir además distorsión de margen y confianza diagnóstica.

No exigir κ≈0 como criterio de éxito: κ mide acuerdo, no equivalencia diagnóstica. Evaluar sensibilidad/especificidad de detección, intervalos y no inferioridad morfométrica con márgenes y tamaño muestral definidos antes de observar resultados. Requiere radiólogos reales; no se han fabricado respuestas.

## Tabla solicitada para tesis — pendiente de experimento confirmatorio

Se preservan los encabezados solicitados. NE = no evaluado con este protocolo; no equivale a cero. Las cifras ilustrativas 54,28; 0,942; 0,184; 0,189; 0%; 66,38%; −0,01 no se trasladan a esta tabla como resultados.

| Dimensión de Calidad | Métrica Evaluada | Telea BBox (V1) | Inpainting Liso | LaMa Places2 (Base) | Pipeline Propuesto (V4 / Acústico) | Valor de Referencia Ideal |
|---|---|---|---|---|---|---|
| Física de Borde | Discontinuidad Sobel (ΔG) | NE | NE | NE | NE | Basal por equipo; <60 exploratorio con definición fijada |
| Estadística Speckle | Ratio de Varianza (βσ; realmente desviaciones estándar) | NE | NE | NE | NE | 1,000; tolerancia predefinida |
| Distribución Acústica | KS-Test (p-value) | NE | NE | NE | NE | p>0,05 no demuestra equivalencia |
| Textura Haralick | Δ Contraste GLCM | NE | NE | NE | NE | <0,200 solo con cuantización y soporte fijados |
| No-Invasión | Falsos Positivos Tisulares | NE | NE | NE | NE | 0% observado + intervalo y cobertura |
| Preservación | Ganancia de Tejido vs BBox | NE | NE | NE | NE | Máximo con remoción completa y GT correcto |
| Downstream CAD | Caída de AUC en Test (Δ) | NE | NE | NE | NE | 0,00 en comparación pareada independiente |

Los resultados efectivamente medidos figuran en `RESULTADOS_V5.md` y en CSV/JSON, con nombres de métodos y alcances reales. La arquitectura FFC se basa en [LaMa, WACV 2022](https://openaccess.thecvf.com/content/WACV2022/papers/Suvorov_Resolution-Robust_Large_Mask_Inpainting_With_Fourier_Convolutions_WACV_2022_paper.pdf); su rendimiento publicado no certifica preservación anatómica en ecografía.
