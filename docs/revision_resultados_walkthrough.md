# Revisión de resultados: `walkthrough.md` + `inpaint_quantitative_qc_benchmark.csv`

## Veredicto general

Verifiqué la tabla agregada del walkthrough directamente contra el CSV exportado. **Dos de las cinco columnas coinciden exactamente** (cobertura de máscara y ratio de varianza de speckle), lo cual es buena señal de que esas cifras sí salen del pipeline real. Pero encontré **una contradicción robusta entre el dato medido y la narrativa del documento** que hay que resolver antes de reportar esto como "sin firmas de inpainting", y dos casos de reporte selectivo/no reproducible que conviene corregir.

---

## 1. Lo que sí coincide (verificado directamente contra el CSV)

| Métrica | Walkthrough | CSV (recalculado) | ✓/✗ |
|---|---|---|---|
| Cobertura de máscara — Baseline / Trazo / Híbrido | 1.32% / 1.11% / 1.11% | 1.323% / 1.113% / 1.113% | ✓ coincide |
| Speckle variance ratio — Baseline / Trazo / Híbrido | 0.8009 / 0.7549 / 0.9614 | 0.8009 / 0.7549 / 0.9614 | ✓ coincide exacto |

Buena noticia: el objetivo central del método (que el parche recupere varianza de speckle ≈ tejido circundante) **sí se sostiene con los 14 archivos de prueba**, no es solo un ejemplo aislado.

---

## 2. Hallazgo crítico: la discontinuidad de borde del método híbrido es **peor**, no mejor

La tabla del walkthrough describe el resultado del Método Híbrido como *"Suave / Fusión Feathering"* — sin dar un número — dando a entender que el feathering resolvió el problema de costuras visibles. **Los datos del CSV dicen lo contrario:**

| Método | `boundary_gradient_mean` (media) | `boundary_gradient_mean` (mediana) |
|---|---:|---:|
| 1. Baseline (BBox) | 64.90 | 50.15 |
| 2. Trazo Simple | 54.75 | 45.77 |
| **3. Híbrido Propuesto** | **86.98** | **78.37** |

En **13 de los 14 archivos**, el método híbrido tiene mayor discontinuidad de gradiente en el borde que el baseline original — no menor. Confirmé que esto no es un par de outliers arrastrando el promedio: la **mediana** muestra el mismo patrón (78.37 vs. 50.15 vs. 45.77).

El mismo patrón aparece en `glcm_contrast_diff` (mediana: Baseline 5.30, Trazo 5.08, **Híbrido 8.97** — también peor).

**Lectura más probable:** el método híbrido sí logra que la *varianza* del parche se parezca a la del tejido (speckle ratio ≈1.0), pero el ruido sintetizado probablemente no tiene la **estructura de autocorrelación espacial** del speckle real (no es solo cuestión de varianza, sino de cómo se distribuye espacialmente) — lo que puede estar generando una transición más abrupta/texturizada justo en el borde del parche que el propio inpainting liso original. Es decir: **resolviste el problema de "parche demasiado liso" pero probablemente introdujiste un problema nuevo de "costura más marcada"**, y el documento no lo reporta.

**Esto es importante porque el argumento científico central del método (prevenir shortcut learning por firma visual) depende de que NO haya discontinuidades de borde detectables.** Si el boundary gradient es más alto que el baseline, eso podría ser en sí mismo una firma explotable — el mismo tipo de atajo que se buscaba eliminar, solo que con otra forma.

**Necesito ver el código de `inpaint_hybrid_speckle` y `compute_qc_metrics`** (no solo el resumen del walkthrough) para saber si esto es: (a) un problema real del método de síntesis de speckle, o (b) un artefacto de cómo se calcula `boundary_gradient_mean` una vez que hay ruido de alta frecuencia inyectado cerca del borde (el gradiente de Sobel sube con cualquier textura de alta frecuencia, sea "costura" o "speckle correctamente sintetizado" — el detector podría no estar distinguiendo entre ambos casos). ¿Puedes compartir esas dos funciones?

---

## 3. El ejemplo destacado (`benign (11).png`) no está en el CSV

La Sección 3 del walkthrough describe en detalle `benign (11).png` (2,795 px → 1,042 px, 59.47% de tejido preservado, speckle ratio 0.3242 → 0.9273). **Ese archivo no aparece en ninguna fila del CSV** — el CSV solo tiene 14 archivos (`benign (1)`, `(10)`, `(100)`–`(105)`, `malignant (1)`, `(10)`, `(100)`, `(101)`, `(103)`, `(104)`), y `benign (11)` no es ninguno de ellos.

Esto significa que el ejemplo más impresionante del documento (59.47% de preservación de tejido) **no se puede verificar con los datos que tengo**, y tampoco es representativo del conjunto: el rango real medido en las 14 imágenes del CSV es 0%–36.84% (media 12.54%), no hasta 59.46% como sugiere la tabla resumen al combinar el promedio agregado con este ejemplo suelto.

**Pregunta directa:** ¿`benign (11).png` se probó por separado y no se incluyó en el batch exportado, o el CSV está incompleto? Si el batch real incluyó más archivos, pásame el CSV completo — la tabla resumen debería salir de un solo conjunto de datos, no mezclar un ejemplo individual con un promedio agregado de otro conjunto.

---

## 4. Reporte selectivo: 2 de 14 imágenes tuvieron ganancia cero

`benign (102).png` y `malignant (101).png` tienen `tissue_preservation_gain_pct = 0.0` — el enmascaramiento a nivel de trazo no mejoró nada sobre el bounding box en esos casos. El walkthrough no menciona estos casos en ningún lado (solo destaca los positivos: 33.1%, 36.84%, etc.). Vale la pena investigar por qué el método no ayudó ahí — capaz el detector no encontró trazo distinguible dentro del bbox y volvió a rellenar la caja completa — y reportarlo junto con los casos exitosos. Un reviewer va a notar la ausencia de casos "sin mejora" si solo se muestran los mejores.

Adicionalmente, `benign (102).png` tiene una cobertura de máscara casi nula (0.0109%) y valores de `glcm_contrast_diff` anormalmente altos (327–370, muy por encima del resto del dataset que ronda 1–25). Esto huele a una región detectada demasiado pequeña para que las features GLCM sean estadísticamente estables — vale la pena añadir un mínimo de área de píxeles antes de confiar en esas métricas por imagen, y no promediarlas sin revisar outliers primero.

---

## 5. Nota menor (no es un problema, pero documentar en métodos)

`ssim_outside_mask` baja ligeramente de baseline (0.9970) a híbrido (0.9957). Esto es esperable **por diseño** — el alpha-feathering mezcla intencionalmente un poco más allá del borde de la máscara binaria original, así que "fuera de máscara" ya no es una frontera tan estricta como en el método bbox simple. No es un defecto, pero como el criterio de éxito original decía "SSIM fuera de máscara ≈ 1.000 siempre", vale la pena aclarar en metodología que la definición de "fuera de máscara" cambia ligeramente entre métodos.

---

## 6. Qué necesito para seguir

Para poder confirmar si el punto 2 (discontinuidad de borde) es un problema real del método o del metric, y si los fixes de SSIM/GLCM que se describen en la tabla de la Sección 1 del walkthrough quedaron bien implementados, necesito ver el código real, no solo el resumen — al menos:
- `inpaint_hybrid_speckle` (cómo se sintetiza y se funde el speckle)
- `compute_qc_metrics` (para confirmar que el fix de SSIM 2D y el fix de GLCM sin sesgo de negro están efectivamente en el código, no solo descritos)

¿Los compartes, o prefieres que trabajemos directo sobre el `.ipynb` actualizado?
