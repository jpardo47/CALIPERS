# Uso del pipeline experimental V5

Ejecutar desde la raíz del proyecto con Python. Dependencias científicas en `requirements-validation.txt`; el equipo usado tiene PyTorch 2.4.1 CPU. Los originales y los pesos existentes no se modifican. Cada ejecución requiere un directorio de salida nuevo.

## Flujo selectivo sin intervención durante el lote

```powershell
python -m acoustic.cli --image bus-cleaning-main/sample_data/image_1.png --outdir resultados/nueva_auditoria
```

Genera `candidate_mask.png`, `result.png` y `audit.json`. Sin soporte independiente de artefactos, `result.png` conserva el original y el estado es `abstained`. `unchanged` significa ausencia de candidatos, no ausencia probada de marcas. `accepted_research` indica únicamente que se superaron los controles implementados.

Para un lote desatendido, preparar una lista JSON con `image` y opcionalmente `artifact_mask`, `certified_artifact`, `protected`. Las rutas se resuelven respecto al manifiesto:

```powershell
python -m acoustic.batch --manifest resultados/v5_lote_muestras.json --outdir resultados/otro_lote
```

Se escribe un registro por imagen, un `ledger.csv` actualizado tras cada caso y `summary.json`. Un caso ilegible queda como error sin detener los demás; el proceso devuelve código distinto de cero si hubo errores. En las cinco muestras reales ejecutadas: cuatro abstenciones, una sin candidatos, cero errores. No son cinco remociones exitosas.

Para una imagen con máscara de artefactos revisada y región anatómica protegida:

```powershell
python -m acoustic.cli --image imagen.png --mask trazos_propuestos.png --certified-artifact artefactos_revisados.png --protected lesion.png --weights mejor_modelo_lama_generator.pt --outdir resultados/nueva_reconstruccion
```

Las máscaras deben ser 2D, de las mismas dimensiones y contener 0/1 o 0/255. **Nunca use la máscara tumoral BUSI como `--mask` o `--certified-artifact`.** Una intersección con `--protected` produce abstención. La certificación corresponde al soporte real del trazo, no a su caja envolvente.

El modo con pesos evalúa el candidato FFC acústico. La regularización es experimental y **no ha superado al FFC puro en el piloto**. Para estudiar FFC puro con una máscara de artefactos explícita se conserva `inferencia_ecografia.py`; esa entrada es de bajo nivel y no aplica los controles selectivos ni produce certificación de seguridad.

## Benchmark exploratorio reproducible

```powershell
python -m acoustic.benchmark --images ecografias_limpias_para_drive --pattern 'Images_bus_*.png' --pattern 'malignant*.png' --limit 2 --weights mejor_modelo_lama_generator.pt --outdir resultados/otra_validacion
```

`--limit` es por patrón. Cada caso guarda referencia, corrupción, GT de marca sintética, detección y salidas de siete métodos; `metrics.csv` y `report.json` contienen resultados y hashes. Las métricas de reconstrucción usan la misma GT sintética para todos los métodos, incluso si el detector omite la mayoría de la marca. Añada `--base-weights` solo si dispone del generador Places2 real; el modelo afinado nunca se etiqueta como baseline Places2.

Las imágenes curadas no están certificadas limpias y podrían pertenecer al entrenamiento del checkpoint. Por ello, estos resultados sirven para depuración/ablación, no para una tabla confirmatoria. La primera prueba `validacion_v5_exploratoria` precede a la mejora de colocación de marcas y puede incluir fondo negro. La prueba `validacion_v5_multifuente` usa contexto visible y es el piloto principal. No mezclar ambos protocolos.

## Manifiestos por paciente

Ya se produjo `resultados/v5_manifiesto_bus_bra.json`: 1.875 imágenes, repartidas en 1.323/280/272 por train/val/test. Las dos vistas del mismo `Case` permanecen juntas. Se comprueban duplicados exactos decodificados; no se afirma haber descartado todos los casi duplicados. La asignación usa semilla 42, no estratificación por patología; comprobar balance antes de un estudio definitivo.

```powershell
python -m acoustic.manifest --metadata data/bases_datos_originales/BUS_BRA/BUSBRA/BUSBRA/bus_data.csv --images data/bases_datos_originales/BUS_BRA/BUSBRA/BUSBRA/Images --output resultados/otro_manifiesto.json
```

Formato común para BUS_BRA, BUSI y UDIAT: lista JSON con `image` (ruta absoluta), `patient_id`, `dataset`, `split` (`train`, `val`, `test` o `external`), `clean_verified`, y opcionalmente `label`, `device`. No inventar IDs de paciente a partir de nombres de imagen BUSI/UDIAT. Si faltan los identificadores, la independencia por paciente queda pendiente. UDIAT no fue localizado en las carpetas examinadas y no se evaluó.

## Afinamiento reproducible con restricciones de textura

Después de revisión independiente de limpieza, un manifiesto debe marcar `clean_verified: true` en train/val. Esta revisión no puede sustituirse por una heurística de brillo. El manifiesto BUS_BRA generado tiene `false` de forma deliberada.

```powershell
python -m acoustic.train --manifest manifiesto_limpio_revisado.json --weights mejor_modelo_lama_generator.pt --outdir resultados/nuevo_finetuning --epochs 10 --device cuda
```

Implementa L1 en máscara, error de desviación, gradiente y covarianza espacial; recortes nativos sin redimensionar, máscaras de validación fijas, pesos elegidos por pérdida de validación, clipping de gradiente y registro de configuración/hashes. No usa test/external para actualizar pesos o seleccionar épocas. Los valores de pesos de pérdidas son iniciales, no hiperparámetros clínicamente validados. El esquema de máscaras de entrenamiento inicial no sustituye una biblioteca validada de estilos reales de equipos.

No se ejecutó un entrenamiento completo con esta función; se verificaron pérdidas y gradientes y se hizo inferencia real con los pesos preexistentes. Esos pesos **no fueron entrenados con las nuevas pérdidas**. En GPU determinista, algunos entornos pueden exigir configuración adicional de CUDA; un fallo debe resolverse explícitamente, no desactivando silenciosamente la reproducibilidad.

El preparador `lama_ultrasound_finetuning/dataset_preparator.py` también exige `--manifest` y exporta únicamente train/val verificados, conservando el tamaño original.

## CAD y XAI

```powershell
python -m acoustic.cad --predictions predicciones_pareadas.csv --output resultados/auditoria_cad.json
```

CSV: `label,clean_score,marked_score,patient_id`. Cada fila corresponde al mismo caso evaluado con y sin marcas por un modelo congelado. Repetir para brazos raw/BBox/inpainting usando idénticos pacientes. `attention_energy` recibe mapas no negativos registrados, máscara de artefactos y, opcionalmente, lesión. Ni el bootstrap ni Grad-CAM demuestran por sí solos eliminación universal de atajos.

## Pruebas

```powershell
python -m unittest discover -s tests -v
```

Incluyen preservación exacta exterior, abstención, exclusiones, máscaras incompatibles, PSNR local, GT independiente para pFPR, guardado sin pérdidas, fuga entre pacientes, inferencia gris/RGB, rechazo de pesos incompletos y gradientes de pérdidas físicas. Los tests del contrato FFC usan un modelo sustituto pequeño; el benchmark sí ejecutó big-LaMa y verificó carga estricta del checkpoint real.
