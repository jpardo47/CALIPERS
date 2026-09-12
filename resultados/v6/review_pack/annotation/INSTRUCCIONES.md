# Revisión pendiente: no hay GT experto de trazos

Las imágenes Axxx son originales a resolución nativa. Las propuestas Axxx_proposal NO son ground truth. Incluyen candidatos amarillos/blancos y posibles falsos positivos; no usarlas como autorización de borrado.

Dos revisores independientes deben dibujar máscaras binarias de artefactos y tejido, marcar incertidumbre y revisar la ausencia de marcas. El adjudicador resuelve discrepancias. Registrar IDs de revisores, rutas a PNG de tamaño original, fecha y estado adjudicated. No rellenar campos pendientes con respuestas automáticas.

Incluir ramas finas, antialiasing, texto de diferentes tamaños, calipers de borde, líneas punteadas y tejido hiperecoico/espiculado como negativos difíciles. La selección por brillo es un criterio de muestreo, no diagnóstico ni GT de negativo.

Las máscaras de lesión originales BUS_BRA pueden orientar una revisión anatómica separada, pero no indican qué píxeles son artefactos ni certifican parénquima sano.
