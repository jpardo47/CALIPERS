"""
Generador Especializado de Máscaras Sintéticas de Artefactos Ecográficos
=======================================================================
Diseñado para el entrenamiento autosupervisado de LaMa (Large Mask Inpainting)
en ecografía mamaria (BUS).

Simula con precisión geométrica:
1. Calipers cruciformes ortogonales (+) y oblicuos (x) de 1 a 3 px de espesor.
2. Trazos punteados de medición entre calipers (dotted lines).
3. Texto médico quemado alfanumérico (8 a 14 px) con cadenas de medición y parámetros.
4. Muestreo dirigido (targeted sampling) hacia zonas hipoecoicas (nódulos) y parénquima.
"""

import numpy as np
import cv2
import random
from typing import Tuple, Optional


class UltrasoundArtifactMaskGenerator:
    """
    Generador de máscaras sintéticas de calipers y texto médico para ecografía.
    Produce máscaras binarias uint8 (0: tejido válido, 255: artefacto a enmascarar).
    """

    def __init__(
        self,
        img_size: Tuple[int, int] = (512, 512),
        min_calipers: int = 2,
        max_calipers: int = 6,
        line_thickness_range: Tuple[int, int] = (1, 3),
        caliper_arm_length_range: Tuple[int, int] = (12, 28),
        include_text: bool = True,
        include_dotted_lines: bool = True,
        nodule_bias_probability: float = 0.65,
    ):
        self.height, self.width = img_size
        self.min_calipers = min_calipers
        self.max_calipers = max_calipers
        self.thickness_range = line_thickness_range
        self.arm_range = caliper_arm_length_range
        self.include_text = include_text
        self.include_dotted_lines = include_dotted_lines
        self.nodule_bias_prob = nodule_bias_probability

        # Cadenas típicas de ecógrafos de mama
        self.sample_texts = [
            "D1: {d1:.2f}cm",
            "D2: {d2:.2f}cm",
            "{d1:.1f}x{d2:.1f}mm",
            "VOL: {vol:.1f}cc",
            "7.5MHz",
            "12.0MHz",
            "DR: 65",
            "MI: 0.8",
            "RAD 9:00",
            "RAD 2:00",
            "SUP LAT",
            "INF MED",
            "DIST: {d1:.2f}cm",
        ]

    def _get_target_location(self, image: Optional[np.ndarray] = None) -> Tuple[int, int]:
        """
        Calcula una ubicación (x, y). Si se suministra la imagen original y se activa
        el sesgo de nódulo, busca preferentemente una región hipoecoica (oscura, nódulo).
        """
        margin = 40
        if image is not None and random.random() < self.nodule_bias_prob:
            # Convertir a escala de grises si viene en 3 canales
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # Recortar márgenes externos para no muestrear las barras negras del marco
            h, w = gray.shape
            roi = gray[margin : h - margin, margin : w - margin]
            
            # Buscar coordenadas de píxeles con baja intensidad (zonas hipoecoicas / nódulos)
            # pero excluyendo el negro puro del fondo exterior (intensidad > 15 y < 90)
            candidate_y, candidate_x = np.where((roi > 15) & (roi < 90))
            if len(candidate_x) > 0:
                idx = random.randint(0, len(candidate_x) - 1)
                return candidate_x[idx] + margin, candidate_y[idx] + margin

        # Ubicación uniforme aleatoria como fallback
        x = random.randint(margin, self.width - margin)
        y = random.randint(margin, self.height - margin)
        return x, y

    def _draw_caliper(
        self,
        mask: np.ndarray,
        center: Tuple[int, int],
        is_oblique: bool = False,
        arm_len: Optional[int] = None,
        thickness: Optional[int] = None,
    ):
        """Dibuja una cruz de caliper ortogonal (+) u oblicua (x)"""
        cx, cy = center
        arm = arm_len or random.randint(*self.arm_range)
        t = thickness or random.randint(*self.thickness_range)

        if not is_oblique:
            # Cruz ortogonal (+)
            cv2.line(mask, (cx - arm, cy), (cx + arm, cy), 255, t)
            cv2.line(mask, (cx, cy - arm), (cx, cy + arm), 255, t)
        else:
            # Cruz oblicua (x) rotada 45 grados
            d = int(arm * 0.707)
            cv2.line(mask, (cx - d, cy - d), (cx + d, cy + d), 255, t)
            cv2.line(mask, (cx - d, cy + d), (cx + d, cy - d), 255, t)

    def _draw_dotted_line(
        self,
        mask: np.ndarray,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        thickness: int = 1,
        gap: int = 6,
    ):
        """Dibuja una línea punteada entre dos calipers de medición"""
        dist = np.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
        if dist < 10:
            return
        points_count = int(dist / gap)
        for i in range(1, points_count):
            r = i / float(points_count)
            x = int(pt1[0] * (1 - r) + pt2[0] * r)
            y = int(pt1[1] * (1 - r) + pt2[1] * r)
            cv2.circle(mask, (x, y), thickness, 255, -1)

    def _draw_text_glyph(
        self,
        mask: np.ndarray,
        pt: Tuple[int, int],
    ):
        """Dibuja una etiqueta de texto médico sintético"""
        template = random.choice(self.sample_texts)
        text = template.format(
            d1=random.uniform(0.5, 3.5),
            d2=random.uniform(0.4, 2.8),
            vol=random.uniform(0.2, 8.5),
        )

        font = random.choice([
            cv2.FONT_HERSHEY_SIMPLEX,
            cv2.FONT_HERSHEY_DUPLEX,
        ])
        scale = random.uniform(0.35, 0.48)
        thickness = 1

        # Evitar salir de la imagen
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        tx = min(max(pt[0], 10), self.width - tw - 10)
        ty = min(max(pt[1], th + 10), self.height - 10)

        cv2.putText(mask, text, (tx, ty), font, scale, 255, thickness, cv2.LINE_AA)

    def generate(self, image: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Genera una máscara sintética completa para una imagen dada.
        Retorna:
            mask: np.ndarray binario uint8 de dimensiones (H, W) con 0 y 255.
        """
        mask = np.zeros((self.height, self.width), dtype=np.uint8)

        num_calipers = random.randint(self.min_calipers, self.max_calipers)
        caliper_points = []

        # 1. Generar calipers en pares o grupos de medición
        for _ in range(num_calipers):
            pt = self._get_target_location(image)
            caliper_points.append(pt)
            is_oblique = random.random() < 0.35
            self._draw_caliper(mask, pt, is_oblique=is_oblique)

        # 2. Conectar pares de calipers con trazos punteados de medición
        if self.include_dotted_lines and len(caliper_points) >= 2:
            num_pairs = min(len(caliper_points) // 2, 2)
            for i in range(num_pairs):
                p1 = caliper_points[2 * i]
                p2 = caliper_points[2 * i + 1]
                t = random.randint(*self.thickness_range)
                self._draw_dotted_line(mask, p1, p2, thickness=t)

        # 3. Superponer texto médico quemado
        if self.include_text:
            num_texts = random.randint(1, 3)
            for _ in range(num_texts):
                t_pos = self._get_target_location(image)
                self._draw_text_glyph(mask, t_pos)

        # Pequeña dilatación morfológica opcional para cubrir bordes antialias de los ecógrafos
        if random.random() < 0.4:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            mask = cv2.dilate(mask, kernel, iterations=1)

        return mask

    def __call__(self, img: np.ndarray) -> dict:
        """
        Interfaz compatible con pipelines de PyTorch / Albumentations / LaMa.
        """
        h, w = img.shape[:2]
        if (h, w) != (self.height, self.width):
            self.height, self.width = h, w

        mask = self.generate(img)
        # Normalizar a rango [0, 1] en float32 si es requerido por el dataloader
        mask_norm = (mask > 127).astype(np.float32)
        return {"image": img, "mask": mask_norm}


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("Demostración del generador sintético de artefactos de ecografía...")
    # Crear un lienzo sintético simulando ecografía con speckle y nódulo hipoecoico
    synth_bus = np.random.gamma(shape=2.0, scale=30.0, size=(512, 512)).astype(np.uint8)
    # Dibujar un nódulo elíptico hipoecoico central
    cv2.ellipse(synth_bus, (256, 256), (90, 60), 15, 0, 360, 35, -1)
    # Suavizado para simular la PSF del transductor
    synth_bus = cv2.GaussianBlur(synth_bus, (5, 5), 1.5)

    gen = UltrasoundArtifactMaskGenerator(img_size=(512, 512))
    mask = gen.generate(synth_bus)

    # Corromper la imagen limpia con la máscara
    corrupted = synth_bus.copy()
    corrupted[mask == 255] = 255  # El artefacto ecográfico quema los píxeles a blanco puro

    print(f"Máscara generada con éxito. Píxeles enmascarados: {np.sum(mask == 255)} px ({(np.mean(mask == 255)*100):.2f}% del área)")
