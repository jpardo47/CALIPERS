"""
=============================================================================
PIPELINE DE INFERENCIA CLÍNICA - LaMa FINE-TUNED EN ECOGRAFÍAS MAMARIAS
=============================================================================
Modelo entrenado para la eliminación de calipers, textos quemados, marcas y
artefactos en imágenes ecográficas conservando íntegramente la textura de moteado
acústico (speckle) y los bordes anatómicos/lesionales.
=============================================================================
"""

import os
import sys
import argparse
from pathlib import Path

# Configurar salida segura para consola Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

# Añadir ruta de lama-main al path
CURRENT_DIR = Path(__file__).resolve().parent
LAMA_DIR = CURRENT_DIR / 'lama-main'
if str(LAMA_DIR) not in sys.path:
    sys.path.insert(0, str(LAMA_DIR))

from saicinpainting.training.modules.ffc import FFCResNetGenerator


def build_generator(device: torch.device) -> nn.Module:
    """Construye la arquitectura idéntica de big-lama (FFC-ResNet, 18 bloques)."""
    generator = FFCResNetGenerator(
        input_nc=4,
        output_nc=3,
        ngf=64,
        n_downsampling=3,
        n_blocks=18,
        add_out_act='sigmoid',
        init_conv_kwargs={'ratio_gin': 0, 'ratio_gout': 0, 'enable_lfu': False},
        downsample_conv_kwargs={'ratio_gin': 0, 'ratio_gout': 0, 'enable_lfu': False},
        resnet_conv_kwargs={'ratio_gin': 0.75, 'ratio_gout': 0.75, 'enable_lfu': False}
    )
    generator.to(device)
    generator.eval()
    return generator


def load_weights(generator: nn.Module, weights_path: str, device: torch.device):
    """
    Carga los pesos desde mejor_modelo_lama_generator.pt (.pt puro)
    o desde el checkpoint completo de PyTorch Lightning (mejor_modelo_lama.ckpt).
    """
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de pesos: {weights_path}")

    print(f"[*] Cargando pesos desde: {weights_path.name} ...")

    state = torch.load(weights_path, map_location=device, weights_only=True)
    state = state.get('state_dict', state)
    if any(k.startswith('generator.') for k in state):
        state = {k[len('generator.'):]: v for k, v in state.items() if k.startswith('generator.')}
    generator.load_state_dict(state, strict=True)
    print(f"[OK] Pesos completos verificados en {device}.")


def pad_to_modulo(tensor: torch.Tensor, modulo: int = 8):
    """Añade padding de reflexión para que el tamaño H, W sea múltiplo exacto de `modulo`."""
    _, _, h, w = tensor.shape
    pad_h = (modulo - (h % modulo)) % modulo
    pad_w = (modulo - (w % modulo)) % modulo

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    tensor_padded = F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect')
    return tensor_padded, (pad_top, pad_bottom, pad_left, pad_right)


def unpad(tensor: torch.Tensor, pad_info):
    """Elimina el padding para restaurar la resolución anatómica exacta original."""
    pad_top, pad_bottom, pad_left, pad_right = pad_info
    h, w = tensor.shape[-2], tensor.shape[-1]
    bottom_idx = h - pad_bottom if pad_bottom > 0 else h
    right_idx = w - pad_right if pad_right > 0 else w
    return tensor[..., pad_top:bottom_idx, pad_left:right_idx]


class InpainterEcografia:
    """Inferencia experimental de bajo nivel; control selectivo en acoustic.cli."""

    def __init__(self, weights_path: str = "mejor_modelo_lama_generator.pt", device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = build_generator(self.device)
        load_weights(self.model, weights_path, self.device)

    def inpaint(self, image_np: np.ndarray, mask_np: np.ndarray) -> np.ndarray:
        """
        Ejecuta el inpainting sobre una ecografía y su máscara binaria correspondiente.

        Args:
            image_np: Imagen RGB o escala de grises [H, W, 3] o [H, W], valores [0, 255].
            mask_np: Máscara binaria [H, W], valores 0 (conservar) y 255 (reconstruir).

        Returns:
            np.ndarray: Imagen ecográfica restaurada [H, W, 3] en formato uint8 BGR/RGB según entrada.
        """
        from acoustic.metrics import binary, gray
        gray(image_np)
        original = image_np.copy()
        mask_bool = binary(mask_np, image_np.shape[:2])
        if not mask_bool.any():
            return original
        if min(image_np.shape[:2]) < 16:
            raise ValueError('FFC requiere dimensiones >= 16 px')
        if image_np.ndim == 2:
            image_np = np.repeat(image_np[..., None], 3, axis=2)
        mask_np = mask_bool.astype(np.uint8) * 255

        # Normalización a float32 [0.0, 1.0]
        img_norm = image_np.astype(np.float32) / 255.0
        mask_norm = (mask_np > 127).astype(np.float32)

        # Conversión a tensores PyTorch [1, C, H, W]
        img_t = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).to(self.device)
        mask_t = torch.from_numpy(mask_norm).unsqueeze(0).unsqueeze(0).to(self.device)

        # Enmascarar la entrada: zonas dañadas se ponen a 0
        img_masked = img_t * (1.0 - mask_t)

        # Concatenar imagen (3 canales) + máscara (1 canal) -> 4 canales
        model_input = torch.cat([img_masked, mask_t], dim=1)

        # Padding simétrico a múltiplo de 8
        padded_input, pad_info = pad_to_modulo(model_input, modulo=8)

        # Inferencia sin gradientes
        with torch.no_grad():
            inpainted_padded = self.model(padded_input)
        if inpainted_padded.shape != padded_input[:, :3].shape or not torch.isfinite(inpainted_padded).all():
            raise RuntimeError('Salida FFC inválida: dimensiones incorrectas o valores no finitos')

        # Quitar padding
        inpainted = unpad(inpainted_padded, pad_info)

        # Fusión clínica conservadora:
        # Los píxeles fuera de la máscara (tejido sano) se preservan 100% idénticos al original.
        final_t = inpainted * mask_t + img_t * (1.0 - mask_t)

        # Conversión de regreso a uint8
        res_np = final_t.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0).cpu().numpy()
        res_uint8 = (res_np * 255.0).round().astype(np.uint8)

        if original.ndim == 2:
            res_uint8 = cv2.cvtColor(res_uint8, cv2.COLOR_RGB2GRAY)
        result = original.copy()
        result[mask_bool] = res_uint8[mask_bool]
        return result


def process_single(inpainter: InpainterEcografia, img_path: str, mask_path: str, out_path: str):
    """Procesa un par individual de ecografía y máscara."""
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"No se pudo abrir la imagen: {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    if not mask_path or not os.path.exists(mask_path):
        raise ValueError('Se requiere máscara explícita de ARTEFACTOS. Nunca use una máscara de lesión BUSI.')
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError('Máscara ilegible')
    if Path(out_path).suffix.lower() != '.png':
        raise ValueError('Salida PNG requerida para conservar píxeles sin pérdidas')
    if Path(out_path).resolve() in (Path(img_path).resolve(), Path(mask_path).resolve()):
        raise ValueError('No se permite sobrescribir entradas')
    # Explicit low-level mask path remains available; guarded workflow: acoustic.cli.
    res_rgb = inpainter.inpaint(img_rgb, mask)
    res_bgr = cv2.cvtColor(res_rgb, cv2.COLOR_RGB2BGR)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    if not cv2.imwrite(out_path, res_bgr):
        raise OSError('No se pudo guardar resultado')
    print(f'[OK] Guardado resultado: {out_path}')


def process_folder(inpainter: InpainterEcografia, indir: str, outdir: str, mask_suffix: str = "_artifact_mask"):
    """Procesa un directorio completo de ecografías con máscaras asociadas."""
    in_path = Path(indir)
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Buscar imágenes que no sean máscaras
    image_exts = {'.png', '.jpg', '.jpeg', '.bmp'}
    images = [
        f for f in in_path.iterdir()
        if f.suffix.lower() in image_exts and "_mask" not in f.stem
    ]

    print(f"[*] Procesando lote de {len(images)} ecografías desde {indir} ...")
    for img_file in images:
        # Buscar máscara correspondiente
        mask_file = in_path / f"{img_file.stem}{mask_suffix}{img_file.suffix}"
        if not mask_file.exists():
            mask_file = in_path / f"{img_file.stem}{mask_suffix}.png"

        cur_mask_path = str(mask_file) if mask_file.exists() else None
        target_out = str(out_path / f"{img_file.stem}_restaurada.png")

        print(f" -> Procesando: {img_file.name}")
        process_single(inpainter, str(img_file), cur_mask_path, target_out)

    print(f"[✓] Lote completado. Resultados en: {outdir}")


def main():
    parser = argparse.ArgumentParser(description="Inferencia LaMa Fine-Tuned para Ecografías Mamarias")
    parser.add_argument("--image", type=str, help="Ruta a la ecografía con marcas/calipers")
    parser.add_argument("--mask", type=str, help="Ruta a la máscara binaria (blanco = remover)")
    parser.add_argument("--output", type=str, default="resultado_ecografia_restaurada.png", help="Ruta del resultado")
    parser.add_argument("--indir", type=str, help="Directorio de entrada para procesamiento por lotes")
    parser.add_argument("--outdir", type=str, default="resultados/lote_restaurado", help="Directorio de salida")
    parser.add_argument("--weights", type=str, default="mejor_modelo_lama_generator.pt", help="Ruta al modelo (.pt o .ckpt)")
    parser.add_argument("--device", type=str, default=None, help="Dispositivo: 'cuda' o 'cpu'")

    args = parser.parse_args()
    if not args.indir and not args.image:
        parser.error('Especifique --image y --mask, o --indir con máscaras _artifact_mask')
    if args.image and not args.mask:
        parser.error('--mask de artefactos obligatorio; no se inventan máscaras')
    inpainter = InpainterEcografia(weights_path=args.weights, device=args.device)
    if args.indir:
        process_folder(inpainter, args.indir, args.outdir)
    else:
        process_single(inpainter, args.image, args.mask, args.output)


if __name__ == '__main__':
    main()
