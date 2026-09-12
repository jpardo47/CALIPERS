"""
Preparador y Estructurador de Datasets Ecográficos Limpios para LaMa
===================================================================
Filtra, valida y organiza imágenes ecográficas mamarias sin artefactos
en la estructura estándar requerida por el repositorio oficial advimman/lama:

dataset_ecografia_clean/
├── train/  (85% - 90% de las imágenes)
└── val/    (10% - 15% de las imágenes)

Asegura:
- Validación de que la imagen sea ecografía válida (sin texto o marcas quemadas previas).
- Replicación a 3 canales idénticos (R=G=B) para compatibilidad con big-lama preentrenado.
- Redimensionamiento preservando relación de aspecto o padding simétrico si es necesario.
"""

import os
import shutil
import argparse
from pathlib import Path
import cv2
import numpy as np
from typing import List, Tuple


def is_image_clean(image_path: Path, max_white_ratio: float = 0.002) -> bool:
    """
    Heurística rápida para descartar imágenes que ya contengan calipers o texto quemado.
    Calcula el porcentaje de píxeles saturados (> 250 en escala de grises) en la región central.
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    
    h, w = img.shape
    # Evitar bordes externos donde a veces hay marcas de escala
    roi = img[int(h * 0.1) : int(h * 0.9), int(w * 0.1) : int(w * 0.9)]
    white_ratio = np.mean(roi >= 252)

    # Si hay demasiados píxeles saturados continuos, es probable que tenga calipers
    return white_ratio < max_white_ratio


def prepare_lama_dataset(
    source_dir: str,
    output_dir: str,
    val_split: float = 0.15,
    target_size: Tuple[int, int] = (512, 512),
    filter_clean: bool = True,
    seed: int = 42,
    manifest_path: str = None,
):
    # The old image-level shuffle leaks multiple views of the same patient.
    # All new preparation goes through explicit patient membership.
    if manifest_path is None:
        raise ValueError('Se requiere manifest_path con patient_id, split y clean_verified; el brillo no certifica limpieza.')
    return prepare_from_manifest(manifest_path, output_dir)


def prepare_from_manifest(manifest_path, output_dir):
    import json
    import hashlib
    import sys
    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0,root)
    from acoustic.manifest import validate
    from acoustic.cli import read, write
    rows = validate(json.loads(Path(manifest_path).read_text(encoding='utf-8')))
    chosen = [r for r in rows if r['split'] in ('train','val')]
    if not chosen or any(r.get('clean_verified') is not True for r in chosen):
        raise ValueError('Se requieren imágenes verificadas limpias por revisión independiente')
    output = Path(output_dir)
    if output.exists():
        raise ValueError('Use una carpeta nueva; no mezcle particiones anteriores')
    output.mkdir(parents=True)
    for row in chosen:
        dest = output/row['split']
        dest.mkdir(exist_ok=True)
        image = read(row['image'])
        if image.ndim==2:
            image = np.repeat(image[...,None],3,axis=2)
        # No resize: retain native spatial scale and avoid anisotropic stretching.
        name = row['dataset']+'_'+row['pixel_sha256'][:16]+'.png'
        if Path(name).name != name:
            raise ValueError('Dataset identifier cannot contain path separators')
        write(dest/name,image)
    (output/'manifest.json').write_text(json.dumps(chosen,indent=2),encoding='utf-8')
    return chosen


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preparación de ecografías limpias para LaMa.")
    parser.add_argument("--source", type=str, default="../data", help="Ruta a carpeta con ecografías originales.")
    parser.add_argument("--output", type=str, default="./dataset_ecografia_clean", help="Ruta destino para train/val.")
    parser.add_argument("--val_split", type=float, default=0.15, help="Proporción de validación (default 0.15).")
    parser.add_argument('--manifest', required=True, help='Manifiesto por paciente con clean_verified=true')
    args = parser.parse_args()

    print(f"Preparando dataset desde '{args.source}' hacia '{args.output}'...")
    prepare_lama_dataset(args.source, args.output, val_split=args.val_split, manifest_path=args.manifest)
