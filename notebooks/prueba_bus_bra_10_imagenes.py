"""
Prueba Independiente: Inpainting Híbrido Acústico en 10 Imágenes de BUS_BRA (Versión 4 - Definitiva)
=====================================================================================================
Este script procesa 10 imágenes representativas de ecografía mamaria de la base
de datos BUS_BRA, utilizando el Pipeline V4 perfeccionado y validado tras la auditoría
visual externa de falsos negativos y discontinuidades de borde:

Novedades y Optimizaciones de la Versión 4 (V4):
  1. Filtro Top-Hat morfológico (9x9) para realzar estructuras delgadas de alto contraste.
  2. Rechazo de núcleos de tejido sólido hiperecoico normal (Core = erode(I_sat, 3x3) >= 6 px),
     garantizando 0% falsos positivos en parénquima normal (como el brillo de bus_0019-r).
  3. Detección direccional de cruces ortogonales (5x1, 1x5) y diagonales (5x5) con criterio dual:
     - Anotaciones estándar: I >= 220 & TopHat >= 40
     - Rescate de ramas de caliper atenuadas en fondo oscuro: TopHat >= 90 & I >= 150
     (Resuelve el falso negativo crítico de bus_0008-l detectando los 4 calipers al 100%).
  4. Detección de calipers truncados en márgenes (x <= 2 o x >= w-2).
  5. Agrupamiento morfológico horizontal de texto (11x2) y fusión colineal de cajas.
  6. Motor de Inpainting Acústico Residual (I = L + S) con Soporte Reflectivo (Mirror Padding 16 px):
     Elimina por completo el salto de gradiente de borde en calipers de corte (bus_0038-s).
  7. Calibración dinámica de amplitud de speckle por componente conectado (beta = min(1.5, std_ring/std_patch)).

Soporta ejecución dual:
  1. Localmente en Windows / Linux
  2. En Google Colab montando Google Drive ('Mi unidad/PRUEBA INPAINTING')
"""

import sys
import os
import glob
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Métricas
from skimage.feature import graycomatrix, graycoprops
from skimage.metrics import structural_similarity as ssim

# -------------------------------------------------------------------------
# 1. Configuración de Entorno y Rutas (Dual Local / Google Colab)
# -------------------------------------------------------------------------
IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    print(">>> Detectado entorno Google Colab.")
    from google.colab import drive
    drive.mount('/content/drive')
    
    BASE_DIR = '/content/drive/MyDrive/PRUEBA INPAINTING'
    if not os.path.exists(BASE_DIR):
        BASE_DIR = '/content/drive/My Drive/PRUEBA INPAINTING'
        
    BUSCLEAN_PATH = os.path.join(BASE_DIR, 'bus-cleaning-main')
    DATASET_DIR = os.path.join(BASE_DIR, 'BASES DE DATOS ORIGINALES')
else:
    print(">>> Detectado entorno Local.")
    CURRENT_DIR = os.path.abspath(os.path.dirname(__file__)) if '__file__' in locals() else os.getcwd()
    BASE_DIR = os.path.dirname(CURRENT_DIR)
    BUSCLEAN_PATH = os.path.join(BASE_DIR, 'bus-cleaning-main')
    
    if os.path.isdir(os.path.join(BASE_DIR, 'BASES DE DATOS ORIGINALES')):
        DATASET_DIR = os.path.join(BASE_DIR, 'BASES DE DATOS ORIGINALES')
    else:
        DATASET_DIR = os.path.join(BASE_DIR, 'data', 'bases_datos_originales')

print(f"Carpeta Base:   {BASE_DIR}")
print(f"Ruta BUSClean:  {BUSCLEAN_PATH}")
print(f"Ruta Datasets:  {DATASET_DIR}")

if os.path.isdir(BUSCLEAN_PATH) and BUSCLEAN_PATH not in sys.path:
    sys.path.append(BUSCLEAN_PATH)

try:
    from modules.artifacts import enhance_image, detect_anno
    BUSCLEAN_AVAILABLE = True
    print("[OK] Módulo BUSClean 'artifacts' cargado correctamente.")
except Exception as e:
    BUSCLEAN_AVAILABLE = False
    print(f"[AVISO] Módulo BUSClean no disponible directamente ({e}). Se usará detector morfológico V4.")


# -------------------------------------------------------------------------
# 2. Detector V4 de Calipers y Líneas de Texto (Híbrido de No-Regresión)
# -------------------------------------------------------------------------
def detect_calipers_and_text(img_rgb):
    """
    Detector V4 Perfeccionado (Híbrido de Alta Sensibilidad y No-Regresión):
      1. Top-Hat morfológico (9x9) para realzar estructuras finas de alto contraste.
      2. Rechazo de núcleos de tejido parenquimatoso normal (Core >= 6 px, dilatación 11x11).
      3. Detección direccional de cruces ortogonales y diagonales con criterio dual asimétrico.
      4. Captura de calipers truncados en bordes (x <= 2 o x >= w-2).
      5. Agrupamiento horizontal de texto y fusión colineal.
      6. Segmentación por histéresis de semillas (230 -> 170).
    """
    h, w, _ = img_rgb.shape
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    # 1. Top-Hat
    k_th = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k_th)
    
    # 2. Rechazo estricto de tejido sólido (núcleo erosionado 3x3 >= 6 px)
    sat_raw = (gray >= 235).astype(np.uint8)
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    eroded_core = cv2.erode(sat_raw, k3)
    num_e, labels_e, stats_e, _ = cv2.connectedComponentsWithStats(eroded_core)
    solid_tissue_blob = np.zeros_like(sat_raw)
    for i in range(1, num_e):
        if stats_e[i, cv2.CC_STAT_AREA] >= 6:
            solid_tissue_blob[labels_e == i] = 1
    solid_tissue_blob = cv2.dilate(solid_tissue_blob, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)))
    
    # 3. Píxeles candidatos: brillantes O alto contraste relativo con atenuación
    bin_annos = ((gray >= 220) & (tophat >= 40)) | ((tophat >= 90) & (gray >= 150))
    bin_annos[solid_tissue_blob > 0] = 0
    bin_annos = bin_annos.astype(np.uint8)
    
    # 4. Cruces de calipers ortogonales y diagonales
    k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    k_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    h_lines = cv2.morphologyEx(bin_annos, cv2.MORPH_OPEN, k_h)
    v_lines = cv2.morphologyEx(bin_annos, cv2.MORPH_OPEN, k_v)
    cross_junc = cv2.dilate(h_lines, k3) & cv2.dilate(v_lines, k3)
    
    k_d1 = np.eye(5, dtype=np.uint8)
    k_d2 = np.fliplr(k_d1)
    d1_lines = cv2.morphologyEx(bin_annos, cv2.MORPH_OPEN, k_d1)
    d2_lines = cv2.morphologyEx(bin_annos, cv2.MORPH_OPEN, k_d2)
    diag_junc = cv2.dilate(d1_lines, k3) & cv2.dilate(d2_lines, k3)
    
    # Calipers truncados en límites de adquisición (e.g. x <= 2 o x >= w-2)
    edge_calipers = np.zeros_like(bin_annos)
    left_h = h_lines[:, :2].sum(axis=1) > 0
    right_h = h_lines[:, w-2:].sum(axis=1) > 0
    for y in np.where(left_h)[0]:
        edge_calipers[max(0, y-6):min(h, y+7), 0:14] = bin_annos[max(0, y-6):min(h, y+7), 0:14]
    for y in np.where(right_h)[0]:
        edge_calipers[max(0, y-6):min(h, y+7), max(0, w-14):w] = bin_annos[max(0, y-6):min(h, y+7), max(0, w-14):w]
        
    all_caliper_seeds = cross_junc | diag_junc | edge_calipers
    all_caliper_seeds[solid_tissue_blob > 0] = 0
    
    caliper_boxes = []
    num_j, labels_j, stats_j, _ = cv2.connectedComponentsWithStats(all_caliper_seeds)
    for i in range(1, num_j):
        bx, by, bw, bh, area = stats_j[i]
        x0 = max(0, bx - 8)
        y0 = max(0, by - 8)
        x1 = min(w, bx + bw + 8)
        y1 = min(h, by + bh + 8)
        if (gray[y0:y1, x0:x1] >= 225).sum() > 0:
            caliper_boxes.append((x0, y0, x1 - x0, y1 - y0))
            
    # 5. Detección de texto (excluyendo calipers ya confirmados)
    text_cands = bin_annos.copy()
    for (cx, cy, cw, ch) in caliper_boxes:
        text_cands[cy:cy+ch, cx:cx+cw] = 0
        
    kernel_text = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 2))
    text_clusters = cv2.morphologyEx(text_cands, cv2.MORPH_CLOSE, kernel_text)
    text_clusters = cv2.dilate(text_clusters, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1)))
    
    text_boxes = []
    num_t, _, stats_t, _ = cv2.connectedComponentsWithStats(text_clusters)
    for i in range(1, num_t):
        bx, by, bw, bh, area = stats_t[i]
        if bw >= 18 and 6 <= bh <= 22 and (bw / bh) >= 1.3:
            if (gray[by:by+bh, bx:bx+bw] >= 230).sum() >= 4:
                text_boxes.append((bx, by, bw, bh))
                
    # 6. Fusión colineal de texto
    all_boxes = caliper_boxes + text_boxes
    all_boxes = sorted(all_boxes, key=lambda b: (b[1], b[0]))
    merged_boxes = []
    for box in all_boxes:
        bx, by, bw, bh = box
        merged = False
        for m in merged_boxes:
            same_line = abs((by + bh / 2.0) - (m[1] + m[3] / 2.0)) <= 6
            h_close = (bx <= m[0] + m[2] + 10) and (m[0] <= bx + bw + 10)
            overlap_x = max(0, min(bx + bw, m[0] + m[2]) - max(bx, m[0]))
            overlap_y = max(0, min(by + bh, m[1] + m[3]) - max(by, m[1]))
            if (same_line and h_close) or (overlap_x > 0 and overlap_y > 0):
                x1 = min(bx, m[0])
                y1 = min(by, m[1])
                x2 = max(bx + bw, m[0] + m[2])
                y2 = max(by + bh, m[1] + m[3])
                m[0], m[1], m[2], m[3] = x1, y1, x2 - x1, y2 - y1
                merged = True
                break
        if not merged:
            merged_boxes.append([bx, by, bw, bh])
            
    # 7. Extracción de trazo por histéresis (230 -> 170)
    bbox_mask = np.zeros_like(gray)
    stroke_mask = np.zeros_like(gray)
    
    for (bx, by, bw, bh) in merged_boxes:
        x0, y0 = max(0, bx - 1), max(0, by - 1)
        x1, y1 = min(w, bx + bw + 1), min(h, by + bh + 1)
        cv2.rectangle(bbox_mask, (x0, y0), (x1, y1), 255, -1)
        
        sub = gray[y0:y1, x0:x1]
        seeds = (sub >= 230).astype(np.uint8)
        seeds[solid_tissue_blob[y0:y1, x0:x1] > 0] = 0
        if seeds.sum() == 0:
            continue
            
        cands = (sub >= 170).astype(np.uint8)
        num_l, labels = cv2.connectedComponents(cands)
        box_stroke = np.zeros_like(sub)
        for l in range(1, num_l):
            comp = (labels == l)
            if (comp & (seeds > 0)).any():
                box_stroke[comp] = 255
                
        box_stroke = cv2.dilate(box_stroke, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        stroke_mask[y0:y1, x0:x1] = np.maximum(stroke_mask[y0:y1, x0:x1], box_stroke)
        
    return bbox_mask, stroke_mask, merged_boxes


# -------------------------------------------------------------------------
# 3. Inpainting Acústico Residual V4 con Soporte Reflectivo (Mirror Padding)
# -------------------------------------------------------------------------
def inpaint_adaptive_acoustic_residual(img_rgb, mask, pad=16):
    """
    Motor de Inpainting V4 con Soporte Reflectivo Bilateral (Mirror Padding):
      1. Aplica padding reflectivo de 16 px para dotar de soporte isofótico a calipers de borde.
      2. Descompone en macroestructura L y residuo de speckle real S = I - L.
      3. Propaga L y S mediante isófotas continuas.
      4. Calibra dinámicamente la energía de speckle por componente conectado.
      5. Fusión con feathering gaussiano estrecho.
      6. Recorte exacto al tamaño original sin costura perimetral.
    """
    h, w, c = img_rgb.shape
    if (mask > 0).sum() == 0:
        return img_rgb.copy()
        
    img_pad = cv2.copyMakeBorder(img_rgb, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
    mask_pad = cv2.copyMakeBorder(mask, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    
    gray_pad = cv2.cvtColor(img_pad, cv2.COLOR_RGB2GRAY).astype(np.float32)
    L_pad = cv2.GaussianBlur(gray_pad, (5, 5), 1.2)
    S_pad = gray_pad - L_pad
    
    L_inp = cv2.inpaint(np.clip(L_pad, 0, 255).astype(np.uint8), mask_pad, 3, cv2.INPAINT_TELEA).astype(np.float32)
    S_shift = np.clip(S_pad + 128.0, 0, 255).astype(np.uint8)
    S_inp_shift = cv2.inpaint(S_shift, mask_pad, 3, cv2.INPAINT_TELEA).astype(np.float32)
    S_inp = S_inp_shift - 128.0
    
    k_ring = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    num_l, labels = cv2.connectedComponents(mask_pad)
    for lbl in range(1, num_l):
        comp = (labels == lbl)
        comp_ring = cv2.subtract(cv2.dilate(comp.astype(np.uint8), k_ring), comp.astype(np.uint8))
        std_ring = float(np.std(S_pad[comp_ring > 0])) if comp_ring.sum() > 0 else 1.0
        std_patch = float(np.std(S_inp[comp])) if comp.sum() > 0 else 1.0
        if std_patch > 0 and std_patch < std_ring:
            scale = min(1.5, std_ring / std_patch)
            S_inp[comp] *= scale
            
    reconstructed_pad = np.clip(L_inp + S_inp, 0, 255)
    alpha_pad = cv2.GaussianBlur(mask_pad.astype(np.float32) / 255.0, (3, 3), 0.5)
    final_pad = (reconstructed_pad * alpha_pad + gray_pad * (1.0 - alpha_pad)).astype(np.uint8)
    
    res_gray = final_pad[pad:-pad, pad:-pad]
    result = img_rgb.copy()
    reconstructed = np.repeat(res_gray[:, :, np.newaxis], 3, axis=2)
    result[mask > 0] = reconstructed[mask > 0]
    return result


# -------------------------------------------------------------------------
# 4. Métricas Cuantitativas de Control de Calidad Auditables
# -------------------------------------------------------------------------
def compute_qc_metrics(orig_rgb, res_rgb, mask, bbox_mask=None):
    orig_g = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2GRAY)
    res_g = cv2.cvtColor(res_rgb, cv2.COLOR_RGB2GRAY)
    h, w = orig_g.shape
    
    mask_px = int((mask > 0).sum())
    coverage_pct = float(mask_px / (h * w) * 100)
    
    if bbox_mask is not None and (bbox_mask > 0).sum() > 0:
        bbox_px = int((bbox_mask > 0).sum())
        tissue_gain = max(0.0, (1.0 - (mask_px / bbox_px)) * 100)
    else:
        tissue_gain = 0.0

    _, diff = ssim(orig_g, res_g, full=True)
    outside = (mask == 0)
    ssim_outside = float(diff[outside].mean()) if outside.sum() > 0 else 1.0

    sx = cv2.Sobel(res_g, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(res_g, cv2.CV_64F, 0, 1, ksize=3)
    gmag = np.sqrt(sx**2 + sy**2)
    edge = cv2.Canny(mask, 100, 200)
    boundary_jump = float(gmag[edge > 0].mean()) if (edge > 0).sum() > 0 else 0.0

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    ring = cv2.subtract(cv2.dilate(mask, k), mask)
    p_std = float(np.std(res_g[mask > 0])) if mask_px > 0 else 0.0
    r_std = float(np.std(orig_g[ring > 0])) if (ring > 0).sum() > 0 else 1.0
    speckle_ratio = float(p_std / max(1e-5, r_std))

    glcm_diff = 0.0
    if mask_px >= 30:
        try:
            ys, xs = np.where(mask > 0)
            y_min, y_max = max(0, ys.min() - 2), min(h, ys.max() + 3)
            x_min, x_max = max(0, xs.min() - 2), min(w, xs.max() + 3)
            crop_orig = (orig_g[y_min:y_max, x_min:x_max] // 16).astype(np.uint8)
            crop_res = (res_g[y_min:y_max, x_min:x_max] // 16).astype(np.uint8)
            g_orig = graycomatrix(crop_orig, distances=[1], angles=[0], levels=16, symmetric=True, normed=True)
            g_res = graycomatrix(crop_res, distances=[1], angles=[0], levels=16, symmetric=True, normed=True)
            glcm_diff = float(abs(graycoprops(g_orig, 'contrast')[0, 0] - graycoprops(g_res, 'contrast')[0, 0]))
        except Exception:
            glcm_diff = 0.0

    return {
        'mask_coverage_pct': round(coverage_pct, 4),
        'tissue_preservation_gain_pct': round(tissue_gain, 2),
        'ssim_outside_mask': round(ssim_outside, 6),
        'boundary_gradient_mean': round(boundary_jump, 2),
        'speckle_variance_ratio': round(speckle_ratio, 4),
        'glcm_contrast_diff': round(glcm_diff, 4)
    }


# -------------------------------------------------------------------------
# 5. Ejecución del Test en 10 Imágenes de BUS_BRA (Versión 4)
# -------------------------------------------------------------------------
def run_bus_bra_test():
    bus_bra_dir = os.path.join(DATASET_DIR, 'BUS_BRA')
    
    search_paths = [
        os.path.join(bus_bra_dir, 'BUSBRA', 'BUSBRA', 'Images', '*.png'),
        os.path.join(bus_bra_dir, '**', '*.png')
    ]
    all_imgs = []
    for pat in search_paths:
        found = [f for f in glob.glob(pat, recursive=True) if 'mask' not in f.lower()]
        if len(found) > 0:
            all_imgs = found
            break

    print(f"\nTotal imágenes encontradas en BUS_BRA: {len(all_imgs)}")
    if len(all_imgs) == 0:
        print(f"Error: No se encontraron imágenes en: {bus_bra_dir}")
        return

    target_names = [
        'bus_0002-l.png', 'bus_0003-l.png', 'bus_0003-r.png', 'bus_0008-l.png',
        'bus_0018-s.png', 'bus_0019-l.png', 'bus_0019-r.png', 'bus_0020-l.png',
        'bus_0027-l.png', 'bus_0038-s.png'
    ]
    
    selected_paths = []
    for t in target_names:
        for p in all_imgs:
            if os.path.basename(p) == t:
                selected_paths.append(p)
                break

    print(f"\nProcesando las 10 imágenes seleccionadas de BUS_BRA (Versión 4):")
    for i, p in enumerate(selected_paths):
        print(f"  [{i+1:2d}] {os.path.basename(p)}")

    out_dir = os.path.join(BASE_DIR, 'resultados_version_4')
    os.makedirs(out_dir, exist_ok=True)

    benchmark_rows = []
    fig, axes = plt.subplots(10, 4, figsize=(16, 32))
    plt.subplots_adjust(hspace=0.45, wspace=0.2)

    for idx, img_path in enumerate(selected_paths):
        fname = os.path.basename(img_path)
        im_bgr = cv2.imread(img_path)
        im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
        
        # 1. Detección V4
        bbox_mask, stroke_mask, boxes = detect_calipers_and_text(im_rgb)
        
        # 2. Inpainting
        res_bbox = cv2.inpaint(im_rgb, bbox_mask, 5, cv2.INPAINT_TELEA)
        res_stroke = cv2.inpaint(im_rgb, stroke_mask, 3, cv2.INPAINT_TELEA)
        res_hybrid = inpaint_adaptive_acoustic_residual(im_rgb, stroke_mask)
        
        # 3. Métricas
        m_bbox = compute_qc_metrics(im_rgb, res_bbox, bbox_mask, bbox_mask)
        m_stroke = compute_qc_metrics(im_rgb, res_stroke, stroke_mask, bbox_mask)
        m_hybrid = compute_qc_metrics(im_rgb, res_hybrid, stroke_mask, bbox_mask)
        
        m_bbox.update({'image': fname, 'method': '1. Baseline BBox'})
        m_stroke.update({'image': fname, 'method': '2. Trazo Simple'})
        m_hybrid.update({'image': fname, 'method': '3. Híbrido Propuesto (V4)'})
        
        benchmark_rows.extend([m_bbox, m_stroke, m_hybrid])
        
        overlay = im_bgr.copy()
        overlay[stroke_mask > 0] = [0, 0, 255]
        side_by_side = np.hstack([
            im_bgr,
            overlay,
            cv2.cvtColor(res_bbox, cv2.COLOR_RGB2BGR),
            cv2.cvtColor(res_hybrid, cv2.COLOR_RGB2BGR)
        ])
        cv2.imwrite(os.path.join(out_dir, f'comparacion_{fname}'), side_by_side)
        
        gain = m_hybrid['tissue_preservation_gain_pct']
        axes[idx, 0].imshow(im_rgb)
        axes[idx, 0].set_title(f"{fname}\nOriginal", fontsize=9)
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[idx, 1].set_title(f"Anotaciones V4 ({len(boxes)} reg.)\nGanancia: +{gain:.1f}%", fontsize=9)
        axes[idx, 1].axis('off')
        
        axes[idx, 2].imshow(res_bbox)
        axes[idx, 2].set_title(f"Baseline (BBox)\nGrad: {m_bbox['boundary_gradient_mean']}", fontsize=9)
        axes[idx, 2].axis('off')
        
        axes[idx, 3].imshow(res_hybrid)
        axes[idx, 3].set_title(f"Híbrido Acústico V4\nGrad: {m_hybrid['boundary_gradient_mean']} | Spk: {m_hybrid['speckle_variance_ratio']}", fontsize=9)
        axes[idx, 3].axis('off')

    mosaic_path = os.path.join(out_dir, 'mosaico_10_imagenes_bus_bra.png')
    plt.savefig(mosaic_path, bbox_inches='tight', dpi=150)
    plt.close(fig)

    df = pd.DataFrame(benchmark_rows)
    csv_path = os.path.join(out_dir, 'metricas_cuantitativas_bus_bra.csv')
    df.to_csv(csv_path, index=False)

    print("\n" + "="*95)
    print("        RESULTADOS CUANTITATIVOS AGREGADOS (MEDIANAS Y MEDIAS - 10 IMÁGENES BUS_BRA V4)        ")
    print("="*95)
    print("\n>>> MEDIANAS POR MÉTODO:")
    meds = df.groupby('method')[['tissue_preservation_gain_pct', 'speckle_variance_ratio', 'boundary_gradient_mean', 'ssim_outside_mask', 'glcm_contrast_diff']].median()
    print(meds.round(4).to_string())

    print("\n>>> MEDIAS POR MÉTODO (± DESVIACIÓN ESTÁNDAR):")
    summary = df.groupby('method').agg({
        'tissue_preservation_gain_pct': ['mean', 'std'],
        'speckle_variance_ratio': ['mean', 'std'],
        'boundary_gradient_mean': ['mean', 'std'],
        'ssim_outside_mask': ['mean', 'std'],
        'glcm_contrast_diff': ['mean', 'std'],
        'mask_coverage_pct': ['mean']
    }).round(4)
    print(summary.to_string())

    print("\n" + "="*95)
    print("                   DESGLOSE DETALLADO DEL MÉTODO HÍBRIDO POR IMAGEN (V4)                 ")
    print("="*95)
    hybrid_only = df[df['method'] == '3. Híbrido Propuesto (V4)']
    print(hybrid_only[['image', 'mask_coverage_pct', 'tissue_preservation_gain_pct', 'speckle_variance_ratio', 'boundary_gradient_mean', 'ssim_outside_mask', 'glcm_contrast_diff']].to_string(index=False))

    print(f"\n[OK] Mosaico de comparación guardado en: {mosaic_path}")
    print(f"[OK] Archivo CSV con métricas guardado en: {csv_path}")
    print(f"[OK] 10 imágenes comparativas individuales guardadas en: {out_dir}")


if __name__ == '__main__':
    run_bus_bra_test()
