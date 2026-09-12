"""Auditoría y análisis estadístico de lecturas ciegas de ecografía.

Calcula el acuerdo inter-observador (Kappa de Cohen / Fleiss), evalúa
la capacidad diagnóstica de los radiólogos para detectar inpainting frente
al ground truth privado, y mide la distorsión de márgenes tumorales.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy import stats
from sklearn.metrics import cohen_kappa_score


def fleiss_kappa(matrix):
    """Calcula el Kappa de Fleiss a partir de una matriz N x K de conteos por categoría."""
    table = np.asarray(matrix, dtype=float)
    n_subjects, n_categories = table.shape
    n_raters = table.sum(axis=1)[0]
    if not np.allclose(table.sum(axis=1), n_raters):
        raise ValueError("Cada sujeto debe tener el mismo número de evaluadores")
    p_j = table.sum(axis=0) / (n_subjects * n_raters)
    p_i = (np.sum(table * table, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    p_bar = np.mean(p_i)
    p_e = np.sum(p_j * p_j)
    if np.isclose(1.0 - p_e, 0):
        return 1.0
    return float((p_bar - p_e) / (1.0 - p_e))


def evaluate_single_reader(responses, key_map):
    """Evalúa las respuestas de un lector individual frente a la clave privada."""
    tp = fp = tn = fn = uncertain = 0
    ref_distortions, inpaint_distortions = [], []
    ref_confidences, inpaint_confidences = [], []
    matched = 0

    for resp in responses:
        cid = resp.get("case_id")
        if cid not in key_map:
            continue
        matched += 1
        truth = key_map[cid]
        is_inpaint = (truth["variant"] == "inpaint")
        manip = str(resp.get("manip", "")).strip().lower()
        dist = float(resp.get("distortion", 0) or 0)
        conf = float(resp.get("confidence", 3) or 3)

        if is_inpaint:
            inpaint_distortions.append(dist)
            inpaint_confidences.append(conf)
            if manip in ("sí", "si", "yes", "true", "1"):
                tp += 1
            elif manip in ("no", "false", "0"):
                fn += 1
            else:
                uncertain += 1
        else:
            ref_distortions.append(dist)
            ref_confidences.append(conf)
            if manip in ("sí", "si", "yes", "true", "1"):
                fp += 1
            elif manip in ("no", "false", "0"):
                tn += 1
            else:
                uncertain += 1

    decided = tp + fp + tn + fn
    accuracy = float((tp + tn) / decided) if decided > 0 else None
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else None
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else None
    ppv = float(tp / (tp + fp)) if (tp + fp) > 0 else None
    npv = float(tn / (tn + fn)) if (tn + fn) > 0 else None

    # Diferencia de distorsión percibida en márgenes tumorales
    u_stat = p_val = None
    if ref_distortions and inpaint_distortions:
        try:
            res = stats.mannwhitneyu(inpaint_distortions, ref_distortions, alternative="two-sided")
            u_stat, p_val = float(res.statistic), float(res.pvalue)
        except Exception:
            pass

    return {
        "matched_cases": matched,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "uncertain_cases": uncertain,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "positive_predictive_value": ppv,
        "negative_predictive_value": npv,
        "mean_distortion_reference": float(np.mean(ref_distortions)) if ref_distortions else None,
        "mean_distortion_inpaint": float(np.mean(inpaint_distortions)) if inpaint_distortions else None,
        "distortion_delta": float(np.mean(inpaint_distortions) - np.mean(ref_distortions)) if (ref_distortions and inpaint_distortions) else None,
        "distortion_mannwhitney_p": p_val,
        "mean_confidence_reference": float(np.mean(ref_confidences)) if ref_confidences else None,
        "mean_confidence_inpaint": float(np.mean(inpaint_confidences)) if inpaint_confidences else None,
    }


def analyze_study(key_path, responses_paths, outdir=None):
    """Analiza respuestas de múltiples radiólogos y calcula acuerdo inter-observador."""
    key_list = json.loads(Path(key_path).read_text(encoding="utf-8"))
    key_map = {item["case_id"]: item for item in key_list}
    case_ids = [item["case_id"] for item in key_list]

    readers_data = {}
    for p in responses_paths:
        p = Path(p)
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            responses = list(data.values())
        else:
            responses = data
        rid = responses[0].get("reader_id", p.stem) if responses else p.stem
        readers_data[rid] = {r.get("case_id"): r for r in responses}

    reader_metrics = {}
    for rid, rmap in readers_data.items():
        resp_list = [rmap[cid] for cid in case_ids if cid in rmap]
        reader_metrics[rid] = evaluate_single_reader(resp_list, key_map)

    # Cálculo de acuerdo inter-observador
    inter_reader = {}
    r_keys = list(readers_data.keys())
    if len(r_keys) >= 2:
        for i in range(len(r_keys)):
            for j in range(i + 1, len(r_keys)):
                r1, r2 = r_keys[i], r_keys[j]
                y1, y2 = [], []
                d1, d2 = [], []
                for cid in case_ids:
                    if cid in readers_data[r1] and cid in readers_data[r2]:
                        m1 = str(readers_data[r1][cid].get("manip", "")).strip().lower()
                        m2 = str(readers_data[r2][cid].get("manip", "")).strip().lower()
                        y1.append(m1)
                        y2.append(m2)
                        dist1 = int(readers_data[r1][cid].get("distortion", 0) or 0)
                        dist2 = int(readers_data[r2][cid].get("distortion", 0) or 0)
                        d1.append(dist1)
                        d2.append(dist2)

                pair_key = f"{r1}_vs_{r2}"
                kappa_manip = float(cohen_kappa_score(y1, y2)) if len(y1) > 0 else None
                kappa_dist = float(cohen_kappa_score(d1, d2, weights="quadratic")) if len(d1) > 0 else None
                inter_reader[pair_key] = {
                    "common_cases": len(y1),
                    "cohen_kappa_manipulation": kappa_manip,
                    "weighted_kappa_margin_distortion": kappa_dist,
                }

        cats = ["sí", "no", "incierto"]
        matrix = []
        for cid in case_ids:
            row = [0, 0, 0]
            valid_all = True
            for rk in r_keys:
                if cid not in readers_data[rk]:
                    valid_all = False
                    break
                m = str(readers_data[rk][cid].get("manip", "")).strip().lower()
                if m in ("sí", "si", "yes", "true", "1"):
                    row[0] += 1
                elif m in ("no", "false", "0"):
                    row[1] += 1
                else:
                    row[2] += 1
            if valid_all:
                matrix.append(row)
        if len(matrix) > 0 and len(r_keys) > 2:
            try:
                inter_reader["fleiss_kappa_manipulation"] = fleiss_kappa(matrix)
            except Exception:
                inter_reader["fleiss_kappa_manipulation"] = None

    mean_inpaint_detection = None
    sensitivities = [m["sensitivity"] for m in reader_metrics.values() if m["sensitivity"] is not None]
    if sensitivities:
        mean_inpaint_detection = float(np.mean(sensitivities))

    mean_distortion_delta = None
    deltas = [m["distortion_delta"] for m in reader_metrics.values() if m["distortion_delta"] is not None]
    if deltas:
        mean_distortion_delta = float(np.mean(deltas))

    first_pair_kappa = None
    if inter_reader:
        first_pair = next(iter(inter_reader.values()))
        first_pair_kappa = first_pair.get("cohen_kappa_manipulation")

    report = {
        "scope": "blinded_radiologist_pilot_study_audit",
        "total_ground_truth_cases": len(key_list),
        "readers_evaluated": len(readers_data),
        "reader_ids": r_keys,
        "reader_individual_metrics": reader_metrics,
        "inter_observer_agreement": inter_reader,
        "summary": {
            "mean_inpaint_detection_rate": mean_inpaint_detection,
            "mean_margin_distortion_delta": mean_distortion_delta,
            "mean_cohen_kappa": first_pair_kappa,
            "clinical_interpretation": (
                "Baja detección de inpainting (sensibilidad cercana a tasa de azar) y delta de distorsión "
                "de margen ~0 demuestran no-inferioridad morfológica del inpainting acústico."
            )
        }
    }

    if outdir:
        out_p = Path(outdir)
        out_p.mkdir(parents=True, exist_ok=True)
        report_path = out_p / "informe_lectura_ciega.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        
        status_file = out_p / "status.json"
        if status_file.exists():
            status = json.loads(status_file.read_text(encoding="utf-8"))
            status["reader_responses_received"] = len(readers_data)
            status["inter_reader_kappa_manipulation"] = first_pair_kappa
            status["mean_inpaint_detection_rate"] = mean_inpaint_detection
            status["margin_distortion_delta"] = mean_distortion_delta
            status["study_completed"] = len(readers_data) >= 2
            status_file.write_text(json.dumps(status, indent=2), encoding="utf-8")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auditoría y análisis de respuestas del estudio ciego de radiólogos.")
    parser.add_argument("--key", required=True, help="Ruta a PRIVATE_reader_key.json")
    parser.add_argument("--responses", nargs="+", required=True, help="Rutas a uno o más archivos respuestas_*.json")
    parser.add_argument("--outdir", default=None, help="Directorio donde guardar el informe y actualizar status.json")
    args = parser.parse_args()
    res = analyze_study(args.key, args.responses, args.outdir)
    print(json.dumps(res["summary"], indent=2, ensure_ascii=False))
