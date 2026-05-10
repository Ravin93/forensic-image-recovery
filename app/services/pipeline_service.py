from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.core.logger import logger
from app.modules.classification.corruption_classifier import classify_corruption_type
from app.modules.corruption.detection import detect_advanced_mask, detect_dark_regions_mask
from app.modules.corruption.policies import build_corruption_profile
from app.modules.corruption.simulator import corrupt_image
from app.modules.evaluation.comparator import compare_images
from app.modules.evaluation.detection_metrics import compute_iou, compute_precision_recall
from app.modules.evaluation.recoverability import classify_recoverability
from app.modules.reconstruction.repair_pipeline import run_repair_pipeline
from app.modules.reporting.json_report import build_report, save_json_report
from app.modules.validation.verifier import verify_image


def _load_mask_array(mask_path: str | Path) -> np.ndarray:
    return np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)


def run_demo_pipeline(
    source_image_path: str | Path,
    corruption_type: str = "rectangle_mask",
    corruption_params: dict[str, Any] | None = None,
    inpaint_method: str = "opencv_inpaint",
    radius: int = 3,
    seed: int | None = None,
    corruption_level: int | None = None,
    execution_mode: str = "assisted",
    detection_mode: str = "basic",
    max_attempts: int = 8,
    randomize: bool = False,
) -> dict[str, Any]:
    """Pipeline complet : corruption → détection → reconstruction multi-essais → scoring.

    Retourne un dict structuré exposant :
        original_image          chemin de l'image source
        corrupted_image         chemin de l'image corrompue
        mask_path               chemin du masque utilisé pour la reconstruction
        reconstructed_image     chemin de la meilleure reconstruction
        selected_repair_strategy  stratégie sélectionnée automatiquement
        repair_score            score 0-100 de la meilleure reconstruction
        retry_count             nombre de candidats testés (hors conservatif)
        repair_candidates       liste complète des candidats avec scores
        evaluation              comparaison PSNR/SSIM original/corrompu/reconstruit
        recoverability_status   'good' | 'partial' | 'poor'
    """
    source_image_path = Path(source_image_path)

    # ------------------------------------------------------------------
    # 1. Profil de corruption (niveau ou params explicites)
    # ------------------------------------------------------------------
    if corruption_level is not None:
        profile = build_corruption_profile(corruption_level)
        corruption_type = profile["corruption_type"]
        corruption_params = profile["corruption_params"]
        randomize = profile["randomize"]
        imperfect_mask = profile["imperfect_mask"]
    else:
        profile = None
        randomize = randomize  # conserve la valeur passée en argument
        imperfect_mask = False

    if not corruption_params:  # None ou dict vide
        corruption_params = {} if randomize else {"x": 20, "y": 20, "width": 40, "height": 40, "fill_value": 0}

    # ------------------------------------------------------------------
    # 2. Validation de l'image source
    # ------------------------------------------------------------------
    validation = verify_image(source_image_path, allowed_formats={"JPEG", "JPG", "PNG"})
    if not validation["valid"]:
        raise ValueError(f"Image source invalide : {validation['reason']}")

    # ------------------------------------------------------------------
    # 3. Corruption
    # ------------------------------------------------------------------
    corruption_kwargs = dict(corruption_params)
    if seed is not None and "seed" not in corruption_kwargs:
        corruption_kwargs["seed"] = seed

    corruption_result = corrupt_image(
        image_path=source_image_path,
        corruption_type=corruption_type,
        randomize=randomize,
        imperfect_mask=imperfect_mask,
        **corruption_kwargs,
    )

    # ------------------------------------------------------------------
    # 4. Détection du masque selon le mode d'exécution
    # ------------------------------------------------------------------
    true_mask_path = corruption_result["mask_path"]
    detected_mask_path = true_mask_path
    corruption_classification = None
    detection_confidence = 0.0

    if execution_mode == "assisted":
        detected_mask_path = true_mask_path
        detection_confidence = 1.0

    elif execution_mode == "blind_basic":
        detected_mask = detect_dark_regions_mask(corruption_result["path"])
        detected_mask_path = str(
            Path(true_mask_path).with_name(
                Path(true_mask_path).stem + "_detected_basic.png"
            )
        )
        detected_mask.save(detected_mask_path)
        detection_confidence = 0.35

    elif execution_mode == "blind_advanced":
        if detection_mode == "advanced":
            det = detect_advanced_mask(corruption_result["path"])
            detected_mask = det["mask"]
            detection_confidence = float(det.get("confidence", 0.0))
        else:
            detected_mask = detect_dark_regions_mask(corruption_result["path"])
            detection_confidence = 0.35

        detected_mask_path = str(
            Path(true_mask_path).with_name(
                Path(true_mask_path).stem + "_detected_advanced.png"
            )
        )
        detected_mask.save(detected_mask_path)

        corruption_classification = classify_corruption_type(
            image_path=corruption_result["path"],
            detected_mask=_load_mask_array(detected_mask_path),
        )

    else:
        raise ValueError(f"execution_mode non supporté : {execution_mode}")

    # ------------------------------------------------------------------
    # 5. Métriques de détection (IoU, précision, rappel)
    # ------------------------------------------------------------------
    true_mask = _load_mask_array(true_mask_path)
    pred_mask = _load_mask_array(detected_mask_path)
    detection_metrics = {
        "iou": compute_iou(true_mask, pred_mask),
        **compute_precision_recall(true_mask, pred_mask),
    }

    estimated_corruption_type = (
        corruption_classification["corruption_type"]
        if corruption_classification
        else "mask_like"
    )

    # ------------------------------------------------------------------
    # 6. Reconstruction multi-essais (Ticket 3)
    # ------------------------------------------------------------------
    reconstruction_result = run_repair_pipeline(
        corrupted_image_path=corruption_result["path"],
        mask_path=detected_mask_path,
        method=inpaint_method,
        radius=radius,
        corruption_type=estimated_corruption_type,
        detection_confidence=detection_confidence,
        original_image_path=source_image_path,
        max_attempts=max_attempts,
    )

    # ------------------------------------------------------------------
    # 7. Évaluation PSNR / SSIM
    # ------------------------------------------------------------------
    comparison_result = compare_images(
        original_path=source_image_path,
        corrupted_path=corruption_result["path"],
        reconstructed_path=reconstruction_result["path"],
    )

    recoverability_status = classify_recoverability(comparison_result)

    # ------------------------------------------------------------------
    # 8. Rapport JSON
    # ------------------------------------------------------------------
    report = build_report(
        source_image=str(source_image_path),
        corruption_result=corruption_result,
        reconstruction_result=reconstruction_result,
        comparison_result=comparison_result,
        status="completed",
        extra={
            "execution_mode": execution_mode,
            "detection_mode": detection_mode,
            "detected_mask_path": detected_mask_path,
            "detection_metrics": detection_metrics,
            "corruption_classification": corruption_classification,
            "selected_repair_strategy": reconstruction_result.get("selected_repair_strategy"),
            "recoverability_status": recoverability_status,
            "detection_confidence": detection_confidence,
            "max_attempts": max_attempts,
        },
    )
    report_path = save_json_report(report)

    # ------------------------------------------------------------------
    # 9. Retour structuré — Ticket 2 + Ticket 3
    # ------------------------------------------------------------------
    return {
        # --- Ticket 2 : les 3 images côte à côte ---
        "original_image":       str(source_image_path),
        "corrupted_image":      corruption_result["image_path"],
        "mask_path":            detected_mask_path,
        "reconstructed_image":  reconstruction_result["path"],

        # --- Ticket 3 : résultats multi-reconstruction ---
        "selected_repair_strategy": reconstruction_result.get("selected_repair_strategy"),
        "repair_score":             reconstruction_result.get("score"),
        "retry_count":              reconstruction_result.get("retry_count", 0),
        "repair_candidates":        reconstruction_result.get("candidates", []),

        # --- Évaluation ---
        "evaluation":           comparison_result,
        "recoverability_status": recoverability_status,
        "detection_metrics":    detection_metrics,

        # --- Contexte ---
        "corruption":           corruption_result,
        "reconstruction":       reconstruction_result,
        "report_path":          str(report_path),
        "status":               "completed",
        "validation":           validation,
        "corruption_classification": corruption_classification,
        "execution_mode":       execution_mode,
        "detection_mode":       detection_mode,
        "detected_mask_path":   detected_mask_path,
        "detection_confidence": detection_confidence,
        "mode": {
            "seed":              seed,
            "max_attempts":      max_attempts,
            "corruption_level":  corruption_level,
            "corruption_profile": profile,
        },
    }