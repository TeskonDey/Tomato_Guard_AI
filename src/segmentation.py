"""
Leaf Preprocessing, Segmentation, and Infection Severity Quantification
"""

import cv2
import numpy as np


def extract_leaf_mask(image_bgr: np.ndarray) -> np.ndarray:
    """
    Isolates the leaf from the background using color analysis and morphological operations.
    Works for plain neutral backgrounds as well as natural field backgrounds.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)

    h_channel = hsv[:, :, 0]
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]
    a_channel = lab[:, :, 1]
    b_channel = lab[:, :, 2]

    # Leaf pixels: green, yellowish-chlorotic, and necrotic brown
    green_mask = (h_channel >= 20) & (h_channel <= 95) & (s_channel > 25) & (v_channel > 25)
    chlorotic_mask = (h_channel >= 10) & (h_channel < 25) & (s_channel > 30) & (v_channel > 30)
    non_gray_mask = (np.abs(a_channel.astype(float) - 128) > 5) | (np.abs(b_channel.astype(float) - 128) > 6)

    combined_candidate = ((green_mask | chlorotic_mask) & non_gray_mask).astype(np.uint8) * 255

    # Fallback to Otsu thresholding if sparse
    if np.count_nonzero(combined_candidate) < (image_bgr.shape[0] * image_bgr.shape[1] * 0.02):
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, combined_candidate = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    cleaned = cv2.morphologyEx(combined_candidate, cv2.MORPH_CLOSE, kernel_small, iterations=2)
    dilated = cv2.dilate(cleaned, kernel_small, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    leaf_mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)

    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        max_area = cv2.contourArea(contours[0])
        valid_contours = [c for c in contours if cv2.contourArea(c) > max_area * 0.08]
        cv2.drawContours(leaf_mask, valid_contours, -1, 255, thickness=cv2.FILLED)
        leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel_large, iterations=1)

    return leaf_mask


def segment_leaf(image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Extracts the leaf from the background.
    Returns (segmented_leaf_bgr, leaf_mask).
    """
    leaf_mask = extract_leaf_mask(image_bgr)
    segmented = cv2.bitwise_and(image_bgr, image_bgr, mask=leaf_mask)
    return segmented, leaf_mask


def segment_lesions(image_bgr: np.ndarray, leaf_mask: np.ndarray) -> np.ndarray:
    """
    Detects genuine fungal lesions while properly ignoring healthy green shadows and surface sheen.
    """
    if np.count_nonzero(leaf_mask) == 0:
        return np.zeros(image_bgr.shape[:2], dtype=np.uint8)

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    b, g, r = cv2.split(image_bgr)

    h_channel = hsv[:, :, 0]
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]
    a_channel = lab[:, :, 1]
    b_channel = lab[:, :, 2]

    # 1. Healthy green definition (tolerant of natural shadows and deep green foliage):
    is_green_hue = (h_channel >= 28) & (h_channel <= 95)
    is_green_dominant = (g.astype(float) > r.astype(float) * 1.04) | (a_channel < 125)
    healthy_green = is_green_hue & is_green_dominant & (leaf_mask > 0)

    # 2. Genuine fungal lesions:
    # A. Necrotic brown/black spots: warm hue or non-green with lower brightness
    is_warm_hue = (h_channel < 25) | (h_channel > 165)
    necrotic_brown = is_warm_hue & (s_channel > 30) & (v_channel < 185) & (leaf_mask > 0)
    very_dark_dead = (v_channel < 45) & (~healthy_green) & (leaf_mask > 0)

    # B. Chlorosis / yellow halos: high yellow chroma in LAB or yellow hue with high saturation
    chlorosis = (h_channel >= 16) & (h_channel <= 35) & (s_channel > 50) & (b_channel > 138) & (leaf_mask > 0)

    # C. Gray mold / fuzzy patches: dull saturation, neutral grey, non-green dominant
    gray_fungus = (s_channel < 38) & (v_channel >= 65) & (v_channel <= 170) & (~healthy_green) & (np.abs(r.astype(float) - g.astype(float)) < 15) & (leaf_mask > 0)

    raw_lesions = (necrotic_brown | very_dark_dead | chlorosis | gray_fungus) & (~healthy_green) & (leaf_mask > 0)
    lesion_mask = raw_lesions.astype(np.uint8) * 255

    # Morphological clean up of single pixel noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    lesion_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    return lesion_mask


def calculate_severity(leaf_mask: np.ndarray, lesion_mask: np.ndarray) -> dict:
    """
    Computes infection area percentage and agricultural severity grade.
    """
    leaf_pixels = int(np.count_nonzero(leaf_mask))
    lesion_pixels = int(np.count_nonzero(lesion_mask))

    if leaf_pixels == 0:
        return {
            "severity_pct": 0.0,
            "stage": "Undetermined",
            "leaf_pixels": 0,
            "lesion_pixels": 0,
            "badge_color": "#9E9E9E",
            "summary": "No leaf detected in image."
        }

    severity_pct = round((lesion_pixels / leaf_pixels) * 100.0, 2)

    if severity_pct < 2.0:
        stage = "Healthy / Negligible"
        color = "#2E7D32"
        summary = "No visible disease lesions or negligible stress detected."
    elif severity_pct < 10.0:
        stage = "Mild (Stage 1)"
        color = "#FBC02D"
        summary = "Early localized infection. Immediate preventive action recommended to stop spread."
    elif severity_pct <= 25.0:
        stage = "Moderate (Stage 2)"
        color = "#F57C00"
        summary = "Established disease with visible spread. Targeted therapeutic treatment needed."
    else:
        stage = "Severe (Stage 3)"
        color = "#D32F2F"
        summary = "Extensive leaf tissue damage. Aggressive intervention required to protect yield."

    return {
        "severity_pct": severity_pct,
        "stage": stage,
        "leaf_pixels": leaf_pixels,
        "lesion_pixels": lesion_pixels,
        "badge_color": color,
        "summary": summary
    }


def create_diagnostic_overlay(image_bgr: np.ndarray, leaf_mask: np.ndarray, lesion_mask: np.ndarray) -> np.ndarray:
    """
    Generates a color-coded visual overlay showing leaf contour (green) and lesion areas (red/yellow).
    """
    overlay = image_bgr.copy()

    leaf_contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, leaf_contours, -1, (0, 255, 0), 2)

    red_layer = np.zeros_like(image_bgr)
    red_layer[:, :] = (0, 0, 230)
    lesion_highlight = cv2.bitwise_and(red_layer, red_layer, mask=lesion_mask)

    alpha = 0.45
    cv2.addWeighted(lesion_highlight, alpha, overlay, 1.0, 0, overlay)

    lesion_contours, _ = cv2.findContours(lesion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, lesion_contours, -1, (0, 255, 255), 1)

    return overlay
