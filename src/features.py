"""
Feature Extraction Engine: 68 Multi-Domain Handcrafted Features
(48 Color, 14 GLCM & Statistical Texture, 6 Morphological Shape Features)
"""

import cv2
import numpy as np
from scipy import stats
from skimage.feature import graycomatrix, graycoprops


def extract_color_features(image_bgr: np.ndarray, leaf_mask: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """
    Extracts 48 color features across 4 color spaces (RGB, HSV, L*a*b*, YCbCr).
    For each band: Mean, Max, Standard Deviation, Median.
    12 bands x 4 stats = 48 features.
    """
    # Convert image into 4 target color spaces
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    ycbcr = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)  # Y, Cr, Cb

    color_spaces = {
        "RGB": (rgb, ["R", "G", "B"]),
        "HSV": (hsv, ["H", "S", "V"]),
        "LAB": (lab, ["L", "a", "b"]),
        "YCbCr": (ycbcr, ["Y", "Cr", "Cb"])
    }

    mask_bool = leaf_mask > 0
    # Fallback to whole image if leaf mask is empty
    if not np.any(mask_bool):
        mask_bool = np.ones(image_bgr.shape[:2], dtype=bool)

    features = []
    names = []

    for space_name, (img_space, band_names) in color_spaces.items():
        for i, band_name in enumerate(band_names):
            channel_pixels = img_space[:, :, i][mask_bool].astype(float)

            if len(channel_pixels) == 0:
                mean_val = max_val = std_val = median_val = 0.0
            else:
                mean_val = float(np.mean(channel_pixels))
                max_val = float(np.max(channel_pixels))
                std_val = float(np.std(channel_pixels))
                median_val = float(np.median(channel_pixels))

            features.extend([mean_val, max_val, std_val, median_val])
            names.extend([
                f"{space_name}_{band_name}_mean",
                f"{space_name}_{band_name}_max",
                f"{space_name}_{band_name}_std",
                f"{space_name}_{band_name}_median"
            ])

    return np.array(features, dtype=np.float32), names


def extract_texture_features(image_bgr: np.ndarray, leaf_mask: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """
    Extracts 14 textural features from the leaf surface using GLCM and pixel statistics:
    Entropy, Energy, Contrast, Homogeneity, Correlation, Dissimilarity, ASM,
    Variance, Smoothness, Kurtosis, Skewness, RMS, Mean, IDM.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    mask_bool = leaf_mask > 0
    if not np.any(mask_bool):
        mask_bool = np.ones(image_bgr.shape[:2], dtype=bool)

    leaf_pixels = gray[mask_bool].astype(float)
    if len(leaf_pixels) == 0:
        leaf_pixels = np.array([0.0])

    # 1. Gray-Level Co-occurrence Matrix (quantize to 16 levels for fast computation)
    gray_quantized = (gray // 16).astype(np.uint8)
    glcm = graycomatrix(gray_quantized, distances=[1], angles=[0], levels=16, symmetric=True, normed=True)

    contrast = float(graycoprops(glcm, 'contrast')[0, 0])
    dissimilarity = float(graycoprops(glcm, 'dissimilarity')[0, 0])
    homogeneity = float(graycoprops(glcm, 'homogeneity')[0, 0])
    energy = float(graycoprops(glcm, 'energy')[0, 0])
    correlation = float(graycoprops(glcm, 'correlation')[0, 0])
    asm = float(graycoprops(glcm, 'ASM')[0, 0])

    # 2. GLCM Entropy & Inverse Difference Moment (IDM)
    glcm_flat = glcm.flatten()
    glcm_nz = glcm_flat[glcm_flat > 0]
    glcm_entropy = -float(np.sum(glcm_nz * np.log2(glcm_nz))) if len(glcm_nz) > 0 else 0.0

    # IDM calculation
    idm = 0.0
    for i in range(16):
        for j in range(16):
            idm += glcm[i, j, 0, 0] / (1.0 + (i - j) ** 2)
    idm = float(idm)

    # 3. First-order and higher-order distribution metrics on leaf pixels
    mean_val = float(np.mean(leaf_pixels))
    var_val = float(np.var(leaf_pixels))
    std_val = float(np.std(leaf_pixels))
    rms_val = float(np.sqrt(np.mean(leaf_pixels ** 2)))

    # Smoothness measure R = 1 - 1 / (1 + variance / (255^2))
    norm_var = var_val / (255.0 ** 2)
    smoothness = float(1.0 - (1.0 / (1.0 + norm_var)))

    skewness = float(stats.skew(leaf_pixels)) if std_val > 1e-5 else 0.0
    kurtosis = float(stats.kurtosis(leaf_pixels)) if std_val > 1e-5 else 0.0

    features = [
        glcm_entropy, energy, contrast, homogeneity, correlation,
        dissimilarity, asm, var_val, smoothness, kurtosis,
        skewness, rms_val, mean_val, idm
    ]

    names = [
        "texture_entropy", "texture_energy", "texture_contrast", "texture_homogeneity",
        "texture_correlation", "texture_dissimilarity", "texture_asm", "texture_variance",
        "texture_smoothness", "texture_kurtosis", "texture_skewness", "texture_rms",
        "texture_mean", "texture_idm"
    ]

    return np.array(features, dtype=np.float32), names


def extract_shape_features(lesion_mask: np.ndarray, leaf_mask: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """
    Extracts 6 geometric and morphological shape features from detected lesions:
    1. Perimeter, 2. Area, 3. Circularity Index, 4. Object Count,
    5. Major Axis Length, 6. Minor Axis Length.
    """
    contours, _ = cv2.findContours(lesion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    num_objects = float(len(contours))
    total_area = float(np.count_nonzero(lesion_mask))
    total_perimeter = 0.0

    major_axis = 0.0
    minor_axis = 0.0

    for c in contours:
        total_perimeter += cv2.arcLength(c, True)
        if len(c) >= 5:  # Fit ellipse requires at least 5 points
            try:
                ellipse = cv2.fitEllipse(c)
                axes = ellipse[1]
                major_axis = max(major_axis, float(max(axes)))
                minor_axis = max(minor_axis, float(min(axes)))
            except Exception:
                pass

    if total_perimeter > 0:
        circularity = float((4.0 * np.pi * total_area) / (total_perimeter ** 2))
    else:
        circularity = 0.0

    features = [
        total_perimeter,
        total_area,
        circularity,
        num_objects,
        major_axis,
        minor_axis
    ]

    names = [
        "shape_perimeter",
        "shape_area",
        "shape_circularity",
        "shape_object_count",
        "shape_major_axis",
        "shape_minor_axis"
    ]

    return np.array(features, dtype=np.float32), names


def extract_all_features(image_bgr: np.ndarray, leaf_mask: np.ndarray, lesion_mask: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """
    Extracts the complete set of 68 multi-domain features (48 color + 14 texture + 6 shape).
    Returns (feature_vector of shape (68,), feature_names of length 68).
    """
    color_feats, color_names = extract_color_features(image_bgr, leaf_mask)
    texture_feats, texture_names = extract_texture_features(image_bgr, leaf_mask)
    shape_feats, shape_names = extract_shape_features(lesion_mask, leaf_mask)

    all_feats = np.concatenate([color_feats, texture_feats, shape_feats])
    all_names = color_names + texture_names + shape_names

    # Clean any accidental NaNs or Infs
    all_feats = np.nan_to_num(all_feats, nan=0.0, posinf=1e5, neginf=-1e5)

    return all_feats, all_names
