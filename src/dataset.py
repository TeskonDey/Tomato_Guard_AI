"""
Dataset Manager, Image Loader, and Synthetic Leaf Generator
Generates realistic tomato leaf samples for demonstration and loads custom image datasets.
"""

import os
import random
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from .segmentation import extract_leaf_mask, segment_lesions
from .features import extract_all_features

CLASS_NAMES = [
    "Alternaria alternata",
    "Alternaria solani",
    "Botrytis cinerea",
    "Fusarium oxysporum",
    "Healthy"
]


def draw_tomato_leaf_shape(img_size: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """
    Renders an organic tomato leaflet silhouette with serrated margins and natural green shading.
    """
    canvas = np.full((img_size, img_size, 3), 195, dtype=np.uint8)  # neutral gray background
    center_x = img_size // 2
    center_y = img_size // 2

    # Generate organic leaf polygon with serrated leaflet lobes
    angles = np.linspace(0, 2 * np.pi, 180, endpoint=False)
    points = []

    for a in angles:
        # Base lanceolate/ovate shape
        r = 130 * (1.0 - 0.4 * np.sin(a)) * (0.8 + 0.2 * np.cos(2 * a))
        # Add natural serrations along edges
        serration = 6.0 * np.sin(16 * a)
        r_total = max(20.0, r + serration)

        x = int(center_x + r_total * np.cos(a) * 0.75)
        y = int(center_y + r_total * np.sin(a) * 1.15)
        points.append([x, y])

    leaf_pts = np.array(points, dtype=np.int32)
    leaf_mask = np.zeros((img_size, img_size), dtype=np.uint8)
    cv2.fillPoly(leaf_mask, [leaf_pts], 255)

    # Base green leaf color with slight gradient
    base_color = np.zeros_like(canvas)
    for y in range(img_size):
        v_factor = 0.85 + 0.3 * (y / img_size)
        base_color[y, :] = (int(35 * v_factor), int(140 * v_factor), int(45 * v_factor))

    # Apply base color to leaf
    canvas[leaf_mask > 0] = base_color[leaf_mask > 0]

    # Draw natural leaf veins
    cv2.line(canvas, (center_x, center_y - 120), (center_x, center_y + 130), (30, 165, 40), 3)
    for offset in range(-90, 110, 28):
        y_pos = center_y + offset
        cv2.line(canvas, (center_x, y_pos), (center_x - 70, y_pos - 35), (32, 155, 42), 2)
        cv2.line(canvas, (center_x, y_pos), (center_x + 70, y_pos - 35), (32, 155, 42), 2)

    return canvas, leaf_mask


def generate_synthetic_sample(disease_type: str, img_size: int = 400, seed: int = None) -> np.ndarray:
    """
    Synthesizes a realistic tomato leaf image exhibiting disease-specific visual symptoms.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    canvas, leaf_mask = draw_tomato_leaf_shape(img_size)
    cx, cy = img_size // 2, img_size // 2

    if disease_type == "Healthy":
        # Add slight natural color variation
        noise = np.random.randint(-8, 8, canvas.shape, dtype=np.int16)
        canvas_noisy = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        canvas[leaf_mask > 0] = canvas_noisy[leaf_mask > 0]
        return canvas

    elif disease_type == "Alternaria alternata":
        # Symptoms: Small, numerous circular brown spots with prominent yellow chlorotic halos
        num_spots = random.randint(8, 16)
        for _ in range(num_spots):
            rx = int(np.clip(cx + np.random.normal(0, 45), cx - 75, cx + 75))
            ry = int(np.clip(cy + np.random.normal(0, 70), cy - 100, cy + 100))
            if leaf_mask[ry, rx] > 0:
                rad = random.randint(7, 14)
                # Outer yellow halo
                cv2.circle(canvas, (rx, ry), rad + 4, (40, 190, 210), -1)
                # Inner dark necrotic spot
                cv2.circle(canvas, (rx, ry), rad, (22, 45, 90), -1)
                # Central point
                cv2.circle(canvas, (rx, ry), max(2, rad - 4), (15, 30, 65), -1)

    elif disease_type == "Alternaria solani":
        # Symptoms: Larger dark brown lesions with classic concentric rings (target board) & chlorotic yellowing
        num_lesions = random.randint(3, 6)
        for _ in range(num_lesions):
            rx = int(np.clip(cx + np.random.normal(0, 50), cx - 80, cx + 80))
            ry = int(np.clip(cy + np.random.normal(0, 65), cy - 95, cy + 95))
            if leaf_mask[ry, rx] > 0:
                base_r = random.randint(18, 32)
                # Wide diffuse yellow border
                cv2.circle(canvas, (rx, ry), base_r + 8, (45, 185, 215), -1)
                # Concentric ring structure
                cv2.circle(canvas, (rx, ry), base_r, (25, 50, 100), -1)
                cv2.circle(canvas, (rx, ry), int(base_r * 0.75), (35, 70, 130), -1)
                cv2.circle(canvas, (rx, ry), int(base_r * 0.50), (20, 40, 85), -1)
                cv2.circle(canvas, (rx, ry), int(base_r * 0.25), (10, 25, 55), -1)

    elif disease_type == "Botrytis cinerea":
        # Symptoms: Water-soaked grayish/tan fuzzy patches with irregular, soft margins
        num_patches = random.randint(2, 4)
        for _ in range(num_patches):
            rx = int(np.clip(cx + np.random.normal(0, 45), cx - 70, cx + 70))
            ry = int(np.clip(cy + np.random.normal(0, 55), cy - 80, cy + 80))
            if leaf_mask[ry, rx] > 0:
                patch_w = random.randint(30, 55)
                patch_h = random.randint(25, 45)
                angle = random.randint(0, 180)
                # Water soaked translucent margin
                cv2.ellipse(canvas, (rx, ry), (patch_w + 6, patch_h + 6), angle, 0, 360, (70, 130, 110), -1)
                # Velvety gray/tan fungal core
                cv2.ellipse(canvas, (rx, ry), (patch_w, patch_h), angle, 0, 360, (110, 125, 135), -1)
                # Inner gray fuzz
                cv2.ellipse(canvas, (rx, ry), (patch_w - 10, patch_h - 8), angle, 0, 360, (140, 145, 150), -1)

    elif disease_type == "Fusarium oxysporum":
        # Symptoms: One-sided chlorosis (yellowing across half the lamina), vein yellowing & marginal drying
        for y in range(img_size):
            for x in range(img_size):
                if leaf_mask[y, x] > 0 and x < cx + 15:  # Left hemisphere yellowing
                    dist = (cx + 15 - x) / 100.0
                    factor = np.clip(dist, 0.0, 1.0)
                    orig = canvas[y, x].astype(float)
                    # Blend towards dry yellow/chlorotic
                    yellow_val = np.array([35.0, 195.0, 220.0])
                    canvas[y, x] = np.clip((1 - factor) * orig + factor * yellow_val, 0, 255).astype(np.uint8)

        # Marginal brown wilt patches on the affected side
        for _ in range(3):
            wx = int(cx - random.randint(45, 80))
            wy = int(cy + random.randint(-60, 60))
            if leaf_mask[wy, wx] > 0:
                cv2.ellipse(canvas, (wx, wy), (18, 30), 20, 0, 360, (25, 60, 115), -1)

    # Add realistic leaf texture grain
    noise = np.random.randint(-6, 6, canvas.shape, dtype=np.int16)
    canvas_noisy = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    canvas[leaf_mask > 0] = canvas_noisy[leaf_mask > 0]

    return canvas


def populate_sample_library(output_dir: str = "data/samples", samples_per_class: int = 5):
    """
    Generates reference demonstration samples for all 5 disease categories.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for class_name in CLASS_NAMES:
        class_dir = out_path / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        for i in range(samples_per_class):
            img = generate_synthetic_sample(class_name, seed=100 + i * 17)
            filename = class_dir / f"sample_{i + 1}.png"
            cv2.imwrite(str(filename), img)

    print(f"Generated {samples_per_class} reference samples per class in {output_dir}")


def build_synthetic_training_dataset(output_dir: str = "data/synthetic", samples_per_class: int = 40):
    """
    Generates a balanced dataset of synthetic images for initial training and model validation.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for class_name in CLASS_NAMES:
        class_dir = out_path / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for i in range(samples_per_class):
            img = generate_synthetic_sample(class_name, seed=2000 + i * 31 + len(class_name))
            cv2.imwrite(str(class_dir / f"img_{i:04d}.png"), img)

    print(f"Created synthetic training dataset with {samples_per_class * len(CLASS_NAMES)} images in {output_dir}")


def extract_dataset_features(dataset_dir: str, cache_path: str = None) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """
    Scans dataset directory organized as: dataset_dir/<class_name>/<image_files>.
    Extracts all 68 features per image. Caches to .npz file if cache_path is provided.
    """
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}...")
        data = np.load(cache_path, allow_pickle=True)
        return data["X"], data["y"], data["feature_names"].tolist(), data["class_names"].tolist()

    data_dir = Path(dataset_dir)
    class_folders = [d for d in data_dir.iterdir() if d.is_dir() and d.name in CLASS_NAMES]
    if not class_folders:
        class_folders = [d for d in data_dir.iterdir() if d.is_dir()]

    class_names = sorted([d.name for d in class_folders])
    label_map = {name: idx for idx, name in enumerate(class_names)}

    all_images = []
    all_labels = []

    for folder in class_folders:
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.JPG", "*.PNG"):
            for f in folder.glob(ext):
                all_images.append((f, label_map[folder.name]))

    if not all_images:
        raise ValueError(f"No valid images found in {dataset_dir}")

    print(f"Extracting 68 features from {len(all_images)} images across {len(class_names)} classes...")

    X_list = []
    y_list = []
    feature_names = None

    for img_path, label in tqdm(all_images):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        leaf_mask = extract_leaf_mask(img_bgr)
        lesion_mask = segment_lesions(img_bgr, leaf_mask)
        feats, names = extract_all_features(img_bgr, leaf_mask, lesion_mask)

        if feature_names is None:
            feature_names = names

        X_list.append(feats)
        y_list.append(label)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_path,
            X=X,
            y=y,
            feature_names=np.array(feature_names),
            class_names=np.array(class_names)
        )
        print(f"Saved feature cache to {cache_path}")

    return X, y, feature_names, class_names
