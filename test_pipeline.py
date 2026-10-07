import os
import sys
import cv2
from pathlib import Path

# Add project root
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from app import load_trained_model, run_pipeline
from src.dataset import CLASS_NAMES

print("Testing TomatoGuard Diagnostic Pipeline...")
model_pkg = load_trained_model()
assert model_pkg is not None, "Error: Model package failed to load!"
print(f"Model successfully loaded. Active features: {len(model_pkg['selected_indices'])} / 68")

samples_dir = project_root / "data" / "samples"
for class_name in CLASS_NAMES:
    cat_dir = samples_dir / class_name
    sample_files = list(cat_dir.glob("*.png"))
    assert len(sample_files) > 0, f"No samples found for {class_name}"
    
    img = cv2.imread(str(sample_files[0]))
    assert img is not None, f"Failed to read image {sample_files[0]}"
    
    res = run_pipeline(img, model_pkg)
    sev = res["severity_info"]
    pred = res["predicted_class"]
    conf = res["probabilities"].get(pred, 0.0) * 100
    
    print(f"[{class_name}] -> Pred: {pred} ({conf:.1f}%) | Severity: {sev['severity_pct']}% [{sev['stage']}]")

print("\nAll 5 classes tested successfully!")
