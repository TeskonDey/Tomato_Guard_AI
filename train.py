"""
Model Training and Optimization Script
Runs Genetic Algorithm optimization for feature selection and hyperparameter tuning,
evaluates performance against unoptimized baselines, and saves trained models.
"""

import os
import sys
import argparse
import json
import joblib
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.dataset import (
    CLASS_NAMES,
    build_synthetic_training_dataset,
    extract_dataset_features,
    populate_sample_library
)
from src.optimizer import GeneticAlgorithmOptimizer, benchmark_models


def main():
    parser = argparse.ArgumentParser(description="Train and optimize TomatoGuard AI disease classifier.")
    parser.add_argument("--data_dir", type=str, default="data/dataset", help="Path to image dataset folder.")
    parser.add_argument("--generations", type=int, default=12, help="Number of GA generations.")
    parser.add_argument("--pop_size", type=int, default=25, help="GA population size.")
    parser.add_argument("--synth_samples", type=int, default=35, help="Number of synthetic samples per class if no dataset.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    cache_file = str(project_root / "data" / "features_cache.npz")

    # Step 1: Ensure dataset availability
    dataset_path = project_root / args.data_dir
    if not dataset_path.exists() or len(list(dataset_path.glob("*/*"))) < 10:
        print(f"No custom dataset detected at {dataset_path}. Generating initial training set...")
        build_synthetic_training_dataset(output_dir=str(dataset_path), samples_per_class=args.synth_samples)

    # Ensure reference test samples exist
    populate_sample_library(output_dir=str(project_root / "data" / "samples"), samples_per_class=4)

    # Step 2: Feature Extraction
    print("\n--- Phase 1: Handcrafted Feature Extraction (68 Features) ---")
    X, y, feature_names, class_names = extract_dataset_features(
        dataset_dir=str(dataset_path),
        cache_path=cache_file
    )
    print(f"Dataset matrix: {X.shape[0]} samples, {X.shape[1]} features, {len(class_names)} classes.")

    # Step 3: Train / Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Split: {len(X_train)} training samples, {len(X_test)} testing samples.")

    # Step 4: Evolutionary Genetic Algorithm Optimization
    print("\n--- Phase 2: Genetic Algorithm Optimization ---")
    ga = GeneticAlgorithmOptimizer(
        population_size=args.pop_size,
        generations=args.generations,
        crossover_rate=0.8,
        mutation_rate=0.06,
        cv_folds=3,
        random_state=42
    )

    ga_results = ga.optimize(X_train, y_train, feature_names)

    selected_indices = ga_results["selected_feature_indices"]
    selected_names = ga_results["selected_feature_names"]
    best_params = ga_results["best_hyperparameters"]

    print(f"\nOptimization Completed!")
    print(f"Selected Features: {len(selected_indices)} of {len(feature_names)}")
    print(f"Best RF Parameters: {best_params}")

    # Step 5: Benchmark Baseline vs. GA-Optimized Models
    print("\n--- Phase 3: Benchmarking Classifiers ---")
    bench = benchmark_models(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        selected_feature_indices=selected_indices,
        tuned_rf_params=best_params,
        class_names=class_names
    )

    print("\n=======================================================")
    print("                 BENCHMARK RESULTS                     ")
    print("=======================================================")
    print(f"Random Forest (Baseline - all 68 features):    {bench['RandomForest_Baseline_Acc']:.2f}%")
    print(f"Random Forest (GA-Optimized - {len(selected_indices)} features):  {bench['RandomForest_Optimized_Acc']:.2f}%")
    print(f"Support Vector Machine (Baseline):             {bench['SVM_Baseline_Acc']:.2f}%")
    print(f"Support Vector Machine (GA-Optimized):         {bench['SVM_Optimized_Acc']:.2f}%")
    print("=======================================================")

    # Step 6: Save Model Artifacts
    package = {
        "rf_model": bench["Optimized_RF_Model"],
        "svm_model": bench["Optimized_SVM_Model"],
        "scaler": bench["Scaler"],
        "selected_indices": selected_indices,
        "selected_names": selected_names,
        "all_feature_names": feature_names,
        "class_names": class_names,
        "best_params": best_params,
        "benchmarks": {
            "rf_baseline": bench["RandomForest_Baseline_Acc"],
            "rf_optimized": bench["RandomForest_Optimized_Acc"],
            "svm_baseline": bench["SVM_Baseline_Acc"],
            "svm_optimized": bench["SVM_Optimized_Acc"]
        },
        "confusion_matrix": bench["Optimized_RF_Confusion_Matrix"],
        "ga_history": ga_results["history"]
    }

    model_path = models_dir / "tomato_rf_model.joblib"
    joblib.dump(package, model_path)
    print(f"\nSaved trained model package to: {model_path}")

    report_path = models_dir / "benchmark_report.json"
    with open(report_path, "w") as f:
        json.dump({
            "benchmarks": package["benchmarks"],
            "selected_features_count": len(selected_indices),
            "selected_feature_names": selected_names,
            "best_hyperparameters": best_params,
            "confusion_matrix": package["confusion_matrix"],
            "class_names": class_names,
            "ga_history": ga_results["history"]
        }, f, indent=2)
    print(f"Saved benchmark report to: {report_path}")


if __name__ == "__main__":
    main()
