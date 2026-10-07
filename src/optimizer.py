"""
Genetic Algorithm Optimizer for Multi-Domain Feature Selection and Model Tuning
Selects the most discriminative features (out of 68) and optimizes Random Forest
and Support Vector Machine classifiers.
"""

import json
import random
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler


class GeneticAlgorithmOptimizer:
    """
    Evolutionary optimizer that performs simultaneous feature selection
    and classifier hyperparameter tuning.
    """

    def __init__(
        self,
        population_size: int = 30,
        generations: int = 20,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.05,
        tournament_size: int = 3,
        elite_count: int = 2,
        cv_folds: int = 4,
        random_state: int = 42
    ):
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elite_count = elite_count
        self.cv_folds = cv_folds
        self.random_state = random_state
        random.seed(random_state)
        np.random.seed(random_state)

        # Hyperparameter search space for Random Forest
        self.param_grid = {
            "n_estimators": [50, 80, 100, 150, 200],
            "max_depth": [5, 8, 12, 16, None],
            "min_samples_split": [2, 4, 6, 8],
            "min_samples_leaf": [1, 2, 4, 6]
        }

    def _random_chromosome(self, num_features: int) -> dict:
        """
        Creates an individual with a random binary feature mask and random hyperparameter choices.
        """
        # Ensure between 15% and 50% of features are initially active
        feature_mask = np.random.choice([0, 1], size=num_features, p=[0.7, 0.3])
        if np.sum(feature_mask) < 6:
            # Force at least 6 features
            idx = np.random.choice(num_features, size=6, replace=False)
            feature_mask[idx] = 1

        params = {
            "n_estimators": random.choice(self.param_grid["n_estimators"]),
            "max_depth": random.choice(self.param_grid["max_depth"]),
            "min_samples_split": random.choice(self.param_grid["min_samples_split"]),
            "min_samples_leaf": random.choice(self.param_grid["min_samples_leaf"])
        }
        return {"features": feature_mask, "params": params, "fitness": -1.0}

    def _calculate_fitness(self, individual: dict, X: np.ndarray, y: np.ndarray) -> float:
        """
        Computes CV accuracy with a light sparsity reward to encourage efficient feature sets.
        """
        mask = individual["features"]
        selected_indices = np.where(mask == 1)[0]

        if len(selected_indices) < 4:
            return 0.0

        X_sub = X[:, selected_indices]
        rf = RandomForestClassifier(
            n_estimators=individual["params"]["n_estimators"],
            max_depth=individual["params"]["max_depth"],
            min_samples_split=individual["params"]["min_samples_split"],
            min_samples_leaf=individual["params"]["min_samples_leaf"],
            random_state=self.random_state,
            n_jobs=-1
        )

        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        scores = cross_val_score(rf, X_sub, y, cv=cv, scoring='accuracy', n_jobs=-1)
        mean_acc = float(np.mean(scores))

        # Reward accuracy heavily; slight bonus for feature parsimony
        sparsity_bonus = 0.02 * (1.0 - (len(selected_indices) / len(mask)))
        return round(mean_acc + sparsity_bonus, 4)

    def _tournament_selection(self, population: list[dict]) -> dict:
        candidates = random.sample(population, self.tournament_size)
        return max(candidates, key=lambda ind: ind["fitness"])

    def _crossover(self, parent1: dict, parent2: dict) -> tuple[dict, dict]:
        """
        Uniform crossover for features and parameter swap.
        """
        if random.random() > self.crossover_rate:
            return {
                "features": parent1["features"].copy(),
                "params": dict(parent1["params"]),
                "fitness": -1.0
            }, {
                "features": parent2["features"].copy(),
                "params": dict(parent2["params"]),
                "fitness": -1.0
            }

        # Feature mask uniform crossover
        num_feats = len(parent1["features"])
        crossover_points = np.random.choice([0, 1], size=num_feats)
        child1_feats = np.where(crossover_points == 1, parent1["features"], parent2["features"])
        child2_feats = np.where(crossover_points == 1, parent2["features"], parent1["features"])

        # Hyperparameter swap
        child1_params = {}
        child2_params = {}
        for k in parent1["params"]:
            if random.random() < 0.5:
                child1_params[k] = parent1["params"][k]
                child2_params[k] = parent2["params"][k]
            else:
                child1_params[k] = parent2["params"][k]
                child2_params[k] = parent1["params"][k]

        return {"features": child1_feats, "params": child1_params, "fitness": -1.0}, \
               {"features": child2_feats, "params": child2_params, "fitness": -1.0}

    def _mutate(self, individual: dict, num_features: int):
        """
        Bit-flip mutation for features and random perturbation for hyperparameters.
        """
        for i in range(num_features):
            if random.random() < self.mutation_rate:
                individual["features"][i] = 1 - individual["features"][i]

        # Ensure at least 4 features are active
        if np.sum(individual["features"]) < 4:
            idx = np.random.choice(num_features, size=4, replace=False)
            individual["features"][idx] = 1

        # Parameter mutation
        for param, values in self.param_grid.items():
            if random.random() < self.mutation_rate * 2:
                individual["params"][param] = random.choice(values)

    def optimize(self, X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> dict:
        """
        Executes the genetic algorithm search.
        """
        num_features = X.shape[1]
        population = [self._random_chromosome(num_features) for _ in range(self.population_size)]

        history = []
        best_overall = None

        print(f"Starting Genetic Algorithm: {self.generations} generations, population {self.population_size}...")

        for gen in range(self.generations):
            for ind in population:
                if ind["fitness"] < 0:
                    ind["fitness"] = self._calculate_fitness(ind, X, y)

            population.sort(key=lambda x: x["fitness"], reverse=True)
            current_best = population[0]

            if best_overall is None or current_best["fitness"] > best_overall["fitness"]:
                best_overall = {
                    "features": current_best["features"].copy(),
                    "params": dict(current_best["params"]),
                    "fitness": current_best["fitness"]
                }

            history.append({
                "generation": gen + 1,
                "best_fitness": round(current_best["fitness"], 4),
                "num_features": int(np.sum(current_best["features"])),
                "best_params": current_best["params"]
            })

            print(f"Gen {gen + 1}/{self.generations}: Best Score = {current_best['fitness']:.4f} "
                  f"({np.sum(current_best['features'])} features)")

            # Breed next generation with elitism
            new_population = [
                {
                    "features": ind["features"].copy(),
                    "params": dict(ind["params"]),
                    "fitness": ind["fitness"]
                }
                for ind in population[:self.elite_count]
            ]

            while len(new_population) < self.population_size:
                p1 = self._tournament_selection(population)
                p2 = self._tournament_selection(population)
                c1, c2 = self._crossover(p1, p2)
                self._mutate(c1, num_features)
                self._mutate(c2, num_features)
                new_population.append(c1)
                if len(new_population) < self.population_size:
                    new_population.append(c2)

            population = new_population

        selected_indices = np.where(best_overall["features"] == 1)[0].tolist()
        selected_names = [feature_names[i] for i in selected_indices]

        return {
            "best_chromosome": best_overall,
            "selected_feature_indices": selected_indices,
            "selected_feature_names": selected_names,
            "best_hyperparameters": best_overall["params"],
            "best_score": best_overall["fitness"],
            "history": history
        }


def benchmark_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    selected_feature_indices: list[int],
    tuned_rf_params: dict,
    class_names: list[str]
) -> dict:
    """
    Evaluates baseline vs GA-optimized models for both Random Forest and SVM.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 1. Baseline Random Forest (all 68 features, default params)
    baseline_rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    baseline_rf.fit(X_train, y_train)
    rf_base_pred = baseline_rf.predict(X_test)
    rf_base_acc = float(accuracy_score(y_test, rf_base_pred))

    # 2. GA-Optimized Random Forest (selected features + tuned params)
    X_train_sub = X_train[:, selected_feature_indices]
    X_test_sub = X_test[:, selected_feature_indices]

    optimized_rf = RandomForestClassifier(
        n_estimators=tuned_rf_params["n_estimators"],
        max_depth=tuned_rf_params["max_depth"],
        min_samples_split=tuned_rf_params["min_samples_split"],
        min_samples_leaf=tuned_rf_params["min_samples_leaf"],
        random_state=42,
        n_jobs=-1
    )
    optimized_rf.fit(X_train_sub, y_train)
    rf_opt_pred = optimized_rf.predict(X_test_sub)
    rf_opt_acc = float(accuracy_score(y_test, rf_opt_pred))

    # 3. Baseline SVM (all features, default)
    baseline_svm = SVC(probability=True, random_state=42)
    baseline_svm.fit(X_train_scaled, y_train)
    svm_base_pred = baseline_svm.predict(X_test_scaled)
    svm_base_acc = float(accuracy_score(y_test, svm_base_pred))

    # 4. GA-Optimized SVM (using selected features + tuned RBF)
    scaler_sub = StandardScaler()
    X_train_sub_scaled = scaler_sub.fit_transform(X_train_sub)
    X_test_sub_scaled = scaler_sub.transform(X_test_sub)

    optimized_svm = SVC(C=10.0, gamma='scale', probability=True, random_state=42)
    optimized_svm.fit(X_train_sub_scaled, y_train)
    svm_opt_pred = optimized_svm.predict(X_test_sub_scaled)
    svm_opt_acc = float(accuracy_score(y_test, svm_opt_pred))

    cm_rf_opt = confusion_matrix(y_test, rf_opt_pred).tolist()

    return {
        "RandomForest_Baseline_Acc": round(rf_base_acc * 100, 2),
        "RandomForest_Optimized_Acc": round(rf_opt_acc * 100, 2),
        "SVM_Baseline_Acc": round(svm_base_acc * 100, 2),
        "SVM_Optimized_Acc": round(svm_opt_acc * 100, 2),
        "Optimized_RF_Confusion_Matrix": cm_rf_opt,
        "Class_Names": class_names,
        "Optimized_RF_Model": optimized_rf,
        "Optimized_SVM_Model": optimized_svm,
        "Scaler": scaler_sub
    }
