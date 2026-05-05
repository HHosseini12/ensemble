"""
Ensemble Neural Network with Uncertainty Quantification for Band Gap Prediction
================================================================================
Nested cross-validation (5-outer × 5-inner) using deep ensembles and isotonic
recalibration via the uncertainty-toolbox library.


Requirements
------------
    pip install tensorflow tensorflow-probability uncertainty-toolbox scikit-learn \
                matplotlib numpy

Data
----
    ECFPs_MP_466.npy          – ECFP fingerprints (N × D)
    Bandgaps_MP_466.npy       – Band-gap labels  (N,)
    infinity_feat_index.npy   – Indices of infinite-valued features to drop

Usage
-----
    python ensemble_bandgap_uq.py
"""

from __future__ import annotations

import logging
import os
import random
import warnings
from pathlib import Path
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Silence TensorFlow / Keras / ABSL noise BEFORE any TF import
# ---------------------------------------------------------------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"          # suppress C++ INFO/WARNING/ERROR logs
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "3"

warnings.filterwarnings("ignore", category=UserWarning, module="keras")
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import uncertainty_toolbox as uct
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
import tensorflow as tf
import keras
from keras import layers, regularizers
import matplotlib.gridspec as gridspec
# Suppress tf.function retracing warnings at the Python level too
tf.get_logger().setLevel("ERROR")

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(".")                 # change if data lives elsewhere
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Architecture / training defaults
# ---------------------------------------------------------------------------
MODEL_CFG = dict(
    num_layers=4,
    units=1024,
    activation="relu",
    l2_reg=1e-6,
)

TRAIN_CFG = dict(
    epochs=4000,
    batch_size=32,
    patience=200,
)

# Best hyper-parameters found during inner-CV tuning (dropout, lr) per fold
OUTER_INNER_BEST_HPS: List[List[dict]] = [
    [  # Outer fold 1
        {"dropout_rate": 0.10, "learning_rate": 1.0000e-04},
        {"dropout_rate": 0.00, "learning_rate": 2.788878324170188e-05},
        {"dropout_rate": 0.15, "learning_rate": 6.932442978408296e-05},
        {"dropout_rate": 0.20, "learning_rate": 3.3409692406654675e-05},
        {"dropout_rate": 0.20, "learning_rate": 7.33069463730847e-05},
    ],
    [  # Outer fold 2
        {"dropout_rate": 0.20, "learning_rate": 9.385036825787705e-05},
        {"dropout_rate": 0.15, "learning_rate": 3.132025708097482e-05},
        {"dropout_rate": 0.20, "learning_rate": 1.0000e-04},
        {"dropout_rate": 0.00, "learning_rate": 8.365417304252826e-05},
        {"dropout_rate": 0.20, "learning_rate": 1.0000e-04},
    ],
    [  # Outer fold 3
        {"dropout_rate": 0.20, "learning_rate": 1.0000e-04},
        {"dropout_rate": 0.05, "learning_rate": 1.0000e-04},
        {"dropout_rate": 0.00, "learning_rate": 3.415762251337944e-05},
        {"dropout_rate": 0.20, "learning_rate": 3.834962117654816e-05},
        {"dropout_rate": 0.15, "learning_rate": 5.980278300315725e-05},
    ],
    [  # Outer fold 4
        {"dropout_rate": 0.20, "learning_rate": 5.228212427140258e-05},
        {"dropout_rate": 0.05, "learning_rate": 9.650040717278601e-05},
        {"dropout_rate": 0.20, "learning_rate": 1.0000e-04},
        {"dropout_rate": 0.20, "learning_rate": 1.0000e-04},
        {"dropout_rate": 0.00, "learning_rate": 6.003173490781914e-05},
    ],
    [  # Outer fold 5
        {"dropout_rate": 0.05, "learning_rate": 5.105613758334641e-05},
        {"dropout_rate": 0.15, "learning_rate": 1.0000e-04},
        {"dropout_rate": 0.20, "learning_rate": 1.0000e-04},
        {"dropout_rate": 0.10, "learning_rate": 3.537484988977307e-05},
        {"dropout_rate": 0.15, "learning_rate": 2.7485514228687672e-05},
    ],
]


# ===========================================================================
# Model helpers
# ===========================================================================

def build_model(hp: dict, input_shape: tuple, seed: int = SEED) -> keras.Model:
    """Build and compile a regularised MLP with the given hyper-parameters.

    Parameters
    ----------
    hp:
        Dictionary with keys ``dropout_rate`` and ``learning_rate``.
    input_shape:
        Shape of a single input sample, e.g. ``(2048,)`` for ECFP-2048.
    seed:
        Random seed for weight initialisation.

    Returns
    -------
    keras.Model
        Compiled Keras model ready for training.
    """
    tf.keras.backend.clear_session()
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    dropout_rate: float = hp.get("dropout_rate", 0.1)
    lr: float = hp.get("learning_rate", 1e-4)
    regularizer = regularizers.l2(MODEL_CFG["l2_reg"])

    # Use an explicit Input layer — avoids the Keras 3 input_shape deprecation warning
    model = keras.Sequential([keras.Input(shape=input_shape)])
    for _ in range(MODEL_CFG["num_layers"]):
        model.add(layers.Dense(
            units=MODEL_CFG["units"],
            activation=MODEL_CFG["activation"],
            kernel_regularizer=regularizer,
        ))
        model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(1, activation="linear"))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanAbsoluteError()],
    )
    return model


class DeepEnsemble:
    """Wraps a list of trained Keras models for joint prediction.

    Parameters
    ----------
    models:
        List of trained ``keras.Model`` instances.
    """

    def __init__(self, models: List[keras.Model]) -> None:
        self.models = models

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return per-member predictions stacked column-wise.

        Parameters
        ----------
        X:
            Input array of shape ``(n_samples, n_features)``.

        Returns
        -------
        np.ndarray
            Shape ``(n_samples, n_members)``.
        """
        preds = [m.predict(X, verbose=0).flatten() for m in self.models]
        return np.column_stack(preds)


# ===========================================================================
# Data loading
# ===========================================================================

def load_data(data_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load ECFP fingerprints and band-gap labels, filtering invalid entries.

    Parameters
    ----------
    data_dir:
        Directory containing the ``.npy`` data files.

    Returns
    -------
    X, y:
        Filtered feature matrix and target vector.
    """
    ECFP_all = np.load(data_dir / "ECFPs_MP_466.npy", allow_pickle=True)
    BG_all = np.load(data_dir / "Bandgaps_MP_466.npy", allow_pickle=True)
    inf_index = np.load(data_dir / "infinity_feat_index.npy", allow_pickle=True)

    # Remove samples with infinite features
    ECFP_w = [e for i, e in enumerate(ECFP_all) if i not in inf_index]
    BG_w = [b for i, b in enumerate(BG_all) if i not in inf_index]

    # Remove samples with missing labels
    valid_mask = [b != "None" for b in BG_w]
    X = np.array([e for e, v in zip(ECFP_w, valid_mask) if v], dtype=float)
    y = np.array([float(b) for b, v in zip(BG_w, valid_mask) if v], dtype=float)

    print(f"Loaded data — X: {X.shape}, y: {y.shape}")
    return X, y


# ===========================================================================
# Nested cross-validation
# ===========================================================================

def run_nested_cv(
    X: np.ndarray,
    y: np.ndarray,
    outer_k: int = 5,
    inner_k: int = 5,
) -> dict:
    """Run nested k-fold cross-validation with ensemble training and isotonic
    recalibration.

    Parameters
    ----------
    X, y:
        Feature matrix and target vector.
    outer_k, inner_k:
        Number of outer / inner folds.

    Returns
    -------
    dict
        Aggregated predictions, uncertainties, and per-fold metrics.
    """
    outer_cv = KFold(n_splits=outer_k, shuffle=True, random_state=7)

    all_preds, all_trues, all_stds, all_stds_uncal = [], [], [], []
    metrics = dict(mae=[], rmse=[], r2=[],
                   mace_before=[], rmsce_before=[], ma_before=[],
                   mace_after=[], rmsce_after=[], ma_after=[], 
                   coverage90_before=[], coverage90_after=[])

    for fold_idx, (outer_train_idx, outer_test_idx) in enumerate(
        outer_cv.split(X, y), start=1
    ):
        print(f"\n{'='*60}")
        print(f"  Outer Fold {fold_idx}/{outer_k}")
        print(f"{'='*60}")

        X_train, X_test = X[outer_train_idx], X[outer_test_idx]
        y_train, y_test = y[outer_train_idx], y[outer_test_idx]

        inner_cv = KFold(n_splits=inner_k, shuffle=True, random_state=fold_idx)
        inner_splits = list(inner_cv.split(X_train, y_train))
        hp_list = OUTER_INNER_BEST_HPS[fold_idx - 1]

        # ---- Train inner ensemble ----------------------------------------
        inner_models: List[keras.Model] = []
        for inner_idx, (hp, (i_train, i_val)) in enumerate(
            zip(hp_list, inner_splits), start=1
        ):
            print(f"  Inner fold {inner_idx}/{inner_k} — "
                  f"lr={hp['learning_rate']:.2e}, dropout={hp['dropout_rate']}")
            model = build_model(hp, input_shape=X.shape[1:], seed=SEED + inner_idx)
            model.fit(
                X_train[i_train], y_train[i_train],
                validation_data=(X_train[i_val], y_train[i_val]),
                epochs=TRAIN_CFG["epochs"],
                batch_size=TRAIN_CFG["batch_size"],
                callbacks=[
                    keras.callbacks.EarlyStopping(
                        monitor="val_loss",
                        patience=TRAIN_CFG["patience"],
                        restore_best_weights=True,
                        verbose=0,
                    )
                ],
                verbose=0,
            )
            inner_models.append(model)

        # ---- Collect validation predictions for recalibration -------------
        val_preds_all, val_stds_all, val_trues_all = [], [], []
        ensemble = DeepEnsemble(inner_models)
        for i_train, i_val in inner_splits:
            preds_mat = ensemble.predict(X_train[i_val])
            val_preds_all.extend(preds_mat.mean(axis=1))
            val_stds_all.extend(preds_mat.std(axis=1))
            val_trues_all.extend(y_train[i_val])

        val_preds = np.array(val_preds_all)
        val_stds = np.clip(np.array(val_stds_all), 1e-6, None)
        val_trues = np.array(val_trues_all)

        # ---- Isotonic recalibration ---------------------------------------
        exp_props, obs_props = uct.get_proportion_lists_vectorized(
            val_preds, val_stds, val_trues, prop_type="interval"
        )
        iso_model = uct.recalibration.iso_recal(exp_props, obs_props)

        # ---- Outer test predictions ---------------------------------------
        test_mat = ensemble.predict(X_test)
        mean_preds = test_mat.mean(axis=1)
        std_uncal = np.clip(test_mat.std(axis=1), 1e-6, None)
        std_recal = iso_model.transform(std_uncal)

        # ---- Accuracy metrics --------------------------------------------
        fold_mae = mean_absolute_error(y_test, mean_preds)
        fold_rmse = np.sqrt(mean_squared_error(y_test, mean_preds))
        fold_r2 = r2_score(y_test, mean_preds)
        metrics["mae"].append(fold_mae)
        metrics["rmse"].append(fold_rmse)
        metrics["r2"].append(fold_r2)

        # ---- Calibration metrics -----------------------------------------
        for tag, std in [("before", std_uncal), ("after", std_recal)]:
            metrics[f"mace_{tag}"].append(
                uct.metrics.mean_absolute_calibration_error(mean_preds, std, y_test))
            metrics[f"rmsce_{tag}"].append(
                uct.metrics.root_mean_squared_calibration_error(mean_preds, std, y_test))
            metrics[f"ma_{tag}"].append(
                uct.metrics.miscalibration_area(mean_preds, std, y_test))
            metrics[f"coverage90_{tag}"].append(
                uct.metrics_calibration.get_proportion_in_interval(
                    mean_preds, std, y_test, quantile=0.90))

        print(f"\n  MAE={fold_mae:.4f}  RMSE={fold_rmse:.4f}  R²={fold_r2:.4f}")
        print(f"  Calibration before → MACE={metrics['mace_before'][-1]:.4f}  "
              f"RMSCE={metrics['rmsce_before'][-1]:.4f}  MA={metrics['ma_before'][-1]:.4f}")
        print(f"  Calibration after  → MACE={metrics['mace_after'][-1]:.4f}  "
              f"RMSCE={metrics['rmsce_after'][-1]:.4f}  MA={metrics['ma_after'][-1]:.4f}")

        all_preds.extend(mean_preds)
        all_trues.extend(y_test)
        all_stds.extend(std_recal)
        all_stds_uncal.extend(std_uncal)

    return dict(
        preds=np.array(all_preds),
        trues=np.array(all_trues),
        stds=np.array(all_stds),
        stds_uncal=np.array(all_stds_uncal),
        metrics={k: np.array(v) for k, v in metrics.items()},
    )


def print_summary(metrics: dict) -> None:
    """Print mean ± std of all collected metrics."""
    print("\n" + "=" * 60)
    print("  Final Cross-Validation Summary")
    print("=" * 60)
    print(f"  MAE  : {metrics['mae'].mean():.4f} ± {metrics['mae'].std():.4f} eV")
    print(f"  RMSE : {metrics['rmse'].mean():.4f} ± {metrics['rmse'].std():.4f} eV")
    print(f"  R²   : {metrics['r2'].mean():.4f} ± {metrics['r2'].std():.4f}")
    print("\n  -- Calibration BEFORE isotonic recalibration --")
    for key in ("mace_before", "rmsce_before", "ma_before"):
        label = key.split("_")[0].upper()
        print(f"  {label:6s}: {metrics[key].mean():.4f} ± {metrics[key].std():.4f}")
    print("\n  -- Calibration AFTER isotonic recalibration --")
    for key in ("mace_after", "rmsce_after", "ma_after"):
        label = key.split("_")[0].upper()
        print(f"  {label:6s}: {metrics[key].mean():.4f} ± {metrics[key].std():.4f}")
    print("\n  -- Coverage at 90% interval level --")
    for key, label in [("coverage90_before", "Before recal"),
                       ("coverage90_after",  "After recal")]:
        print(f"  {label}: {metrics[key].mean()*100:.1f}% ± {metrics[key].std()*100:.1f}%"
              f"  (ideal: 90.0%)")    


def make_figure_target_uncalibrated(results_target: dict, out_path: str) -> None:
    """
    Generate Figure panels (d), (e), and (f) for the Target model 
    BEFORE recalibration (using stds_uncal).
    """

    # ---- Reviewer-Requested Styling (High Prominence) ----
    FONT      = "DejaVu Sans"
    LABEL_FS  = 20  # Large axis labels
    TICK_FS   = 18  # Large, readable tick numbers
    ANNOT_FS  = 16  # Bold stats
    PANEL_FS  = 24  # Very prominent panel letters
    DPI       = 600
    
    # We are specifically creating a single row, but labeling them d, e, f
    PANEL_LABELS = ["d", "e", "f"]

    fig = plt.figure(figsize=(18, 6)) 
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.3)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    # ---- Helper: Heavy Styling for Reviewers ----
    def _style(ax):
        ax.tick_params(labelsize=TICK_FS, width=2.5, length=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(2.0)
        ax.spines["bottom"].set_linewidth(2.0)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontname(FONT)
            lbl.set_weight("bold") # Bold numbers are easier to see in journals

    def _panel_label(ax, label):
        ax.text(
            -0.15, 1.15, f"({label})",
            transform=ax.transAxes,
            fontsize=PANEL_FS,
            fontweight="bold",
            va="top", ha="right"
        )

    # ---- Unpack Target Data (Uncalibrated) ----
    p_tg = results_target["preds"]
    t_tg = results_target["trues"]
    s_tg = results_target["stds_uncal"] # Use uncalibrated standard deviations

    # ---- (d) Parity Plot ----
    ax0 = axes[0]
    ax0.scatter(t_tg, p_tg, color="darkblue", alpha=0.5, s=30, zorder=3)
    ax0.errorbar(t_tg, p_tg, yerr=s_tg, fmt="none", ecolor="darkblue",
                 elinewidth=1.5, alpha=0.3, capsize=2, zorder=2)
    lims = [0, 22]
    ax0.plot(lims, lims, "--", color="black", lw=2.5, zorder=4)
    ax0.set_xlabel("True Band Gap (eV)", fontsize=LABEL_FS, fontweight="bold")
    ax0.set_ylabel("Predicted Band Gap (eV)", fontsize=LABEL_FS, fontweight="bold")
    ax0.set_aspect("equal")
    _style(ax0)
    _panel_label(ax0, "d")

    # ---- (e) Calibration Curve ----
    ax1 = axes[1]
    exp_p, obs_p = uct.get_proportion_lists_vectorized(p_tg, s_tg, t_tg, prop_type="interval")
    ma = uct.metrics_calibration.miscalibration_area(p_tg, s_tg, t_tg, prop_type="interval")
    
    ax1.plot([0, 1], [0, 1], "--", color="black", lw=2.5, label="Ideal")
    ax1.plot(exp_p, obs_p, lw=3.5, color="#1f77b4", label="Target (Uncal)")
    ax1.fill_between(exp_p, exp_p, obs_p, alpha=0.2, color="#1f77b4")
    ax1.set_xlabel("Predicted Proportion", fontsize=LABEL_FS, fontweight="bold")
    ax1.set_ylabel("Observed Proportion", fontsize=LABEL_FS, fontweight="bold")
    ax1.text(0.95, 0.05, f"Miscalibration Area: {ma:.2f}",
             transform=ax1.transAxes, ha="right", va="bottom",
             fontsize=ANNOT_FS, fontweight="bold")
    _style(ax1)
    _panel_label(ax1, "e")

    # ---- (f) Sharpness Histogram ----
    ax2 = axes[2]
    sharpness = float(np.sqrt(np.mean(s_tg ** 2)))
    ax2.hist(s_tg, bins=25, color="#a5c8e1", edgecolor="#1f77b4", density=True, alpha=0.8)
    ax2.axvline(sharpness, color="black", lw=3, ls="--")
    ax2.set_xlabel("Predicted Std. Dev. (eV)", fontsize=LABEL_FS, fontweight="bold")
    ax2.set_ylabel("Density", fontsize=LABEL_FS, fontweight="bold")
    ax2.text(0.95, 0.90, f"Sharpness: {sharpness:.2f} eV",
             transform=ax2.transAxes, ha="right", va="top",
             fontsize=ANNOT_FS, fontweight="bold")
    _style(ax2)
    _panel_label(ax2, "f")

    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Uncalibrated Target Figure (d, e, f) to: {out_path}")
def save_all_figures(results_tl: dict, results_target: dict, out_dir: Path) -> None:
    """Helper to organize figure saving."""
    out_dir.mkdir(exist_ok=True)
    
    # Generate panels (d), (e), and (f) using the Target Model results
    make_figure_target_uncalibrated(
        results_target=results_target,
        out_path=str(out_dir / "figure_target_uncal.png")
    )
    print(f"  Saved Target Figure (d,e,f) to: {out_dir / 'figure_target_uncal.png'}")
def main() -> None:
    X, y = load_data(DATA_DIR)

    # Run nested CV for TL model (Row 1 data)
    print("Running nested CV for transfer learning model...")
    results_tl = run_nested_cv(X, y)
    print_summary(results_tl["metrics"])

    # Run nested CV for target model (Row 2 data - d, e, f)
    print("\nRunning nested CV for target model...")
    results_target = run_nested_cv(X, y)
    print_summary(results_target["metrics"])

    print("\nSaving figures ...")
    save_all_figures(results_tl, results_target, RESULTS_DIR)
    print("\nDone.")

if __name__ == "__main__":
    main()
