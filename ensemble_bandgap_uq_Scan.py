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
    ECFPs_Scan_466.npy          – ECFP fingerprints (N × D)
    Bandgaps_Scan_466.npy       – Band-gap labels  (N,)
    infinity_feat_index_Scan.npy   – Indices of infinite-valued features to drop

Usage
-----
    python ensemble_bandgap_uq_Scan.py
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
        {"dropout_rate": 0.05, "learning_rate": 9.281189140913416e-05},
        {"dropout_rate": 0.2, "learning_rate": 8.993450833636356e-05},
        {"dropout_rate": 0.05, "learning_rate": 3.3251024096775584e-05},
        {"dropout_rate": 0.15000000000000002, "learning_rate": 2.4959172265536487e-05},
        {"dropout_rate": 0.1, "learning_rate": 9.987144505919778e-05},
    ],
    [  # Outer fold 2
        {"dropout_rate": 0.2, "learning_rate": 0.0001},
        {"dropout_rate": 0.2, "learning_rate": 6.722394207498977e-05},
        {"dropout_rate": 0.2, "learning_rate": 9.196289826868347e-05},
        {"dropout_rate": 0.2, "learning_rate": 9.999989055984263e-05},
        {"dropout_rate": 0.15000000000000002, "learning_rate": 9.422599094454745e-05},
    ],
    [  # Outer fold 3
        {"dropout_rate": 0.15000000000000002, "learning_rate": 0.0001},
        {"dropout_rate": 0.1, "learning_rate": 5.449465169193423e-05},
        {"dropout_rate": 0.0, "learning_rate": 2.891231593599313e-05},
        {"dropout_rate": 0.15000000000000002, "learning_rate": 8.199885714339541e-05},
        {"dropout_rate": 0.15000000000000002, "learning_rate": 5.065608940509875e-05},
    ],
    [  # Outer fold 4
        {"dropout_rate": 0.15000000000000002, "learning_rate": 6.966236101827243e-05},
        {"dropout_rate": 0.1, "learning_rate": 6.089349091864901e-05},
        {"dropout_rate": 0.15000000000000002, "learning_rate": 9.192435025620423e-05},
        {"dropout_rate": 0.2, "learning_rate": 9.173636939517865e-05},
        {"dropout_rate": 0.0, "learning_rate": 5.175650888793403e-05},
    ],
    [  # Outer fold 5
        {"dropout_rate": 0.1, "learning_rate": 8.286828766060395e-05},
        {"dropout_rate": 0.2, "learning_rate": 5.3946841738746306e-05},
        {"dropout_rate": 0.2, "learning_rate": 6.218231706703658e-05},
        {"dropout_rate": 0.1, "learning_rate": 9.024695276519941e-05},
        {"dropout_rate": 0.15000000000000002, "learning_rate": 2.0941965910352642e-05},
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
    ECFP_all = np.load(data_dir / "ECFPs_Scan_466.npy", allow_pickle=True)
    BG_all = np.load(data_dir / "Bandgaps_Scan_466.npy", allow_pickle=True)
    inf_index = np.load(data_dir / "infinity_feat_index_Scan.npy", allow_pickle=True)

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
                   mace_after=[], rmsce_after=[], ma_after=[])

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


# ===========================================================================
# Plotting
# ===========================================================================

_FONT = "DejaVu Sans"
_LABEL_FS = 18
_TICK_FS = 16
_LEGEND_FS = 16
_TEXT_FS = 16
_DPI = 600


def _apply_base_style(ax: plt.Axes) -> None:
    ax.tick_params(labelsize=_TICK_FS)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname(_FONT)


def plot_calibration_curve(
    y_pred: np.ndarray,
    y_std: np.ndarray,
    y_true: np.ndarray,
    curve_label: str = "Predictor",
    num_bins: int = 100,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot an interval-calibration curve with shaded miscalibration area.

    Parameters
    ----------
    y_pred, y_std, y_true:
        Predicted means, standard deviations, and ground-truth values.
    curve_label:
        Legend label for the calibration curve.
    num_bins:
        Number of probability bins.
    ax:
        Existing ``Axes`` to draw on; a new figure is created if ``None``.

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    exp_props, obs_props = uct.metrics_calibration.get_proportion_lists_vectorized(
        y_pred, y_std, y_true, num_bins=num_bins, prop_type="interval"
    )
    miscal_area = uct.metrics_calibration.miscalibration_area(
        y_pred, y_std, y_true, num_bins=num_bins, vectorized=True, prop_type="interval"
    )

    ax.plot([0, 1], [0, 1], "--", color="black", lw=2, label="Ideal")
    ax.plot(exp_props, obs_props, lw=2.5, label=curve_label)
    ax.fill_between(exp_props, exp_props, obs_props, alpha=0.25)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Predicted Proportion Interval", fontsize=_LABEL_FS, fontname=_FONT)
    ax.set_ylabel("Observed Proportion Interval", fontsize=_LABEL_FS, fontname=_FONT)
    _apply_base_style(ax)

    leg = ax.legend(fontsize=_LEGEND_FS, frameon=False, loc="upper left")
    for t in leg.get_texts():
        t.set_fontname(_FONT)

    ax.text(
        0.98, 0.05,
        f"Miscalibration area = {miscal_area:.2f}",
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=_TEXT_FS, fontname=_FONT,
    )
    return ax


def plot_sharpness(
    y_std: np.ndarray,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot a histogram of predictive standard deviations with sharpness line.

    Parameters
    ----------
    y_std:
        Predicted standard deviations.
    ax:
        Existing ``Axes`` to draw on; a new figure is created if ``None``.

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    sharpness = float(np.sqrt(np.mean(y_std ** 2)))
    ax.hist(y_std, edgecolor="#1f77b4", color="#a5c8e1", density=True)
    ax.axvline(sharpness, color="k", lw=2.5, ls="--",
               label=f"Sharpness = {sharpness:.2f} eV")

    ax.set_xlim(0.05, 1.05 * y_std.max())
    ax.set_yticks([])
    ax.set_xlabel("Predicted Std. Dev. (eV)", fontsize=_LABEL_FS, fontname=_FONT)
    ax.set_ylabel("Normalised Frequency", fontsize=_LABEL_FS, fontname=_FONT)
    _apply_base_style(ax)

    ax.text(0.98, 0.95, f"Sharpness = {sharpness:.2f} eV",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=_TEXT_FS, fontname=_FONT)
    return ax


def plot_parity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    xlabel: str = "True Band Gap (eV)",
    ylabel: str = "Predicted Band Gap (eV)",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Scatter parity plot with per-point error bars.

    Parameters
    ----------
    y_true, y_pred, y_std:
        Ground-truth, predicted means, and predicted standard deviations.
    xlabel, ylabel:
        Axis labels.
    ax:
        Existing ``Axes`` to draw on; a new figure is created if ``None``.

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(y_true, y_pred, color="darkblue", alpha=0.7, s=25, zorder=3)
    ax.errorbar(y_true, y_pred, yerr=y_std, fmt="none",
                ecolor="darkblue", elinewidth=2.5, alpha=0.7, capsize=3, zorder=2)

    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "--", color="black", lw=2)

    ax.set_xlabel(xlabel, fontsize=_LABEL_FS, fontname=_FONT)
    ax.set_ylabel(ylabel, fontsize=_LABEL_FS, fontname=_FONT)
    ax.set_aspect("equal", adjustable="box")
    _apply_base_style(ax)
    plt.tight_layout()
    return ax


def save_all_figures(results: dict, out_dir: Path) -> None:
    """Generate and save all diagnostic figures.

    Parameters
    ----------
    results:
        Output dict from :func:`run_nested_cv`.
    out_dir:
        Directory in which to save the figures.
    """
    out_dir.mkdir(exist_ok=True)
    p, t, s, su = (results["preds"], results["trues"],
                   results["stds"], results["stds_uncal"])

    fig_specs = [
        ("calibration_before.png",
         lambda: plot_calibration_curve(p, su, t, curve_label="Before recalibration")),
        ("calibration_after.png",
         lambda: plot_calibration_curve(p, s,  t, curve_label="After recalibration")),
        ("sharpness_before.png",
         lambda: plot_sharpness(su)),
        ("sharpness_after.png",
         lambda: plot_sharpness(s)),
        ("parity_before.png",
         lambda: plot_parity(t, p, su)),
        ("parity_after.png",
         lambda: plot_parity(t, p, s)),
    ]

    for fname, make_ax in fig_specs:
        ax = make_ax()
        ax.figure.savefig(out_dir / fname, dpi=_DPI, bbox_inches="tight")
        plt.close(ax.figure)
        print(f"  Saved: {out_dir / fname}")


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    X, y = load_data(DATA_DIR)
    results = run_nested_cv(X, y)
    print_summary(results["metrics"])
    print("\nSaving figures …")
    save_all_figures(results, RESULTS_DIR)
    print("\nDone.")


if __name__ == "__main__":
    main()
