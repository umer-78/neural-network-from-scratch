"""Charts for the reports folder."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

STYLE = {"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
         "axes.spines.right": False, "font.size": 9}


def learning_curves(history, path: str | Path) -> Path:
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
        epochs = range(1, len(history.loss) + 1)
        ax[0].plot(epochs, history.loss, label="train")
        if history.val_loss:
            ax[0].plot(epochs, history.val_loss, label="validation")
        ax[0].set_title("Cross-entropy loss")
        ax[1].plot(epochs, history.accuracy, label="train")
        if history.val_accuracy:
            ax[1].plot(epochs, history.val_accuracy, label="validation")
        ax[1].set_title("Accuracy")
        for a in ax:
            a.set_xlabel("epoch")
            a.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
    return Path(path)


def confusion(y_true: np.ndarray, y_pred: np.ndarray, path: str | Path, labels=None) -> Path:
    k = int(max(y_true.max(), y_pred.max())) + 1
    matrix = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[int(t), int(p)] += 1
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        ax.imshow(matrix, cmap="Blues")
        ax.set_xlabel("predicted")
        ax.set_ylabel("actual")
        ax.set_xticks(range(k), labels or range(k))
        ax.set_yticks(range(k), labels or range(k))
        ax.grid(False)
        for i in range(k):
            for j in range(k):
                if matrix[i, j]:
                    ax.text(j, i, matrix[i, j], ha="center", va="center", fontsize=7,
                            color="white" if matrix[i, j] > matrix.max() / 2 else "black")
        ax.set_title("Confusion matrix (test set)")
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
    return Path(path)


def mistakes(images, y_true, y_pred, path: str | Path, count: int = 10) -> Path | None:
    wrong = np.flatnonzero(y_true != y_pred)[:count]
    if not len(wrong):
        return None
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 5, figsize=(7, 3.2))
        for ax, idx in zip(axes.ravel(), wrong):
            ax.imshow(images[idx], cmap="gray_r")
            ax.set_title(f"{y_true[idx]} → {y_pred[idx]}", fontsize=8)
            ax.axis("off")
        for ax in axes.ravel()[len(wrong):]:
            ax.axis("off")
        fig.suptitle("Every mistake on the test set (actual → predicted)", fontsize=9)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
    return Path(path)
