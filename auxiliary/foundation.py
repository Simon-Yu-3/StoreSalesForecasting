"""Frozen T0 Beta forecasts for independent store-family series."""

from __future__ import annotations

import numpy as np

MODEL_ID = "theforecastingcompany/t0-beta"
MODEL_REVISION = "941e522547ec63f0f2aad57089fa0f5b8498bdf8"


class T0Forecaster:
    """Benchmark adapter for The Forecasting Company's pretrained T0 Beta.

    The public interface takes sales in original units. The model sees log1p
    sales so its target is on the scale used in the RMSLE loss. The returned
    median is inverted with expm1 and clipped at zero for nonnegative sales.
    Promotion counts are supplied in original units as known covariates.
    Missing historical dates remain NaN at their original daily positions.
    """

    def __init__(
        self, *, batch_size: int = 16, device: str = "cpu", context_length: int = 512
    ) -> None:
        if batch_size < 1 or context_length < 1:
            raise ValueError("batch_size and context_length must both be positive")
        self.batch_size = batch_size
        self.context_length = context_length
        self.device = device

        try:
            if device == "mlx":
                from t0.mlx import T0Forecaster as PretrainedT0
            else:
                from t0 import T0Forecaster as PretrainedT0
        except ImportError as exc:
            extra = "mlx" if device == "mlx" else "torch"
            raise RuntimeError(
                f"T0 requires a working tfc-t0[{extra}]==0.6.1 installation. {exc}"
            ) from exc

        self.model = PretrainedT0.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, token=False
        ).eval()
        if device != "mlx":
            self.model.to(device)

    def predict(
        self,
        history: np.ndarray,
        horizon: int,
        historical_promo: np.ndarray | None = None,
        future_promo: np.ndarray | None = None,
    ) -> np.ndarray:
        """Predict a fixed horizon from a (series, time) array ending at the origin."""
        history = np.asarray(history, dtype=np.float32)
        if history.ndim != 2 or min(history.shape) == 0 or horizon < 1:
            raise ValueError(
                "Expected a nonempty (series, time) history and a positive horizon"
            )
        if np.isinf(history).any() or (history < 0).any():
            raise ValueError("Historical sales must be nonnegative or NaN")

        recent_history = np.ascontiguousarray(history[:, -self.context_length :])
        if np.isnan(recent_history).all(axis=1).any():
            raise ValueError(
                "Each series must have an observation in the retained context"
            )
        log_history = np.log1p(recent_history)

        if (historical_promo is None) != (future_promo is None):
            raise ValueError(
                "Provide both historical and future promotion counts, or neither"
            )
        promotion = None
        if historical_promo is not None:
            historical_promo = np.asarray(historical_promo, dtype=np.float32)
            future_promo = np.asarray(future_promo, dtype=np.float32)
            if historical_promo.shape != history.shape or future_promo.shape != (
                len(history),
                horizon,
            ):
                raise ValueError(
                    "Promotion arrays must align with every series, date, and lead"
                )
            if np.isinf(historical_promo).any() or (historical_promo < 0).any():
                raise ValueError(
                    "Historical promotion counts must be nonnegative or NaN"
                )
            if not np.isfinite(future_promo).all() or (future_promo < 0).any():
                raise ValueError(
                    "Nonnegative promotion counts must be known across the horizon"
                )
            promotion = np.concatenate(
                (historical_promo[:, -self.context_length :], future_promo), axis=1
            )[:, None, :]

        predictions = np.empty((len(history), horizon), dtype=np.float64)
        for start in range(0, len(history), self.batch_size):
            batch = slice(start, start + self.batch_size)
            forecast = self.model.predict(
                log_history[batch],
                horizon=horizon,
                quantile_levels=(0.5,),
                future_covariates=None if promotion is None else promotion[batch],
            )
            median = forecast.median
            if self.device != "mlx":
                median = median.detach().cpu().numpy()
            median = np.asarray(median, dtype=np.float64)
            if median.shape != predictions[batch].shape:
                raise RuntimeError(f"Unexpected T0 forecast shape: {median.shape}")
            predictions[batch] = np.expm1(np.maximum(median, 0))

        if not np.isfinite(predictions).all():
            raise RuntimeError("T0 returned nonfinite forecast values")
        return predictions
