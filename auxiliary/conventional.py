"""Conventional forecasts in the store-family panel's series order."""

import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

CONTEXT_LENGTH = 512
RECENT_WEEKS = 8


def weekday_forecast(history, history_dates, forecast_dates, *, average=False):
    """Repeat the last observed weekday or pool the last eight such weekdays."""
    horizon = len(forecast_dates)
    prediction = np.empty((history.shape[0], horizon), dtype=float)
    rows = np.arange(history.shape[0])
    weekday = history_dates.dayofweek.to_numpy()
    if not np.isfinite(history).any(axis=1).all():
        raise ValueError("Each series must have observed history before the cutoff.")

    for future_day in forecast_dates.dayofweek.unique():
        candidates = history[:, weekday == future_day]
        observed = np.isfinite(candidates)
        if not observed.any(axis=1).all():
            raise ValueError(
                f"No observed value for at least one series on weekday {future_day}."
            )
        last_valid = candidates.shape[1] - 1 - observed[:, ::-1].argmax(axis=1)
        values = candidates[rows, last_valid]
        if average:
            log_values = np.log1p(candidates[:, -RECENT_WEEKS:])
            counts = np.isfinite(log_values).sum(axis=1)
            log_means = np.divide(
                np.nansum(log_values, axis=1),
                counts,
                out=np.log1p(values),
                where=counts > 0,
            )
            values = np.expm1(log_means)
        prediction[:, forecast_dates.dayofweek == future_day] = values[:, None]
    return prediction


def prophet_forecast(
    history,
    history_dates,
    forecast_dates,
    *,
    historical_promo=None,
    future_promo=None,
    changepoint_prior_scale=0.05,
    jobs=1,
):
    """Fit independent Prophet models in log1p sales units."""
    from prophet import Prophet

    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    context = history[:, -CONTEXT_LENGTH:]
    context_dates = history_dates[-CONTEXT_LENGTH:]
    if (historical_promo is None) != (future_promo is None):
        raise ValueError(
            "Promotion forecasts require historical and future promotions."
        )

    def fit_series(series_idx):
        sales = context[series_idx]
        observed = np.isfinite(sales)
        log_sales = np.log1p(sales[observed])
        if log_sales.size == 0:
            raise ValueError(
                f"Series {series_idx} has no observations in Prophet's context."
            )
        if np.all(log_sales == log_sales[0]):
            return np.full(len(forecast_dates), sales[observed][0])
        training = pd.DataFrame({"ds": context_dates[observed], "y": log_sales})
        future = pd.DataFrame({"ds": forecast_dates})
        model = Prophet(
            weekly_seasonality=True,
            yearly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=changepoint_prior_scale,
            uncertainty_samples=0,
        )
        if historical_promo is not None:
            promotion = historical_promo[series_idx, -CONTEXT_LENGTH:][observed]
            future_promotion = future_promo[series_idx]
            if not (
                np.isfinite(promotion).all() and np.isfinite(future_promotion).all()
            ):
                raise ValueError(f"Missing promotion count for series {series_idx}.")
            if np.ptp(promotion) > 0:
                model.add_regressor("onpromotion")
                training["onpromotion"] = promotion
                future["onpromotion"] = future_promotion
        forecast = model.fit(training).predict(future)["yhat"].to_numpy()
        return np.maximum(np.expm1(forecast), 0)

    series_indices = range(context.shape[0])
    if jobs == 1:
        predictions = [fit_series(i) for i in series_indices]
    else:
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            predictions = list(pool.map(fit_series, series_indices))
    return np.stack(predictions)
