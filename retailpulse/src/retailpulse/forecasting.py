"""F03 – Demand forecasting: Prophet baseline + PyTorch-Lightning LSTM + weighted ensemble.

Target: daily units sold (winsorised), modelled in log1p space on *trading days*.
Evaluation: rolling-origin back-test, 30-day horizon, plus a final untouched hold-out.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from retailpulse import config

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)

H = config.FORECAST_HORIZON
WINDOW = 56  # LSTM look-back in calendar days (8 weeks)


# ---------------------------------------------------------------- calendar ---
def _uk_holidays(years) -> set[pd.Timestamp]:
    try:
        import holidays

        return {pd.Timestamp(d) for d in holidays.country_holidays("GB", years=list(years)).keys()}
    except Exception:  # pragma: no cover
        return set()


def is_trading_day(dates: pd.DatetimeIndex) -> np.ndarray:
    """The retailer never trades on Saturdays, UK bank holidays or 24 Dec - 3 Jan."""
    dates = pd.DatetimeIndex(dates)
    hol = _uk_holidays(range(dates.year.min(), dates.year.max() + 1))
    xmas = ((dates.month == 12) & (dates.day >= 24)) | ((dates.month == 1) & (dates.day <= 3))
    return ~((dates.dayofweek == 5) | xmas | dates.isin(list(hol)))


def calendar_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
    d = pd.DatetimeIndex(dates)
    doy = d.dayofyear.values
    dow = d.dayofweek.values
    xmas = pd.to_datetime(d.year.astype(str) + "-12-25")
    to_xmas = ((xmas - d).days).values
    return pd.DataFrame(
        {
            "dow_sin": np.sin(2 * np.pi * dow / 7),
            "dow_cos": np.cos(2 * np.pi * dow / 7),
            "doy_sin": np.sin(2 * np.pi * doy / 365.25),
            "doy_cos": np.cos(2 * np.pi * doy / 365.25),
            "pre_xmas": np.clip(1 - np.abs(to_xmas - 14) / 60, 0, 1) * (to_xmas >= 0),
            "is_open": is_trading_day(d).astype(float),
        },
        index=d,
    )


# ---------------------------------------------------------------- metrics ----
def mape(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = y > 0
    return float(np.mean(np.abs((y[m] - p[m]) / y[m])) * 100) if m.any() else float("nan")


def wape(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.abs(y - p).sum() / max(y.sum(), 1e-9) * 100)


def horizon_metrics(df: pd.DataFrame, col: str) -> dict:
    """df has date, actual, <col>.  Daily MAPE/WAPE, weekly MAPE (4 blocks of 7 days), 30-day total APE."""
    d = df.sort_values("date").reset_index(drop=True)
    d["blk"] = np.arange(len(d)) // 7
    wk = d[d["blk"] < 4].groupby("blk")[["actual", col]].sum()
    tot_a, tot_p = d["actual"].sum(), d[col].sum()
    return {
        "mape_daily": mape(d["actual"], d[col]),
        "wape_daily": wape(d["actual"], d[col]),
        "mape_weekly": mape(wk["actual"], wk[col]),
        "ape_30d_total": float(abs(tot_a - tot_p) / max(tot_a, 1e-9) * 100),
    }


# ---------------------------------------------------------------- Prophet ----
def fit_prophet(train: pd.DataFrame):
    """train: DataFrame[date, y(units)] on trading days.  Multiplicative seasonality, GB holidays.

    Hyper-parameters were chosen on rolling folds ending before 2011-10 (never on the final hold-out)."""
    from prophet import Prophet

    m = Prophet(
        growth="linear",
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.15,
        seasonality_prior_scale=12.0,
        interval_width=0.80,
        uncertainty_samples=300,
    )
    m.add_country_holidays(country_name="GB")
    m.fit(pd.DataFrame({"ds": train["date"], "y": train["y"].clip(lower=0)}))
    return m


def predict_prophet(m, dates: pd.DatetimeIndex) -> pd.DataFrame:
    f = m.predict(pd.DataFrame({"ds": dates}))
    return pd.DataFrame(
        {
            "date": dates,
            "prophet": f["yhat"].clip(lower=0).values,
            "prophet_lo": f["yhat_lower"].clip(lower=0).values,
            "prophet_hi": f["yhat_upper"].clip(lower=0).values,
        }
    )


# ------------------------------------------------------------------ LSTM -----
def _build_lstm_cls():
    import lightning as L
    import torch
    from torch import nn

    class DemandLSTM(L.LightningModule):
        """Encoder LSTM over the last 8 weeks + per-day decoder with known future calendar."""

        def __init__(self, n_in=7, n_cal=6, hidden=48, lr=3e-3):
            super().__init__()
            self.save_hyperparameters()
            self.enc = nn.LSTM(n_in, hidden, batch_first=True, num_layers=1)
            self.dec = nn.Sequential(nn.Linear(hidden + n_cal, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 1))
            self.lr = lr

        def forward(self, x, fut):
            _, (h, _) = self.enc(x)
            h = h[-1].unsqueeze(1).expand(-1, fut.shape[1], -1)
            return self.dec(torch.cat([h, fut], dim=-1)).squeeze(-1)

        def training_step(self, batch, _):
            x, fut, y, mask = batch
            loss = (((self(x, fut) - y) ** 2) * mask).sum() / mask.sum().clamp(min=1)
            self.log("train_loss", loss, prog_bar=False)
            return loss

        def validation_step(self, batch, _):
            x, fut, y, mask = batch
            loss = (((self(x, fut) - y) ** 2) * mask).sum() / mask.sum().clamp(min=1)
            self.log("val_loss", loss, prog_bar=False)

        def configure_optimizers(self):
            return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-3)

    return DemandLSTM


class LSTMForecaster:
    """Direct 30-day multi-horizon forecaster trained on rolling windows."""

    def __init__(self, epochs: int = 120, hidden: int = 48, seed: int = config.SEED):
        self.epochs, self.hidden, self.seed = epochs, hidden, seed
        self.model = None
        self.mu = 0.0
        self.sd = 1.0

    # build calendar-day grid ------------------------------------------------
    def _grid(self, series: pd.Series) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
        idx = pd.date_range(series.index.min(), series.index.max(), freq="D")
        s = series.reindex(idx)
        opened = s.notna().values.astype(float)
        z = np.where(s.notna(), (np.log1p(s.fillna(0).clip(lower=0)) - self.mu) / self.sd, 0.0)
        cal = calendar_features(idx).values.astype(np.float32)
        inp = np.column_stack([z, opened, cal[:, :4], cal[:, 4]]).astype(np.float32)  # 7 channels
        return inp, cal, idx

    def fit(self, series: pd.Series, val_tail: int = 0):
        import lightning as L
        import torch

        L.seed_everything(self.seed, workers=True)
        logv = np.log1p(series.clip(lower=0))
        self.mu, self.sd = float(logv.mean()), float(logv.std() or 1.0)
        inp, cal, idx = self._grid(series)
        z = inp[:, 0]
        opened = inp[:, 1]
        n = len(idx)
        xs, fs, ys, ms = [], [], [], []
        for o in range(WINDOW, n - H + 1):
            xs.append(inp[o - WINDOW : o])
            fs.append(cal[o : o + H])
            ys.append(z[o : o + H])
            ms.append(opened[o : o + H])
        X, F, Y, M = (torch.tensor(np.stack(a)) for a in (xs, fs, ys, ms))
        n_val = max(int(0.12 * len(X)), 0) if val_tail == 0 else val_tail
        tr = torch.utils.data.TensorDataset(X[:-n_val], F[:-n_val], Y[:-n_val], M[:-n_val])
        va = torch.utils.data.TensorDataset(X[-n_val:], F[-n_val:], Y[-n_val:], M[-n_val:])
        cls = _build_lstm_cls()
        self.model = cls(hidden=self.hidden)
        from lightning.pytorch.callbacks import EarlyStopping

        trainer = L.Trainer(
            max_epochs=self.epochs,
            enable_progress_bar=False,
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            accelerator="cpu",
            callbacks=[EarlyStopping("val_loss", patience=20)],
            deterministic=False,
        )
        trainer.fit(
            self.model,
            torch.utils.data.DataLoader(tr, batch_size=32, shuffle=True),
            torch.utils.data.DataLoader(va, batch_size=64),
        )
        self._last_series = series
        return self

    def predict(self, series: pd.Series, origin: pd.Timestamp) -> pd.DataFrame:
        """Forecast the H calendar days starting at `origin` using data strictly before it."""
        import torch

        hist = series[series.index < origin]
        inp, _, idx = self._grid(hist)
        # extend grid through origin-1 (days after the last observation are closed/unknown)
        full_idx = pd.date_range(idx.min(), origin - pd.Timedelta(days=1), freq="D")
        if len(full_idx) > len(idx):
            pad_idx = full_idx[len(idx) :]
            cal_p = calendar_features(pad_idx).values.astype(np.float32)
            pad = np.column_stack([np.zeros(len(pad_idx)), np.zeros(len(pad_idx)), cal_p[:, :4], cal_p[:, 4]])
            inp = np.vstack([inp, pad.astype(np.float32)])
        x = torch.tensor(inp[-WINDOW:][None])
        fdates = pd.date_range(origin, periods=H, freq="D")
        fut = torch.tensor(calendar_features(fdates).values.astype(np.float32)[None])
        self.model.eval()
        with torch.no_grad():
            z = self.model(x, fut).numpy()[0]
        pred = np.expm1(z * self.sd + self.mu).clip(min=0)
        return pd.DataFrame({"date": fdates, "lstm": pred})


# ------------------------------------------------------------- baselines -----
def seasonal_naive(train: pd.Series, dates: pd.DatetimeIndex, weeks: int = 4) -> np.ndarray:
    tail = train[train.index >= train.index.max() - pd.Timedelta(days=7 * weeks)]
    by_dow = tail.groupby(tail.index.dayofweek).mean()
    return np.array([by_dow.get(d.dayofweek, tail.mean()) for d in dates])


# ------------------------------------------------------------ orchestration --
def series_from_daily(daily: pd.DataFrame) -> pd.Series:
    d = daily[daily["is_open"].astype(bool)].copy()
    s = pd.Series(d["units_capped"].values, index=pd.DatetimeIndex(d["date"]), name="units")
    return s


def run_fold(series: pd.Series, origin: pd.Timestamp, lstm_epochs: int = 120) -> pd.DataFrame:
    """Train on data < origin, forecast [origin, origin+H); returns predictions + actuals."""
    train = series[series.index < origin]
    fdates = pd.date_range(origin, periods=H, freq="D")
    fdates = fdates[is_trading_day(fdates)]
    m = fit_prophet(pd.DataFrame({"date": train.index, "y": train.values}))
    p = predict_prophet(m, fdates)
    lf = LSTMForecaster(epochs=lstm_epochs).fit(train)
    lstm_pred = lf.predict(train, origin)
    p = p.merge(lstm_pred, on="date", how="left")
    p["naive"] = seasonal_naive(train, fdates)
    p["actual"] = series.reindex(fdates).values
    p["origin"] = origin
    return p


def best_weight(folds: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """LSTM weight w in [0,1] minimising a composite error over the tuning folds.

    composite = mean(daily MAPE, weekly MAPE, 30-day total APE) per fold, then averaged over folds -
    daily accuracy alone would ignore the weekly / monthly totals that drive replenishment."""
    rows = []
    for w in np.linspace(0, 1, 21):
        per = []
        for _, g in folds.groupby("fold"):
            g = g.assign(blend=(1 - w) * g["prophet"] + w * g["lstm"])
            m = horizon_metrics(g, "blend")
            per.append([m["mape_daily"], m["mape_weekly"], m["ape_30d_total"]])
        per = np.array(per).mean(0)
        rows.append(
            {"w_lstm": w, "mape_daily": per[0], "mape_weekly": per[1], "ape_30d_total": per[2], "composite": per.mean()}
        )
    g = pd.DataFrame(rows)
    return float(g.loc[g["composite"].idxmin(), "w_lstm"]), g
