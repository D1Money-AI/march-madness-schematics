"""
Load NCAA tournament CSV, analyze misses vs published total (honest small-sample stats),
bias-adjusted line, Ridge on residuals, walk-forward backtest, emit HTML.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

try:
    from scipy.stats import binomtest
except ImportError:
    binomtest = None  # type: ignore[misc, assignment]

DIR = Path(__file__).resolve().parent
CSV_PATH = DIR / "NCAA_Mens_Tournament_2026_Complete_Results.csv"
OUT_HTML = DIR / "march_madness_totals_2026.html"
TEMPLATE_PATH = DIR / "html_template.html"

BURN_IN = 10
MAE_TIE_EPS = 0.5
RIDGE_ALPHA = 1.0
# Dates with scheduled games and valid lines but no final score yet (shown as upcoming in HTML).
UPCOMING_DATES: set[str] = {"04/04/26"}
BOOTSTRAP_B = 2500
# Smaller bootstrap when analyze_misses_vs_line runs inside per-game walk-forward (many calls).
BOOTSTRAP_B_WALKFORWARD = 600
BOOTSTRAP_SEED = 0
BOOTSTRAP_ALPHA = 0.10  # 90% CI for mean residual
# Treat projection "on the line" if within this many points (no directional vote).
OU_VOTE_EPS = 0.25


def normalize_team(name: str) -> str:
    s = str(name).strip().lower()
    for prefix in ("university of ", "the "):
        if s.startswith(prefix):
            s = s[len(prefix) :]
    return s.strip()


def matchup_key(row: pd.Series) -> tuple:
    a, b = normalize_team(row["Team A"]), normalize_team(row["Team B"])
    return (row["_dt"], frozenset((a, b)))


def parse_spread_magnitude(spread: str) -> float | None:
    if spread is None or (isinstance(spread, float) and np.isnan(spread)):
        return None
    s = str(spread).strip()
    if not s or s.upper() == "TBD":
        return None
    s_up = s.upper().replace("'", "")
    if s_up in ("PK", "PICK", "PICKEM", "PICK EM"):
        return 0.0
    if re.search(r"\bPK\b", s_up):
        return 0.0
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*$", s)
    if not m:
        return None
    return abs(float(m.group(1)))


def _line_columns(df: pd.DataFrame) -> tuple[str, str]:
    ou = "Market Over/Under" if "Market Over/Under" in df.columns else "MGM Over/Under"
    sp = "Market Point Spread" if "Market Point Spread" in df.columns else "MGM Point Spread"
    return ou, sp


def load_and_clean() -> tuple[pd.DataFrame, pd.DataFrame, datetime]:
    df = pd.read_csv(CSV_PATH)
    df["_dt"] = pd.to_datetime(df["Date"], format="%m/%d/%y", errors="coerce")
    df = df.dropna(subset=["_dt"])
    ou_col, sp_col = _line_columns(df)
    df["MGM_OU"] = pd.to_numeric(df[ou_col], errors="coerce")
    df["Total"] = pd.to_numeric(df["Total Combined Score"], errors="coerce")
    df["spread_mag"] = df[sp_col].apply(parse_spread_magnitude)

    completed = df[
        (df["Total"] > 0)
        & df["MGM_OU"].notna()
        & (df["MGM_OU"] > 0)
        & df["spread_mag"].notna()
        & ~df["Team A"].astype(str).str.upper().eq("TBD")
    ].copy()

    completed = completed.sort_values(["_dt", "Team A", "Team B"]).reset_index(drop=True)
    completed["_key"] = completed.apply(matchup_key, axis=1)
    completed = completed.drop_duplicates(subset=["_key"], keep="last").drop(
        columns=["_key"]
    )
    completed = completed.sort_values("_dt").reset_index(drop=True)

    t0 = completed["_dt"].min()
    completed["days_since_start"] = (completed["_dt"] - t0).dt.days.astype(float)

    upcoming = df[
        df["Date"].astype(str).isin(UPCOMING_DATES)
        & ~df["Team A"].astype(str).str.upper().eq("TBD")
        & df["MGM_OU"].notna()
        & (df["MGM_OU"] > 0)
        & df["spread_mag"].notna()
    ].copy()
    upcoming = upcoming.sort_values(["_dt", "Team A"]).reset_index(drop=True)

    return completed, upcoming, t0.to_pydatetime()


def build_X(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            df["MGM_OU"].values,
            df["spread_mag"].values,
            df["days_since_start"].values,
        ]
    )


def bootstrap_mean_ci(
    d: np.ndarray, n_boot: int, seed: int, alpha: float
) -> tuple[float, float]:
    """Percentile CI for the mean of d (nonparametric bootstrap)."""
    n = len(d)
    if n == 0:
        return float("nan"), float("nan")
    if n == 1:
        m = float(d[0])
        return m, m
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[b] = float(np.mean(d[idx]))
    lo = float(np.percentile(means, 100 * (alpha / 2)))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, hi


def analyze_misses_vs_line(
    completed: pd.DataFrame, *, n_boot: int | None = None
) -> dict:
    """
    D = actual total − published total for each completed game.
    Tournament-wide evidence for over vs under vs the line.
    """
    line = completed["MGM_OU"].values
    total = completed["Total"].values
    d = (total - line).astype(float)
    n = int(len(d))
    if n == 0:
        return {
            "n_games_residual": 0,
            "ou_pick_rule": (
                "Majority vote among Bias+line vs total, Residual ridge vs total, and tournament lean."
            ),
            "residual_mean": None,
            "residual_sd": None,
            "residual_se": None,
            "residual_boot_ci90_low": None,
            "residual_boot_ci90_high": None,
            "miss_p10": None,
            "miss_p90": None,
            "n_over": 0,
            "n_under": 0,
            "pct_over": None,
            "binom_p_two_sided_vs_50": None,
            "lean_vs_line": "none",
        }

    mean_d = float(np.mean(d))
    sd_d = float(np.std(d, ddof=1)) if n > 1 else 0.0
    se_d = float(sd_d / np.sqrt(n)) if n > 0 else float("nan")
    n_boot_eff = BOOTSTRAP_B if n_boot is None else int(n_boot)
    boot_lo, boot_hi = bootstrap_mean_ci(
        d, n_boot_eff, BOOTSTRAP_SEED, BOOTSTRAP_ALPHA
    )
    p10 = float(np.percentile(d, 10))
    p90 = float(np.percentile(d, 90))

    over_mask = d > 0
    under_mask = d < 0
    n_over = int(np.sum(over_mask))
    n_under = int(np.sum(under_mask))
    n_non_push = n_over + n_under
    pct_over = round(100.0 * n_over / n, 1) if n else None

    p_binom = None
    if binomtest is not None and n_non_push > 0:
        bt = binomtest(n_over, n_non_push, p=0.5, alternative="two-sided")
        p_binom = float(bt.pvalue)

    if boot_lo > 0:
        lean = "over"
    elif boot_hi < 0:
        lean = "under"
    else:
        lean = "none"

    return {
        "n_games_residual": n,
        "ou_pick_rule": (
            "Each row’s pick is a majority vote among up to three signals: "
            "(1) Bias + line vs published total, (2) Residual ridge vs published total, "
            "(3) tournament-wide bootstrap lean (over / under / none). "
            f"Projections within {OU_VOTE_EPS:g} pt of the line count as no vote. "
            "Ties → No pick."
        ),
        "residual_mean": round(mean_d, 3),
        "residual_sd": round(sd_d, 3),
        "residual_se": round(se_d, 4) if n > 1 and not np.isnan(se_d) else None,
        "residual_boot_ci90_low": round(boot_lo, 3),
        "residual_boot_ci90_high": round(boot_hi, 3),
        "miss_p10": round(p10, 2),
        "miss_p90": round(p90, 2),
        "n_over": n_over,
        "n_under": n_under,
        "pct_over": pct_over,
        "binom_p_two_sided_vs_50": round(p_binom, 4) if p_binom is not None else None,
        "lean_vs_line": lean,
    }


def _vote_side(pred: float, line: float) -> str | None:
    if pred > line + OU_VOTE_EPS:
        return "over"
    if pred < line - OU_VOTE_EPS:
        return "under"
    return None


def combine_ou_pick(
    pred_bias: float,
    pred_reg: float,
    line: float,
    tournament_lean: str,
) -> tuple[str, str]:
    """
    Majority vote: bias vs line, ridge vs line, tournament lean (if over/under).
    Returns (pick, short_reason) with pick in over|under|none.
    """
    votes: list[str] = []
    vb = _vote_side(pred_bias, line)
    if vb:
        votes.append(vb)
    vr = _vote_side(pred_reg, line)
    if vr:
        votes.append(vr)
    if tournament_lean in ("over", "under"):
        votes.append(tournament_lean)

    if not votes:
        return "none", "No signal: both projections on the line and no tournament lean."

    n_over = sum(1 for v in votes if v == "over")
    n_under = sum(1 for v in votes if v == "under")
    if n_over > n_under:
        side = "over"
    elif n_under > n_over:
        side = "under"
    else:
        return "none", "Split vote — no majority."

    who: list[str] = []
    if vb == side:
        who.append("Bias+line")
    if vr == side:
        who.append("Residual ridge")
    if tournament_lean == side:
        who.append("Tournament field")
    reason = "Majority " + side + ": " + " + ".join(who) + "."
    return side, reason


def grade_vs_line(pick_ou: str, actual_total: float, line: float) -> tuple[str, str]:
    """Pick vs closing total line. Push on exact line → No Action."""
    if pick_ou not in ("over", "under"):
        return "no_action", "No Action"
    if actual_total > line + 1e-9:
        outcome = "over"
    elif actual_total < line - 1e-9:
        outcome = "under"
    else:
        return "no_action", "No Action"
    if pick_ou == outcome:
        return "confirmed", "Confirmed"
    return "failed", "Failed"


def build_prediction_row(
    r: pd.Series,
    t0: datetime,
    bias_mean: float,
    reg: Ridge,
    resid_stats: dict,
) -> dict:
    days = (r["_dt"].to_pydatetime() - t0).days
    mgm = float(r["MGM_OU"])
    spread = float(r["spread_mag"])
    X = np.array([[mgm, spread, float(days)]])
    pred_bias = mgm + bias_mean
    pred_res = float(reg.predict(X)[0])
    pred_reg = mgm + pred_res

    boot_lo = resid_stats.get("residual_boot_ci90_low")
    boot_hi = resid_stats.get("residual_boot_ci90_high")
    p10 = resid_stats.get("miss_p10")
    p90 = resid_stats.get("miss_p90")

    bias_lo = bias_hi = None
    if boot_lo is not None and not (isinstance(boot_lo, float) and np.isnan(boot_lo)):
        bias_lo = round(mgm + float(boot_lo), 2)
        bias_hi = round(mgm + float(boot_hi), 2)
    fan_lo = fan_hi = None
    if p10 is not None and p90 is not None:
        fan_lo = round(mgm + float(p10), 2)
        fan_hi = round(mgm + float(p90), 2)

    lean = resid_stats.get("lean_vs_line", "none")
    pick, pick_reason = combine_ou_pick(pred_bias, pred_reg, mgm, lean)

    return {
        "date": r["Date"],
        "team_a": r["Team A"],
        "team_b": r["Team B"],
        "mgm_ou": mgm,
        "spread_mag": spread,
        "pick_ou": pick,
        "pick_reason": pick_reason,
        "pred_bias_adjusted": round(pred_bias, 2),
        "pred_regression": round(pred_reg, 2),
        "bias_offset_used": round(bias_mean, 3),
        "bias_interval_lo": bias_lo,
        "bias_interval_hi": bias_hi,
        "empirical_miss_fan_lo": fan_lo,
        "empirical_miss_fan_hi": fan_hi,
        "lean_vs_line": lean,
    }


def walk_forward_dashboard_rows(completed: pd.DataFrame, t0: datetime) -> list[dict]:
    """One row per completed game: pick uses only prior games (honest lookback)."""
    out: list[dict] = []
    for i in range(len(completed)):
        r = completed.iloc[i]
        past = completed.iloc[:i]
        act = float(r["Total"])
        mgm = float(r["MGM_OU"])
        if len(past) < BURN_IN:
            rc, rl = grade_vs_line("none", act, mgm)
            out.append(
                {
                    "date": r["Date"],
                    "team_a": r["Team A"],
                    "team_b": r["Team B"],
                    "mgm_ou": mgm,
                    "spread_mag": float(r["spread_mag"]),
                    "pick_ou": "none",
                    "pick_reason": (
                        f"Warm-up: first {BURN_IN} games have no prior sample for walk-forward pick."
                    ),
                    "pred_bias_adjusted": None,
                    "pred_regression": None,
                    "bias_offset_used": None,
                    "bias_interval_lo": None,
                    "bias_interval_hi": None,
                    "empirical_miss_fan_lo": None,
                    "empirical_miss_fan_hi": None,
                    "lean_vs_line": None,
                    "actual_total": round(act, 1),
                    "result_code": rc,
                    "result_label": rl,
                    "row_kind": "completed",
                }
            )
            continue

        bias_mean, reg = fit_final(past)
        resid_stats_wf = analyze_misses_vs_line(
            past, n_boot=BOOTSTRAP_B_WALKFORWARD
        )
        row = build_prediction_row(r, t0, bias_mean, reg, resid_stats_wf)
        row["actual_total"] = round(act, 1)
        rc, rl = grade_vs_line(row["pick_ou"], act, mgm)
        row["result_code"] = rc
        row["result_label"] = rl
        row["row_kind"] = "completed"
        out.append(row)
    return out


def walk_forward_backtest(completed: pd.DataFrame) -> dict:
    """MAE on total score; regression predicts line + residual_hat."""
    n = len(completed)
    bias_errors: list[float] = []
    reg_errors: list[float] = []
    if n <= BURN_IN:
        return {
            "mae_bias": None,
            "mae_regression": None,
            "n_backtest": 0,
            "bias_errors": [],
            "reg_errors": [],
        }

    X_all = build_X(completed)
    y_all = completed["Total"].values
    mgm_all = completed["MGM_OU"].values
    resid_all = y_all - mgm_all

    for i in range(BURN_IN, n):
        train_idx = slice(0, i)
        bias = float(np.mean(resid_all[train_idx]))
        pred_bias = mgm_all[i] + bias

        X_tr = X_all[train_idx]
        y_res_tr = resid_all[train_idx]
        reg = Ridge(alpha=RIDGE_ALPHA, random_state=0)
        reg.fit(X_tr, y_res_tr)
        pred_res = float(reg.predict(X_all[i : i + 1])[0])
        pred_reg = mgm_all[i] + pred_res

        actual = float(y_all[i])
        bias_errors.append(abs(actual - pred_bias))
        reg_errors.append(abs(actual - pred_reg))

    mae_b = float(np.mean(bias_errors)) if bias_errors else None
    mae_r = float(np.mean(reg_errors)) if reg_errors else None
    return {
        "mae_bias": mae_b,
        "mae_regression": mae_r,
        "n_backtest": len(bias_errors),
        "bias_errors": bias_errors,
        "reg_errors": reg_errors,
    }


def recommendation(mae_b: float | None, mae_r: float | None) -> tuple[str, str]:
    if mae_b is None or mae_r is None:
        return "tie", "Insufficient games for walk-forward backtest."
    diff = abs(mae_b - mae_r)
    if diff < MAE_TIE_EPS:
        return "tie", f"Backtest MAE effectively tied (bias {mae_b:.2f} vs regression {mae_r:.2f})."
    if mae_b < mae_r:
        return (
            "bias",
            f"Lower backtest MAE for bias-adjusted line: {mae_b:.2f} vs residual regression {mae_r:.2f}.",
        )
    return (
        "regression",
        f"Lower backtest MAE for residual regression: {mae_r:.2f} vs bias-adjusted {mae_b:.2f}.",
    )


def fit_final(completed: pd.DataFrame) -> tuple[float, Ridge]:
    resid = completed["Total"].values - completed["MGM_OU"].values
    bias_mean = float(np.mean(resid))
    reg = Ridge(alpha=RIDGE_ALPHA, random_state=0)
    reg.fit(build_X(completed), resid)
    return bias_mean, reg


def predict_upcoming(
    upcoming: pd.DataFrame,
    t0: datetime,
    bias_mean: float,
    reg: Ridge,
    resid_stats: dict,
) -> list[dict]:
    rows = []
    for _, r in upcoming.iterrows():
        row = build_prediction_row(r, t0, bias_mean, reg, resid_stats)
        row["actual_total"] = None
        row["result_code"] = "pending"
        row["result_label"] = "—"
        row["row_kind"] = "upcoming"
        rows.append(row)
    return rows


def main() -> None:
    completed, upcoming, t0 = load_and_clean()
    resid_stats = analyze_misses_vs_line(completed)
    back = walk_forward_backtest(completed)
    mae_b, mae_r = back["mae_bias"], back["mae_regression"]
    rec_model, rec_reason = recommendation(mae_b, mae_r)
    if mae_b is not None and mae_r is not None:
        diff = abs(mae_b - mae_r)
        if diff < MAE_TIE_EPS:
            conf = "weak / tie"
        elif diff < 2.0:
            conf = "moderate"
        else:
            conf = "stronger preference"
    else:
        conf = "unknown"

    game_rows = walk_forward_dashboard_rows(completed, t0)
    if len(upcoming) > 0:
        bias_mean, reg = fit_final(completed)
        game_rows.extend(
            predict_upcoming(upcoming, t0, bias_mean, reg, resid_stats)
        )

    acc_attempted = 0
    acc_correct = 0
    for g in game_rows:
        if g.get("pick_ou") not in ("over", "under"):
            continue
        if g.get("row_kind") != "completed":
            continue
        acc_attempted += 1
        if g.get("result_code") == "confirmed":
            acc_correct += 1
    acc_pct = (
        round(100.0 * acc_correct / acc_attempted, 2) if acc_attempted else None
    )

    rec_badge_tie_text = ""
    rec_badge_tie_title = ""
    if rec_model == "tie":
        if mae_b is None or mae_r is None:
            rec_badge_tie_text = "Tie"
            rec_badge_tie_title = (
                "Not enough completed games with valid lines to rank the models in the walk-forward backtest."
            )
        elif round(mae_b, 2) == round(mae_r, 2):
            rec_badge_tie_text = "Tie"
            rec_badge_tie_title = (
                "Both models equally good at predicting total score."
            )
        else:
            rec_badge_tie_text = "Tie / weak"
            rec_badge_tie_title = (
                "No clear winner in the historical horse race,"
            )

    meta = {
        "mae_bias": mae_b,
        "mae_regression": mae_r,
        "n_backtest": back["n_backtest"],
        "recommended_model": rec_model,
        "recommendation_reason": rec_reason,
        "confidence_label": conf,
        "n_completed_train": len(completed),
        "rec_badge_tie_text": rec_badge_tie_text,
        "rec_badge_tie_title": rec_badge_tie_title,
        "model_accuracy_correct": acc_correct,
        "model_accuracy_attempted": acc_attempted,
        "model_accuracy_pct": acc_pct,
        **resid_stats,
    }

    payload = {"meta": meta, "games": game_rows}

    json_str = json.dumps(payload, separators=(",", ":"))
    html = TEMPLATE_PATH.read_text(encoding="utf-8").replace("__PAYLOAD__", json_str)
    OUT_HTML.write_text(html, encoding="utf-8")
    print(
        f"Wrote {OUT_HTML} ({len(completed)} completed, {len(upcoming)} upcoming, "
        f"{len(game_rows)} dashboard rows)."
    )


if __name__ == "__main__":
    main()
