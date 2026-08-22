# AI Diagnostic Summary

- Overall status: **CRITICAL**
- Strongest component: entry
- Weakest component: direction
- Biggest data problem: none measured
- Biggest ML problem: direction is the weakest scored component
- Biggest validation problem: none measured (walk-forward across 1 folds, accuracy std=0.000)
- Biggest trading problem: backtest metrics are not statistically reliable: only 0 trade(s) generated from 3160 candidate signal(s) (3160 rejected) - treat win rate/profit factor/Sharpe as noise, not a performance estimate
- Most important improvement: NOT_AVAILABLE
- Most important degradation: NOT_AVAILABLE
- Recommended next action: Fix before anything else: exit model has no trained artifact

# ML Diagnostic Report

Run `baseline-prefix` generated 2026-08-22T21:15:27+00:00 (git `ebb1fb8c6bcf`)

## Training Overview
- Timeframe: 5m
- Symbols: SYM0/USDT:USDT, SYM1/USDT:USDT, SYM2/USDT:USDT, SYM3/USDT:USDT
- Training period: {'start': 1704067200000, 'end': 1704656700000, 'rows': 7864}
- Validation period: {'start': 1704660000000, 'end': 1704953100000, 'rows': 3912}
- Model versions: {'direction': '2026-08-22T21:14:55+00:00', 'entry': '2026-08-22T21:14:55+00:00', 'exit': 'untrained', 'risk': 'untrained'}

## Dataset Health
```json
{
  "total_candidate_rows": 16000,
  "valid_samples": 15808,
  "rejected_invalid_label_rows": 192,
  "dropped_missing_or_inf_rows": 0,
  "duplicate_feature_rows": 30,
  "training_samples": 7864,
  "validation_samples": 3912,
  "test_samples": 3952,
  "feature_count": 57,
  "feature_names": [
    "kama_distance",
    "kama_slope",
    "kama_slope_fast",
    "ema_fast_slow_spread",
    "close_ema_slow_ratio",
    "adx",
    "di_spread",
    "fdi",
    "fdi_trending",
    "fdi_delta",
    "bb_width",
    "bb_position",
    "rsi",
    "rsi_delta",
    "log_return_1",
    "log_return_3",
    "log_return_12",
    "log_return_48",
    "momentum_rank",
    "atr_pct",
    "atr_rank",
    "realized_vol_12",
    "realized_vol_48",
    "garch_volatility",
    "garch_vol_rank",
    "garch_vol_ratio",
    "vol_of_vol",
    "realized_vol_12_is_zero",
    "wick_ratio",
    "whipsaw_rate",
    "hmm_regime",
    "hmm_prob_bull",
    "hmm_prob_bear",
    "hmm_prob_high_vol",
    "hmm_prob_sideways",
    "hmm_regime_age",
    "volume_zscore",
    "volume_rank",
    "volume_trend",
    "dollar_volume_rank",
    "ob_imbalance",
    "ob_imbalance_delta",
    "ob_spread_bps",
    "ob_spread_rank",
    "liquidation_imbalance",
    "microstructure_is_missing",
    "funding_rate",
    "funding_rate_delta",
    "funding_rate_rank",
    "open_interest_change",
    "open_interest_rank",
    "long_short_ratio",
    "taker_buy_sell_ratio",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos"
  ],
  "per_symbol_rows": {
    "SYM0/USDT:USDT": 3952,
    "SYM1/USDT:USDT": 3952,
    "SYM2/USDT:USDT": 3952,
    "SYM3/USDT:USDT": 3952
  },
  "null_counts_by_feature": {
    "kama_distance": 40,
    "kama_slope": 60,
    "kama_slope_fast": 48,
    "ema_fast_slow_spread": 188,
    "close_ema_slow_ratio": 188,
    "adx": 52,
    "di_spread": 0,
    "fdi": 116,
    "fdi_trending": 0,
    "fdi_delta": 128,
    "bb_width": 0,
    "bb_position": 8800,
    "rsi": 0,
    "rsi_delta": 12,
    "log_return_1": 4,
    "log_return_3": 12,
    "log_return_12": 48,
    "log_return_48": 192,
    "momentum_rank": 108,
    "atr_pct": 56,
    "atr_rank": 108,
    "realized_vol_12": 20,
    "realized_vol_48": 92,
    "garch_volatility": 2000,
    "garch_vol_rank": 108,
    "garch_vol_ratio": 2000,
    "vol_of_vol": 64,
    "realized_vol_12_is_zero": 0,
    "wick_ratio": 0,
    "whipsaw_rate": 0,
    "hmm_regime": 0,
    "hmm_prob_bull": 0,
    "hmm_prob_bear": 0,
    "hmm_prob_high_vol": 0,
    "hmm_prob_sideways": 0,
    "hmm_regime_age": 0,
    "volume_zscore": 0,
    "volume_rank": 108,
    "volume_trend": 8800,
    "dollar_volume_rank": 108,
    "ob_imbalance": 15808,
    "ob_imbalance_delta": 15808,
    "ob_spread_bps": 15808,
    "ob_spread_rank": 15808,
    "liquidation_imbalance": 15808,
    "microstructure_is_missing": 0,
    "funding_rate": 0,
    "funding_rate_delta": 0,
    "funding_rate_rank": 0,
    "open_interest_change": 0,
    "open_interest_rank": 0,
    "long_short_ratio": 0,
    "taker_buy_sell_ratio": 0,
    "hour_sin": 0,
    "hour_cos": 0,
    "dow_sin": 0,
    "dow_cos": 0
  },
  "null_rate_by_symbol_month": {
    "SYM0/USDT:USDT": {
      "2024-01": 1.0
    },
    "SYM1/USDT:USDT": {
      "2024-01": 1.0
    },
    "SYM2/USDT:USDT": {
      "2024-01": 1.0
    },
    "SYM3/USDT:USDT": {
      "2024-01": 1.0
    }
  },
  "split_coverage_pct": {
    "train": {
      "rows": 7864,
      "span_days": 6.8,
      "capacity_rows": 7860,
      "coverage_pct": 1.0
    },
    "validation": {
      "rows": 3912,
      "span_days": 3.4,
      "capacity_rows": 3908,
      "coverage_pct": 1.0
    },
    "test": {
      "rows": 3952,
      "span_days": 3.4,
      "capacity_rows": 3948,
      "coverage_pct": 1.0
    }
  }
}
```

## Data Quality / Healing
```json
{
  "status": "NO_HEALING_NEEDED",
  "heal_attempts_total": 0,
  "heal_attempts_by_result": {},
  "recent_heal_attempts": "[0 entries omitted from Markdown - see JSON export]",
  "symbol_exclusions_total": 0,
  "recent_symbol_exclusions": "[0 entries omitted from Markdown - see JSON export]"
}
```

## Direction
```json
{
  "status": "TRAINED",
  "trained_at": "2026-08-22T21:14:55+00:00",
  "git_commit": "ebb1fb8c6bcf",
  "rows": 7864,
  "validation_rows": 3912,
  "classes": [
    "LONG_SUCCESS",
    "SHORT_SUCCESS",
    "NO_TRADE_OR_FAIL"
  ],
  "distribution": {
    "NO_TRADE_OR_FAIL": 11797,
    "SHORT_SUCCESS": 2137,
    "LONG_SUCCESS": 1874
  },
  "hyperparameters": {
    "booster": "lightgbm",
    "n_estimators": 20,
    "learning_rate": 0.05,
    "max_depth": 6,
    "num_leaves": 63,
    "min_child_samples": 40,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "random_state": 42,
    "train_months": 12.0,
    "validation_months": 6.0,
    "test_months": 6.0,
    "purge_bars": 10,
    "early_stopping_rounds": 5,
    "direction_stage2_hyperparameters": {
      "n_estimators": 700,
      "learning_rate": 0.03,
      "max_depth": 5,
      "num_leaves": 31,
      "subsample": 0.8,
      "colsample_bytree": 0.7,
      "min_child_samples": 100,
      "reg_lambda": 2.0,
      "reg_alpha": 0.5
    },
    "risk_hyperparameters": {
      "n_estimators": 600,
      "learning_rate": 0.04,
      "max_depth": 7,
      "num_leaves": 95,
      "subsample": 0.85,
      "colsample_bytree": 0.85,
      "min_child_samples": 25,
      "reg_lambda": 1.5,
      "reg_alpha": 0.1
    }
  },
  "metrics": {
    "accuracy": 0.5728527607361963,
    "balanced_accuracy": 0.3333333333333333,
    "log_loss": 15.049218887666127,
    "macro_precision": 0.19095092024539875,
    "macro_recall": 0.3333333333333333,
    "macro_f1": 0.24280838615309605,
    "weighted_f1": 0.41728036301310445,
    "per_class": {
      "LONG_SUCCESS": {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "support": 704
      },
      "SHORT_SUCCESS": {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "support": 967
      },
      "NO_TRADE_OR_FAIL": {
        "precision": 0.5728527607361963,
        "recall": 1.0,
        "f1": 0.7284251584592881,
        "support": 2241
      }
    },
    "confusion_matrix": {
      "labels": [
        "LONG_SUCCESS",
        "SHORT_SUCCESS",
        "NO_TRADE_OR_FAIL"
      ],
      "raw": [
        [
          0,
          0,
          704
        ],
        [
          0,
          0,
          967
        ],
        [
          0,
          0,
          2241
        ]
      ],
      "normalized": [
        [
          0.0,
          0.0,
          1.0
        ],
        [
          0.0,
          0.0,
          1.0
        ],
        [
          0.0,
          0.0,
          1.0
        ]
      ]
    },
    "class_distribution": {
      "LONG_SUCCESS": 704,
      "SHORT_SUCCESS": 967,
      "NO_TRADE_OR_FAIL": 2241
    },
    "predicted_class_distribution": {
      "LONG_SUCCESS": 0,
      "SHORT_SUCCESS": 0,
      "NO_TRADE_OR_FAIL": 3912
    },
    "probability_stats": {
      "LONG_SUCCESS": {
        "mean": 5.000000018137454e-16,
        "median": 5.000000018137457e-16,
        "std": 2.9582283945787943e-31,
        "min": 5.000000018137457e-16,
        "max": 5.000000018137457e-16
      },
      "SHORT_SUCCESS": {
        "mean": 5.000000018137454e-16,
        "median": 5.000000018137457e-16,
        "std": 2.9582283945787943e-31,
        "min": 5.000000018137457e-16,
        "max": 5.000000018137457e-16
      },
      "NO_TRADE_OR_FAIL": {
        "mean": 0.9999999999999986,
        "median": 0.999999999999999,
        "std": 4.440892098500626e-16,
        "min": 0.999999999999999,
        "max": 0.999999999999999
      }
    },
    "confidence_threshold_analysis": [
      {
        "confidence_threshold": 0.3,
        "n_predictions": 3912,
        "pct_of_samples": 1.0,
        "accuracy": 0.5728527607361963,
        "balanced_accuracy": 0.3333333333333333,
        "precision": 0.19095092024539875,
        "recall": 0.3333333333333333,
        "f1": 0.24280838615309605
      },
      {
        "confidence_threshold": 0.35,
        "n_predictions": 3912,
        "pct_of_samples": 1.0,
        "accuracy": 0.5728527607361963,
        "balanced_accuracy": 0.3333333333333333,
        "precision": 0.19095092024539875,
        "recall": 0.3333333333333333,
        "f1": 0.24280838615309605
      },
      {
        "confidence_threshold": 0.4,
        "n_predictions": 3912,
        "pct_of_samples": 1.0,
        "accuracy": 0.5728527607361963,
        "balanced_accuracy": 0.3333333333333333,
        "precision": 0.19095092024539875,
        "recall": 0.3333333333333333,
        "f1": 0.24280838615309605
      },
      {
        "confidence_threshold": 0.45,
        "n_predictions": 3912,
        "pct_of_samples": 1.0,
        "accuracy": 0.5728527607361963,
        "balanced_accuracy": 0.3333333333333333,
        "precision": 0.19095092024539875,
        "recall": 0.3333333333333333,
        "f1": 0.24280838615309605
      },
      {
        "confidence_threshold": 0.5,
        "n_predictions": 3912,
        "pct_of_samples": 1.0,
        "accuracy": 0.5728527607361963,
        "balanced_accuracy": 0.3333333333333333,
        "precision": 0.19095092024539875,
        "recall": 0.3333333333333333,
        "f1": 0.24280838615309605
      },
      {
        "confidence_threshold": 0.55,
        "n_predictions": 3912,
        "pct_of_samples": 1.0,
        "accuracy": 0.5728527607361963,
        "balanced_accuracy": 0.3333333333333333,
        "precision": 0.19095092024539875,
        "recall": 0.3333333333333333,
        "f1": 0.24280838615309605
      },
      {
        "confidence_threshold": 0.6,
        "n_predictions": 3912,
        "pct_of_samples": 1.0,
        "accuracy": 0.5728527607361963,
        "balanced_accuracy": 0.3333333333333333,
        "precision": 0.19095092024539875,
        "recall": 0.3333333333333333,
        "f1": 0.24280838615309605
      },
      {
        "confidence_threshold": 0.65,
        "n_predictions": 3912,
        "pct_of_samples": 1.0,
        "accuracy": 0.5728527607361963,
        "balanced_accuracy": 0.3333333333333333,
        "precision": 0.19095092024539875,
        "recall": 0.3333333333333333,
        "f1": 0.24280838615309605
      },
      {
        "confidence_threshold": 0.7,
        "n_predictions": 3912,
        "pct_of_samples": 1.0,
        "accuracy": 0.5728527607361963,
        "balanced_accuracy": 0.3333333333333333,
        "precision": 0.19095092024539875,
        "recall": 0.3333333333333333,
        "f1": 0.24280838615309605
      },
      {
        "confidence_threshold": 0.75,
        "n_predictions": 3912,
        "pct_of_samples": 1.0,
        "accuracy": 0.5728527607361963,
        "balanced_accuracy": 0.3333333333333333,
        "precision": 0.19095092024539875,
        "recall": 0.3333333333333333,
        "f1": 0.24280838615309605
      },
      {
        "confidence_threshold": 0.8,
        "n_predictions": 3912,
        "pct_of_samples": 1.0,
        "accuracy": 0.5728527607361963,
        "balanced_accuracy": 0.3333333333333333,
        "precision": 0.19095092024539875,
        "recall": 0.3333333333333333,
        "f1": 0.24280838615309605
      },
      {
        "confidence_threshold": 0.85,
        "n_predictions": 3912,
        "pct_of_samples": 1.0,
        "accuracy": 0.5728527607361963,
        "balanced_accuracy": 0.3333333333333333,
        "precision": 0.19095092024539875,
        "recall": 0.3333333333333333,
        "f1": 0.24280838615309605
      }
    ]
  },
  "feature_importance": {
    "gate": {
      "status": "AVAILABLE",
      "method": "native_gain_or_split",
      "top_features": [
        {
          "feature": "kama_distance",
          "importance": 0.0,
          "importance_pct": 0.0
        },
        {
          "feature": "kama_slope",
          "importance": 0.0,
          "importance_pct": 0.0
        },
        {
          "feature": "kama_slope_fast",
          "importance": 0.0,
          "importance_pct": 0.0
        },
        {
          "feature": "ema_fast_slow_spread",
          "importance": 0.0,
          "importance_pct": 0.0
        },
        {
          "feature": "close_ema_slow_ratio",
          "importance": 0.0,
          "importance_pct": 0.0
        },
        {
          "feature": "adx",
          "importance": 0.0,
          "importance_pct": 0.0
        },
        {
          "feature": "di_spread",
          "importance": 0.0,
          "importance_pct": 0.0
        },
        {
          "feature": "fdi",
          "importance": 0.0,
          "importance_pct": 0.0
        },
        {
          "feature": "fdi_trending",
          "importance": 0.0,
          "importance_pct": 0.0
        },
        {
          "feature": "fdi_delta",
          "importance": 0.0,
          "importance_pct": 0.0
        },
        {
          "feature": "bb_width",
          "importance": 0.0,
          "importance_pct": 0.0
        },
        {
          "feature": "bb_position",
          "importance": 0.0,
          "importance_pct": 0.0
        },
        {
          "feature": "rsi",
          "importance": 0.0,
          "importance_pct": 0.0
        },
        {
          "feature": "rsi_delta",
          "importance": 0.0,
          "importance_pct": 0.0
        },
        {
          "feature": "log_return_1",
          "importance": 0.0,
          "importance_pct": 0.0
        },
        {
          "feature": "log_return_3",
          "importance": 0.0,
          "importance_pct": 0.0
        },
        {
          "feature": "log_return_12",
          "importance": 0.0,
          "importance_pct": 0.0
        },
        {
          "feature": "log_return_48",
          "importance": 0.0,
          "importance_pct": 0.0
        },
        {
          "feature": "momentum_rank",
          "importance": 0.0,
          "importance_pct": 0.0
        },
        {
          "feature": "atr_pct",
          "importance": 0.0,
          "importance_pct": 0.0
        }
      ],
      "shap": {
        "status": "NOT_AVAILABLE",
        "reason": "shap is not an installed project dependency"
      }
    },
    "direction": {
      "status": "NOT_AVAILABLE",
      "reason": "not enough trade rows to fit stage 2"
    }
  },
  "calibration": {
    "gate": {
      "status": "NOT_AVAILABLE",
      "reason": "calibration failed: Only 1 class/es in training fold, but 2 in overall dataset. This is not supported for decision_function with imbalanced folds. To fix this, use a cross-validation technique resulting in properly stratified folds"
    },
    "direction": {
      "status": "NOT_AVAILABLE",
      "reason": "stage 2 was not fitted (not enough trade rows)"
    },
    "joint": {
      "status": "AVAILABLE",
      "method": "isotonic_per_class",
      "calibration_rows": 1956,
      "eval_rows": 1956,
      "brier_score_raw": 1.2351738241308772,
      "brier_score_calibrated": 0.8848245239857646,
      "log_loss_raw": 21.75877488486018,
      "log_loss_calibrated": 1.4920367128771879,
      "improved": true,
      "recommended_for_production": true,
      "note": "Calibrates the joint long/short/no_trade probability directly, on top of whatever per-stage calibration already happened above - corrects residual miscalibration that the product of two independently-calibrated probabilities can still leave behind, which per-stage calibration alone cannot see. Applied only to the reported `probabilities` dict at inference (R3's consistency check, audit logging); trade_probability and direction_given_trade_probability, which R1a/R1b gate on, are never touched by this."
    }
  },
  "production_calibration": {
    "gate": "raw",
    "direction": "raw"
  },
  "per_symbol": {
    "SYM0/USDT:USDT": {
      "samples": 978,
      "accuracy": 0.5388548057259713,
      "balanced_accuracy": 0.3333333333333333
    },
    "SYM1/USDT:USDT": {
      "samples": 978,
      "accuracy": 0.5531697341513292,
      "balanced_accuracy": 0.3333333333333333
    },
    "SYM2/USDT:USDT": {
      "samples": 978,
      "accuracy": 0.5531697341513292,
      "balanced_accuracy": 0.3333333333333333
    },
    "SYM3/USDT:USDT": {
      "samples": 978,
      "accuracy": 0.6462167689161554,
      "balanced_accuracy": 0.3333333333333333
    }
  },
  "architecture": "two_stage_cascade",
  "gate_threshold_sweep": [
    {
      "threshold": 0.3,
      "signals": 0,
      "precision": 0.0,
      "recall": 0.0,
      "f1": 0.0,
      "meets_min_sample_size": false,
      "average_r": "NOT_AVAILABLE",
      "win_rate": "NOT_AVAILABLE",
      "profit_factor": "NOT_AVAILABLE",
      "expectancy": "NOT_AVAILABLE",
      "net_pnl": "NOT_AVAILABLE",
      "max_drawdown": "NOT_AVAILABLE"
    },
    {
      "threshold": 0.35,
      "signals": 0,
      "precision": 0.0,
      "recall": 0.0,
      "f1": 0.0,
      "meets_min_sample_size": false,
      "average_r": "NOT_AVAILABLE",
      "win_rate": "NOT_AVAILABLE",
      "profit_factor": "NOT_AVAILABLE",
      "expectancy": "NOT_AVAILABLE",
      "net_pnl": "NOT_AVAILABLE",
      "max_drawdown": "NOT_AVAILABLE"
    },
    {
      "threshold": 0.4,
      "signals": 0,
      "precision": 0.0,
      "recall": 0.0,
      "f1": 0.0,
      "meets_min_sample_size": false,
      "average_r": "NOT_AVAILABLE",
      "win_rate": "NOT_AVAILABLE",
      "profit_factor": "NOT_AVAILABLE",
      "expectancy": "NOT_AVAILABLE",
      "net_pnl": "NOT_AVAILABLE",
      "max_drawdown": "NOT_AVAILABLE"
    },
    {
      "threshold": 0.45,
      "signals": 0,
      "precision": 0.0,
      "recall": 0.0,
      "f1": 0.0,
      "meets_min_sample_size": false,
      "average_r": "NOT_AVAILABLE",
      "win_rate": "NOT_AVAILABLE",
      "profit_factor": "NOT_AVAILABLE",
      "expectancy": "NOT_AVAILABLE",
      "net_pnl": "NOT_AVAILABLE",
      "max_drawdown": "NOT_AVAILABLE"
    },
    {
      "threshold": 0.5,
      "signals": 0,
      "precision": 0.0,
      "recall": 0.0,
      "f1": 0.0,
      "meets_min_sample_size": false,
      "average_r": "NOT_AVAILABLE",
      "win_rate": "NOT_AVAILABLE",
      "profit_factor": "NOT_AVAILABLE",
      "expectancy": "NOT_AVAILABLE",
      "net_pnl": "NOT_AVAILABLE",
      "max_drawdown": "NOT_AVAILABLE"
    },
    {
      "threshold": 0.55,
      "signals": 0,
      "precision": 0.0,
      "recall": 0.0,
      "f1": 0.0,
      "meets_min_sample_size": false,
      "average_r": "NOT_AVAILABLE",
      "win_rate": "NOT_AVAILABLE",
      "profit_factor": "NOT_AVAILABLE",
      "expectancy": "NOT_AVAILABLE",
      "net_pnl": "NOT_AVAILABLE",
      "max_drawdown": "NOT_AVAILABLE"
    },
    {
      "threshold": 0.6,
      "signals": 0,
      "precision": 0.0,
      "recall": 0.0,
      "f1": 0.0,
      "meets_min_sample_size": false,
      "average_r": "NOT_AVAILABLE",
      "win_rate": "NOT_AVAILABLE",
      "profit_factor": "NOT_AVAILABLE",
      "expectancy": "NOT_AVAILABLE",
      "net_pnl": "NOT_AVAILABLE",
      "max_drawdown": "NOT_AVAILABLE"
    },
    {
      "threshold": 0.65,
      "signals": 0,
      "precision": 0.0,
      "recall": 0.0,
      "f1": 0.0,
      "meets_min_sample_size": false,
      "average_r": "NOT_AVAILABLE",
      "win_rate": "NOT_AVAILABLE",
      "profit_factor": "NOT_AVAILABLE",
      "expectancy": "NOT_AVAILABLE",
      "net_pnl": "NOT_AVAILABLE",
      "max_drawdown": "NOT_AVAILABLE"
    },
    {
      "threshold": 0.7,
      "signals": 0,
      "precision": 0.0,
      "recall": 0.0,
      "f1": 0.0,
      "meets_min_sample_size": false,
      "average_r": "NOT_AVAILABLE",
      "win_rate": "NOT_AVAILABLE",
      "profit_factor": "NOT_AVAILABLE",
      "expectancy": "NOT_AVAILABLE",
      "net_pnl": "NOT_AVAILABLE",
      "max_drawdown": "NOT_AVAILABLE"
    },
    {
      "threshold": 0.75,
      "signals": 0,
      "precision": 0.0,
      "recall": 0.0,
      "f1": 0.0,
      "meets_min_sample_size": false,
      "average_r": "NOT_AVAILABLE",
      "win_rate": "NOT_AVAILABLE",
      "profit_factor": "NOT_AVAILABLE",
      "expectancy": "NOT_AVAILABLE",
      "net_pnl": "NOT_AVAILABLE",
      "max_drawdown": "NOT_AVAILABLE"
    },
    {
      "threshold": 0.8,
      "signals": 0,
      "precision": 0.0,
      "recall": 0.0,
      "f1": 0.0,
      "meets_min_sample_size": false,
      "average_r": "NOT_AVAILABLE",
      "win_rate": "NOT_AVAILABLE",
      "profit_factor": "NOT_AVAILABLE",
      "expectancy": "NOT_AVAILABLE",
      "net_pnl": "NOT_AVAILABLE",
      "max_drawdown": "NOT_AVAILABLE"
    },
    {
      "threshold": 0.85,
      "signals": 0,
      "precision": 0.0,
      "recall": 0.0,
      "f1": 0.0,
      "meets_min_sample_size": false,
      "average_r": "NOT_AVAILABLE",
      "win_rate": "NOT_AVAILABLE",
      "profit_factor": "NOT_AVAILABLE",
      "expectancy": "NOT_AVAILABLE",
      "net_pnl": "NOT_AVAILABLE",
      "max_drawdown": "NOT_AVAILABLE"
    }
  ],
  "recommended_gate_threshold": 0.55,
  "direction_threshold_sweep": [],
  "recommended_direction_threshold": 0.6,
  "split": {
    "train": {
      "start": "2024-01-01T00:00:00+00:00",
      "end": "2024-01-07T19:45:00+00:00",
      "rows": 7864
    },
    "validation": {
      "start": "2024-01-07T20:40:00+00:00",
      "end": "2024-01-11T06:05:00+00:00",
      "rows": 3912
    },
    "test": {
      "start": "2024-01-11T07:00:00+00:00",
      "end": "2024-01-14T17:15:00+00:00",
      "rows": 3952
    },
    "embargo_ms": 3000000,
    "scaled_down_from_nominal_months": true
  }
}
```

## Entry
```json
{
  "status": "TRAINED",
  "trained_at": "2026-08-22T21:14:55+00:00",
  "git_commit": "ebb1fb8c6bcf",
  "rows": 7864,
  "validation_rows": 3912,
  "positive_rate": 0.17307692307692307,
  "decision_threshold": 0.55,
  "configured_floor_threshold": 0.55,
  "hyperparameters": {
    "booster": "lightgbm",
    "n_estimators": 20,
    "learning_rate": 0.05,
    "max_depth": 6,
    "num_leaves": 63,
    "min_child_samples": 40,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "random_state": 42,
    "train_months": 12.0,
    "validation_months": 6.0,
    "test_months": 6.0,
    "purge_bars": 10,
    "early_stopping_rounds": 5,
    "direction_stage2_hyperparameters": {
      "n_estimators": 700,
      "learning_rate": 0.03,
      "max_depth": 5,
      "num_leaves": 31,
      "subsample": 0.8,
      "colsample_bytree": 0.7,
      "min_child_samples": 100,
      "reg_lambda": 2.0,
      "reg_alpha": 0.5
    },
    "risk_hyperparameters": {
      "n_estimators": 600,
      "learning_rate": 0.04,
      "max_depth": 7,
      "num_leaves": 95,
      "subsample": 0.85,
      "colsample_bytree": 0.85,
      "min_child_samples": 25,
      "reg_lambda": 1.5,
      "reg_alpha": 0.1
    }
  },
  "metrics": {
    "threshold": 0.55,
    "accuracy": 0.7024539877300614,
    "precision": 0.0,
    "recall": 0.0,
    "f1": 0.0,
    "roc_auc": 0.5,
    "pr_auc": 0.29754601226993865,
    "confusion_matrix": {
      "labels": [
        0,
        1
      ],
      "raw": [
        [
          2748,
          0
        ],
        [
          1164,
          0
        ]
      ]
    },
    "class_distribution": {
      "positive": 1164,
      "negative": 2748
    },
    "predicted_positive_rate": 0.0,
    "probability_stats": {
      "mean": 1.0000000036274908e-15,
      "median": 1.0000000036274914e-15,
      "std": 5.9164567891575885e-31,
      "min": 1.0000000036274914e-15,
      "max": 1.0000000036274914e-15
    }
  },
  "threshold_sweep": [
    {
      "threshold": 0.3,
      "signals": 0,
      "precision": 0.0,
      "recall": 0.0,
      "f1": 0.0,
      "meets_min_sample_size": false,
      "average_r": "NOT_AVAILABLE",
      "win_rate": "NOT_AVAILABLE",
      "profit_factor": "NOT_AVAILABLE",
      "expectancy": "NOT_AVAILABLE",
      "net_pnl": "NOT_AVAILABLE",
      "max_drawdown": "NOT_AVAILABLE"
    },
    {
      "threshold": 0.35,
      "signals": 0,
      "precision": 0.0,
      "recall": 0.0,
      "f1": 0.0,
      "meets_min_sample_size": false,
      "average_r": "NOT_AVAILABLE",
      "win_rate": "NOT_AVAILABLE",
      "profit_factor": "NOT_AVAILABLE",
      "expectancy": "NOT_AVAILABLE",
      "net_pnl": "NOT_AVAILABLE",
      "max_drawdown": "NOT_AVAILABLE"
    },
    {
      "threshold": 0.4,
      "signals": 0,
      "precision": 0.0,
      "recall": 0.0,
      "f1": 0.0,
      "meets_min_sample_size": false,
      "average_r": "NOT_AVAILABLE",
      "win_rate": "NOT_AVAILABLE",
      "profit_factor": "NOT_AVAILABLE",
      "expectancy": "NOT_AVAILABLE",
      "net_pnl": "NOT_AVAILABLE",
      "max_drawdown": "NOT_AVAILABLE"
    },
    {
      "threshold": 0.45,
      "signals": 0,
      "precision": 0.0,
      "recall": 0.0,
      "f1": 0.0,
      "meets_min_sample_size": false,
      "average_r": "NOT_AVAILABLE",
      "win_rate": "NOT_AVAILABLE",
      "profit_factor": "NOT_AVAILABLE",
      "expectancy": "NOT_AVAILABLE",
      "net_pnl": "NOT_AVAILABLE",
      "max_drawdown": "NOT_AVAILABLE"
    },
    {
      "threshold": 0.5,
      "signals": 0,
      "precision": 0.0,
      "recall": 0.0,
      "f1": 0.0,
      "meets_min_sample_size": false,
      "average_r": "NOT_AVAILABLE",
      "win_rate": "NOT_AVAILABLE",
      "profit_factor": "NOT_AVAILABLE",
      "expectancy": "NOT_AVAILABLE",
      "net_pnl": "NOT_AVAILABLE",
      "max_drawdown": "NOT_AVAILABLE"
    },
    {
      "threshold": 0.55,
      "signals": 0,
      "precision": 0.0,
      "recall": 0.0,
      "f1": 0.0,
      "meets_min_sample_size": false,
      "average_r": "NOT_AVAILABLE",
      "win_rate": "NOT_AVAILABLE",
      "profit_factor": "NOT_AVAILABLE",
      "expectancy": "NOT_AVAILABLE",
      "net_pnl": "NOT_AVAILABLE",
      "max_drawdown": "NOT_AVAILABLE"
    },
    {
      "threshold": 0.6,
      "signals": 0,
      "precision": 0.0,
      "recall": 0.0,
      "f1": 0.0,
      "meets_min_sample_size": false,
      "average_r": "NOT_AVAILABLE",
      "win_rate": "NOT_AVAILABLE",
      "profit_factor": "NOT_AVAILABLE",
      "expectancy": "NOT_AVAILABLE",
      "net_pnl": "NOT_AVAILABLE",
      "max_drawdown": "NOT_AVAILABLE"
    },
    {
      "threshold": 0.65,
      "signals": 0,
      "precision": 0.0,
      "recall": 0.0,
      "f1": 0.0,
      "meets_min_sample_size": false,
      "average_r": "NOT_AVAILABLE",
      "win_rate": "NOT_AVAILABLE",
      "profit_factor": "NOT_AVAILABLE",
      "expectancy": "NOT_AVAILABLE",
      "net_pnl": "NOT_AVAILABLE",
      "max_drawdown": "NOT_AVAILABLE"
    },
    {
      "threshold": 0.7,
      "signals": 0,
      "precision": 0.0,
      "recall": 0.0,
      "f1": 0.0,
      "meets_min_sample_size": false,
      "average_r": "NOT_AVAILABLE",
      "win_rate": "NOT_AVAILABLE",
      "profit_factor": "NOT_AVAILABLE",
      "expectancy": "NOT_AVAILABLE",
      "net_pnl": "NOT_AVAILABLE",
      "max_drawdown": "NOT_AVAILABLE"
    },
    {
      "threshold": 0.75,
      "signals": 0,
      "precision": 0.0,
      "recall": 0.0,
      "f1": 0.0,
      "meets_min_sample_size": false,
      "average_r": "NOT_AVAILABLE",
      "win_rate": "NOT_AVAILABLE",
      "profit_factor": "NOT_AVAILABLE",
      "expectancy": "NOT_AVAILABLE",
      "net_pnl": "NOT_AVAILABLE",
      "max_drawdown": "NOT_AVAILABLE"
    },
    {
      "threshold": 0.8,
      "signals": 0,
      "precision": 0.0,
      "recall": 0.0,
      "f1": 0.0,
      "meets_min_sample_size": false,
      "average_r": "NOT_AVAILABLE",
      "win_rate": "NOT_AVAILABLE",
      "profit_factor": "NOT_AVAILABLE",
      "expectancy": "NOT_AVAILABLE",
      "net_pnl": "NOT_AVAILABLE",
      "max_drawdown": "NOT_AVAILABLE"
    },
    {
      "threshold": 0.85,
      "signals": 0,
      "precision": 0.0,
      "recall": 0.0,
      "f1": 0.0,
      "meets_min_sample_size": false,
      "average_r": "NOT_AVAILABLE",
      "win_rate": "NOT_AVAILABLE",
      "profit_factor": "NOT_AVAILABLE",
      "expectancy": "NOT_AVAILABLE",
      "net_pnl": "NOT_AVAILABLE",
      "max_drawdown": "NOT_AVAILABLE"
    },
    {
      "threshold": 0.9,
      "signals": 0,
      "precision": 0.0,
      "recall": 0.0,
      "f1": 0.0,
      "meets_min_sample_size": false,
      "average_r": "NOT_AVAILABLE",
      "win_rate": "NOT_AVAILABLE",
      "profit_factor": "NOT_AVAILABLE",
      "expectancy": "NOT_AVAILABLE",
      "net_pnl": "NOT_AVAILABLE",
      "max_drawdown": "NOT_AVAILABLE"
    }
  ],
  "feature_importance": {
    "status": "AVAILABLE",
    "method": "native_gain_or_split",
    "top_features": [
      {
        "feature": "kama_distance",
        "importance": 0.0,
        "importance_pct": 0.0
      },
      {
        "feature": "kama_slope",
        "importance": 0.0,
        "importance_pct": 0.0
      },
      {
        "feature": "kama_slope_fast",
        "importance": 0.0,
        "importance_pct": 0.0
      },
      {
        "feature": "ema_fast_slow_spread",
        "importance": 0.0,
        "importance_pct": 0.0
      },
      {
        "feature": "close_ema_slow_ratio",
        "importance": 0.0,
        "importance_pct": 0.0
      },
      {
        "feature": "adx",
        "importance": 0.0,
        "importance_pct": 0.0
      },
      {
        "feature": "di_spread",
        "importance": 0.0,
        "importance_pct": 0.0
      },
      {
        "feature": "fdi",
        "importance": 0.0,
        "importance_pct": 0.0
      },
      {
        "feature": "fdi_trending",
        "importance": 0.0,
        "importance_pct": 0.0
      },
      {
        "feature": "fdi_delta",
        "importance": 0.0,
        "importance_pct": 0.0
      },
      {
        "feature": "bb_width",
        "importance": 0.0,
        "importance_pct": 0.0
      },
      {
        "feature": "bb_position",
        "importance": 0.0,
        "importance_pct": 0.0
      },
      {
        "feature": "rsi",
        "importance": 0.0,
        "importance_pct": 0.0
      },
      {
        "feature": "rsi_delta",
        "importance": 0.0,
        "importance_pct": 0.0
      },
      {
        "feature": "log_return_1",
        "importance": 0.0,
        "importance_pct": 0.0
      },
      {
        "feature": "log_return_3",
        "importance": 0.0,
        "importance_pct": 0.0
      },
      {
        "feature": "log_return_12",
        "importance": 0.0,
        "importance_pct": 0.0
      },
      {
        "feature": "log_return_48",
        "importance": 0.0,
        "importance_pct": 0.0
      },
      {
        "feature": "momentum_rank",
        "importance": 0.0,
        "importance_pct": 0.0
      },
      {
        "feature": "atr_pct",
        "importance": 0.0,
        "importance_pct": 0.0
      }
    ],
    "shap": {
      "status": "NOT_AVAILABLE",
      "reason": "shap is not an installed project dependency"
    }
  },
  "calibration": {
    "status": "NOT_AVAILABLE",
    "reason": "calibration failed: Only 1 class/es in training fold, but 2 in overall dataset. This is not supported for decision_function with imbalanced folds. To fix this, use a cross-validation technique resulting in properly stratified folds"
  },
  "production_calibration": "raw",
  "split": {
    "train": {
      "start": "2024-01-01T00:00:00+00:00",
      "end": "2024-01-07T19:45:00+00:00",
      "rows": 7864
    },
    "validation": {
      "start": "2024-01-07T20:40:00+00:00",
      "end": "2024-01-11T06:05:00+00:00",
      "rows": 3912
    },
    "test": {
      "start": "2024-01-11T07:00:00+00:00",
      "end": "2024-01-14T17:15:00+00:00",
      "rows": 3952
    },
    "embargo_ms": 3000000,
    "scaled_down_from_nominal_months": true
  }
}
```

## Exit
```json
{
  "status": "NOT_TRAINED"
}
```

## Risk
```json
{
  "status": "NOT_TRAINED"
}
```

## Labels
```json
{
  "configuration": {
    "tp_atr_multiple": 2.0,
    "sl_atr_multiple": 1.0,
    "max_holding_bars": 48,
    "low_risk_mae_ratio": 0.35,
    "medium_risk_mae_ratio": 0.6,
    "high_risk_mae_ratio": 0.85,
    "discard_very_high_risk": true
  },
  "class_distribution": {
    "NO_TRADE_OR_FAIL": 11797,
    "SHORT_SUCCESS": 2137,
    "LONG_SUCCESS": 1874
  }
}
```

## Walk-Forward
{'status': 'AVAILABLE', 'method': 'expanding_window', 'n_folds': 1, 'folds': [{'fold': 4, 'train_rows': 9444, 'validation_rows': 2372, 'accuracy': 0.40345699831365933, 'balanced_accuracy': 0.31601555381607943, 'log_loss': 1.735842063096927, 'macro_f1': 0.21780040101404732}], 'accuracy_mean': 0.40345699831365933, 'accuracy_std': 0.0, 'balanced_accuracy_mean': 0.31601555381607943, 'balanced_accuracy_std': 0.0, 'note': 'Each fold fits an independent two-stage cascade (not the production model) purely to measure how much accuracy varies across different time periods.'}

## Backtest
```json
{
  "start": "2024-01-12T03:30:00+00:00",
  "end": "2024-01-14T21:15:00+00:00",
  "initial_equity": 1000.0,
  "final_equity": 1000.0,
  "symbols": [
    "SYM0/USDT:USDT",
    "SYM1/USDT:USDT",
    "SYM2/USDT:USDT",
    "SYM3/USDT:USDT"
  ],
  "metrics": {
    "total_trades": 0.0,
    "winning_trades": 0.0,
    "losing_trades": 0.0,
    "win_rate": 0.0,
    "profit_factor": 0.0,
    "expectancy": 0.0,
    "average_win": 0.0,
    "average_loss": 0.0,
    "largest_win": 0.0,
    "largest_loss": 0.0,
    "gross_profit": 0.0,
    "gross_loss": 0.0,
    "net_profit": 0.0,
    "total_return_pct": 0.0,
    "annualised_return_pct": 0.0,
    "max_drawdown_pct": 0.0,
    "sharpe_ratio": 0.0,
    "sortino_ratio": 0.0,
    "calmar_ratio": 0.0,
    "total_fees": 0.0,
    "total_funding": 0.0,
    "liquidations": 0.0
  },
  "signals_generated": 3160,
  "signals_rejected": 3160,
  "rejection_breakdown": {
    "R1A_GATE_CONFIDENCE_TOO_LOW": 3160
  },
  "oos_disclosure": null,
  "trades": "[0 entries omitted from Markdown - see JSON export]",
  "equity_curve": "[790 entries omitted from Markdown - see JSON export]"
}
```

## Backtest Reliability
```json
{
  "status": "AVAILABLE",
  "total_trades": 0,
  "minimum_trades_for_reliability": 30,
  "statistically_reliable": false,
  "signals_generated": 3160,
  "signals_rejected": 3160,
  "rejection_breakdown": {
    "R1A_GATE_CONFIDENCE_TOO_LOW": 3160
  }
}
```

## Backtest (Diagnostic, Relaxed Thresholds)
```json
{
  "status": "NOT_AVAILABLE"
}
```

## Symbols
```json
{
  "per_symbol_rows": {
    "SYM0/USDT:USDT": 3952,
    "SYM1/USDT:USDT": 3952,
    "SYM2/USDT:USDT": 3952,
    "SYM3/USDT:USDT": 3952
  },
  "per_symbol_direction_accuracy": {
    "SYM0/USDT:USDT": {
      "samples": 978,
      "accuracy": 0.5388548057259713,
      "balanced_accuracy": 0.3333333333333333
    },
    "SYM1/USDT:USDT": {
      "samples": 978,
      "accuracy": 0.5531697341513292,
      "balanced_accuracy": 0.3333333333333333
    },
    "SYM2/USDT:USDT": {
      "samples": 978,
      "accuracy": 0.5531697341513292,
      "balanced_accuracy": 0.3333333333333333
    },
    "SYM3/USDT:USDT": {
      "samples": 978,
      "accuracy": 0.6462167689161554,
      "balanced_accuracy": 0.3333333333333333
    }
  },
  "note": "Entry/Exit/Risk per-symbol performance is not broken out separately in this run; only Direction is scored per-symbol."
}
```

## Pipeline Timing
```json
{
  "status": "NOT_AVAILABLE",
  "reason": "no completed trading cycle yet"
}
```

## Before vs After (previous accepted baseline)
No previous baseline report to compare against.

## Warnings / Errors
Warnings: []
Errors: []

## Recommendations
```json
{
  "CRITICAL": [
    "Train the exit model - no artifact is currently available",
    "Train the risk model - no artifact is currently available"
  ],
  "HIGH": [
    "Backtest produced only 0 trade(s) from 3160 candidate signal(s) (3160 rejected, top rejection reason: R1A_GATE_CONFIDENCE_TOO_LOW) - win rate/profit factor/Sharpe/expectancy are not statistically meaningful below 30 trades; widen the validation window or run a walk-forward-style backtest across multiple periods before trusting these numbers",
    "Microstructure/derivatives feed(s) sit at their neutral default for nearly every row (likely not being collected): funding_rate, long_short_ratio, open_interest_change, taker_buy_sell_ratio - check data collection for these sources before trusting Entry/Direction feature importance that involves them"
  ],
  "MEDIUM": [],
  "LOW": []
}
```