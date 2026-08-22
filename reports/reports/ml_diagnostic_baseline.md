# AI Diagnostic Summary

- Overall status: **WARNING**
- Strongest component: entry
- Weakest component: risk
- Biggest data problem: none measured
- Biggest ML problem: risk is the weakest scored component
- Biggest validation problem: none measured (walk-forward across 3 folds, accuracy std=0.038)
- Biggest trading problem: backtest metrics are not statistically reliable: only 0 trade(s) generated from 21160 candidate signal(s) (21160 rejected) - treat win rate/profit factor/Sharpe as noise, not a performance estimate
- Most important improvement: NOT_AVAILABLE
- Most important degradation: NOT_AVAILABLE
- Recommended next action: Improve the risk model - it is the weakest measured component

# ML Diagnostic Report

Run `baseline` generated 2026-08-22T21:21:46+00:00 (git `ebb1fb8c6bcf`)

## Training Overview
- Timeframe: 5m
- Symbols: SYM0/USDT:USDT, SYM1/USDT:USDT, SYM2/USDT:USDT, SYM3/USDT:USDT
- Training period: {'start': 1704067200000, 'end': 1705406700000, 'rows': 17864}
- Validation period: {'start': 1705410000000, 'end': 1706078100000, 'rows': 8912}
- Model versions: {'direction': '2026-08-22T21:16:51+00:00', 'entry': '2026-08-22T21:16:52+00:00', 'exit': '2026-08-22T21:16:52+00:00', 'risk': '2026-08-22T21:16:52+00:00'}

## Dataset Health
```json
{
  "total_candidate_rows": 36000,
  "valid_samples": 35808,
  "rejected_invalid_label_rows": 192,
  "dropped_missing_or_inf_rows": 0,
  "duplicate_feature_rows": 5410,
  "training_samples": 17864,
  "validation_samples": 8912,
  "test_samples": 8952,
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
    "SYM0/USDT:USDT": 8952,
    "SYM1/USDT:USDT": 8952,
    "SYM2/USDT:USDT": 8952,
    "SYM3/USDT:USDT": 8952
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
    "bb_position": 10800,
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
    "volume_trend": 10800,
    "dollar_volume_rank": 108,
    "ob_imbalance": 35808,
    "ob_imbalance_delta": 35808,
    "ob_spread_bps": 35808,
    "ob_spread_rank": 35808,
    "liquidation_imbalance": 35808,
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
      "2024-01": 1.0,
      "2024-02": 1.0
    },
    "SYM1/USDT:USDT": {
      "2024-01": 1.0,
      "2024-02": 1.0
    },
    "SYM2/USDT:USDT": {
      "2024-01": 1.0,
      "2024-02": 1.0
    },
    "SYM3/USDT:USDT": {
      "2024-01": 1.0,
      "2024-02": 1.0
    }
  },
  "split_coverage_pct": {
    "train": {
      "rows": 17864,
      "span_days": 15.5,
      "capacity_rows": 17860,
      "coverage_pct": 1.0
    },
    "validation": {
      "rows": 8912,
      "span_days": 7.7,
      "capacity_rows": 8908,
      "coverage_pct": 1.0
    },
    "test": {
      "rows": 8952,
      "span_days": 7.8,
      "capacity_rows": 8948,
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
  "trained_at": "2026-08-22T21:16:51+00:00",
  "git_commit": "ebb1fb8c6bcf",
  "rows": 17864,
  "validation_rows": 8912,
  "classes": [
    "LONG_SUCCESS",
    "SHORT_SUCCESS",
    "NO_TRADE_OR_FAIL"
  ],
  "distribution": {
    "NO_TRADE_OR_FAIL": 21322,
    "SHORT_SUCCESS": 7313,
    "LONG_SUCCESS": 7173
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
    "accuracy": 0.391270197486535,
    "balanced_accuracy": 0.4000309165524532,
    "log_loss": 1.0368391079194963,
    "macro_precision": 0.43110563903107807,
    "macro_recall": 0.4000309165524532,
    "macro_f1": 0.38807624020633064,
    "weighted_f1": 0.3931498988231236,
    "per_class": {
      "LONG_SUCCESS": {
        "precision": 0.32700672202451564,
        "recall": 0.3100862392200975,
        "f1": 0.3183217859892225,
        "support": 2667
      },
      "SHORT_SUCCESS": {
        "precision": 0.3232233783484614,
        "recall": 0.5602455871066769,
        "f1": 0.40993963217745333,
        "support": 2606
      },
      "NO_TRADE_OR_FAIL": {
        "precision": 0.6430868167202572,
        "recall": 0.3297609233305853,
        "f1": 0.4359673024523161,
        "support": 3639
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
          827,
          1527,
          313
        ],
        [
          793,
          1460,
          353
        ],
        [
          909,
          1530,
          1200
        ]
      ],
      "normalized": [
        [
          0.3100862392200975,
          0.5725534308211474,
          0.11736032995875516
        ],
        [
          0.30429777436684574,
          0.5602455871066769,
          0.13545663852647735
        ],
        [
          0.24979389942291838,
          0.42044517724649627,
          0.3297609233305853
        ]
      ]
    },
    "class_distribution": {
      "LONG_SUCCESS": 2667,
      "SHORT_SUCCESS": 2606,
      "NO_TRADE_OR_FAIL": 3639
    },
    "predicted_class_distribution": {
      "LONG_SUCCESS": 2529,
      "SHORT_SUCCESS": 4517,
      "NO_TRADE_OR_FAIL": 1866
    },
    "probability_stats": {
      "LONG_SUCCESS": {
        "mean": 0.324611457967263,
        "median": 0.34260806546893524,
        "std": 0.053016965835227935,
        "min": 0.13440425826975153,
        "max": 0.362611575277149
      },
      "SHORT_SUCCESS": {
        "mean": 0.32577770059564065,
        "median": 0.34363925682556196,
        "std": 0.053213899861484414,
        "min": 0.13439964773014712,
        "max": 0.3629996648355682
      },
      "NO_TRADE_OR_FAIL": {
        "mean": 0.3496108414370964,
        "median": 0.31223091489579374,
        "std": 0.10612342935817556,
        "min": 0.27960799715077533,
        "max": 0.7290827404287805
      }
    },
    "confidence_threshold_analysis": [
      {
        "confidence_threshold": 0.3,
        "n_predictions": 8912,
        "pct_of_samples": 1.0,
        "accuracy": 0.391270197486535,
        "balanced_accuracy": 0.4000309165524532,
        "precision": 0.43110563903107807,
        "recall": 0.4000309165524532,
        "f1": 0.38807624020633064
      },
      {
        "confidence_threshold": 0.35,
        "n_predictions": 2981,
        "pct_of_samples": 0.3344928186714542,
        "accuracy": 0.5109023817510903,
        "balanced_accuracy": 0.4668614490532299,
        "precision": 0.4714244081840098,
        "recall": 0.4668614490532299,
        "f1": 0.45899569421094827
      },
      {
        "confidence_threshold": 0.4,
        "n_predictions": 828,
        "pct_of_samples": 0.09290843806104129,
        "accuracy": 0.9722222222222222,
        "balanced_accuracy": 0.3333333333333333,
        "precision": 0.32407407407407407,
        "recall": 0.3333333333333333,
        "f1": 0.3286384976525822
      },
      {
        "confidence_threshold": 0.45,
        "n_predictions": 801,
        "pct_of_samples": 0.08987881508078994,
        "accuracy": 0.9887640449438202,
        "balanced_accuracy": 0.3333333333333333,
        "precision": 0.3295880149812734,
        "recall": 0.3333333333333333,
        "f1": 0.3314500941619586
      },
      {
        "confidence_threshold": 0.5,
        "n_predictions": 798,
        "pct_of_samples": 0.08954219030520646,
        "accuracy": 0.9912280701754386,
        "balanced_accuracy": 0.3333333333333333,
        "precision": 0.3304093567251462,
        "recall": 0.3333333333333333,
        "f1": 0.3318649045521292
      },
      {
        "confidence_threshold": 0.55,
        "n_predictions": 790,
        "pct_of_samples": 0.08864452423698384,
        "accuracy": 1.0,
        "balanced_accuracy": 1.0,
        "precision": 0.3333333333333333,
        "recall": 0.3333333333333333,
        "f1": 0.3333333333333333
      },
      {
        "confidence_threshold": 0.6,
        "n_predictions": 790,
        "pct_of_samples": 0.08864452423698384,
        "accuracy": 1.0,
        "balanced_accuracy": 1.0,
        "precision": 0.3333333333333333,
        "recall": 0.3333333333333333,
        "f1": 0.3333333333333333
      },
      {
        "confidence_threshold": 0.65,
        "n_predictions": 790,
        "pct_of_samples": 0.08864452423698384,
        "accuracy": 1.0,
        "balanced_accuracy": 1.0,
        "precision": 0.3333333333333333,
        "recall": 0.3333333333333333,
        "f1": 0.3333333333333333
      },
      {
        "confidence_threshold": 0.7,
        "n_predictions": 104,
        "pct_of_samples": 0.011669658886894075,
        "accuracy": 1.0,
        "balanced_accuracy": 1.0,
        "precision": 0.3333333333333333,
        "recall": 0.3333333333333333,
        "f1": 0.3333333333333333
      },
      {
        "confidence_threshold": 0.75,
        "n_predictions": 0,
        "pct_of_samples": 0.0,
        "accuracy": null,
        "balanced_accuracy": null,
        "precision": null,
        "recall": null,
        "f1": null
      },
      {
        "confidence_threshold": 0.8,
        "n_predictions": 0,
        "pct_of_samples": 0.0,
        "accuracy": null,
        "balanced_accuracy": null,
        "precision": null,
        "recall": null,
        "f1": null
      },
      {
        "confidence_threshold": 0.85,
        "n_predictions": 0,
        "pct_of_samples": 0.0,
        "accuracy": null,
        "balanced_accuracy": null,
        "precision": null,
        "recall": null,
        "f1": null
      }
    ]
  },
  "feature_importance": {
    "gate": {
      "status": "AVAILABLE",
      "method": "native_gain_or_split",
      "top_features": [
        {
          "feature": "hour_cos",
          "importance": 25.0,
          "importance_pct": 0.10460251046025104
        },
        {
          "feature": "garch_vol_rank",
          "importance": 24.0,
          "importance_pct": 0.100418410041841
        },
        {
          "feature": "log_return_48",
          "importance": 17.0,
          "importance_pct": 0.07112970711297072
        },
        {
          "feature": "dow_sin",
          "importance": 16.0,
          "importance_pct": 0.06694560669456066
        },
        {
          "feature": "hour_sin",
          "importance": 14.0,
          "importance_pct": 0.058577405857740586
        },
        {
          "feature": "kama_distance",
          "importance": 12.0,
          "importance_pct": 0.0502092050209205
        },
        {
          "feature": "garch_vol_ratio",
          "importance": 12.0,
          "importance_pct": 0.0502092050209205
        },
        {
          "feature": "hmm_prob_bear",
          "importance": 12.0,
          "importance_pct": 0.0502092050209205
        },
        {
          "feature": "garch_volatility",
          "importance": 9.0,
          "importance_pct": 0.03765690376569038
        },
        {
          "feature": "adx",
          "importance": 8.0,
          "importance_pct": 0.03347280334728033
        },
        {
          "feature": "hmm_prob_bull",
          "importance": 7.0,
          "importance_pct": 0.029288702928870293
        },
        {
          "feature": "volume_trend",
          "importance": 7.0,
          "importance_pct": 0.029288702928870293
        },
        {
          "feature": "close_ema_slow_ratio",
          "importance": 6.0,
          "importance_pct": 0.02510460251046025
        },
        {
          "feature": "rsi",
          "importance": 5.0,
          "importance_pct": 0.02092050209205021
        },
        {
          "feature": "atr_rank",
          "importance": 5.0,
          "importance_pct": 0.02092050209205021
        },
        {
          "feature": "realized_vol_48",
          "importance": 5.0,
          "importance_pct": 0.02092050209205021
        },
        {
          "feature": "wick_ratio",
          "importance": 5.0,
          "importance_pct": 0.02092050209205021
        },
        {
          "feature": "whipsaw_rate",
          "importance": 5.0,
          "importance_pct": 0.02092050209205021
        },
        {
          "feature": "bb_position",
          "importance": 4.0,
          "importance_pct": 0.016736401673640166
        },
        {
          "feature": "realized_vol_12",
          "importance": 4.0,
          "importance_pct": 0.016736401673640166
        }
      ],
      "shap": {
        "status": "NOT_AVAILABLE",
        "reason": "shap is not an installed project dependency"
      }
    },
    "direction": {
      "status": "AVAILABLE",
      "method": "native_gain_or_split",
      "top_features": [
        {
          "feature": "fdi_delta",
          "importance": 2.0,
          "importance_pct": 0.15384615384615385
        },
        {
          "feature": "hmm_prob_bull",
          "importance": 2.0,
          "importance_pct": 0.15384615384615385
        },
        {
          "feature": "kama_distance",
          "importance": 1.0,
          "importance_pct": 0.07692307692307693
        },
        {
          "feature": "ema_fast_slow_spread",
          "importance": 1.0,
          "importance_pct": 0.07692307692307693
        },
        {
          "feature": "di_spread",
          "importance": 1.0,
          "importance_pct": 0.07692307692307693
        },
        {
          "feature": "fdi",
          "importance": 1.0,
          "importance_pct": 0.07692307692307693
        },
        {
          "feature": "realized_vol_12",
          "importance": 1.0,
          "importance_pct": 0.07692307692307693
        },
        {
          "feature": "garch_vol_ratio",
          "importance": 1.0,
          "importance_pct": 0.07692307692307693
        },
        {
          "feature": "hmm_prob_bear",
          "importance": 1.0,
          "importance_pct": 0.07692307692307693
        },
        {
          "feature": "hmm_regime_age",
          "importance": 1.0,
          "importance_pct": 0.07692307692307693
        },
        {
          "feature": "volume_trend",
          "importance": 1.0,
          "importance_pct": 0.07692307692307693
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
          "feature": "fdi_trending",
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
        }
      ],
      "shap": {
        "status": "NOT_AVAILABLE",
        "reason": "shap is not an installed project dependency"
      }
    }
  },
  "calibration": {
    "gate": {
      "status": "AVAILABLE",
      "method": "isotonic",
      "calibration_rows": 4456,
      "eval_rows": 4456,
      "brier_score_raw": 0.4396871657950236,
      "brier_score_calibrated": 0.4232105404518632,
      "log_loss_raw": 0.6316265947658348,
      "log_loss_calibrated": 0.7055749637402543,
      "improved": false,
      "recommended_for_production": false,
      "note": "Measured here; the caller (DirectionModel/EntryModel) swaps this stage onto the isotonic-calibrated estimator for live inference whenever `improved` is True, and leaves it on the raw estimator otherwise - see the model artifact's own `production_calibration` field for what was actually applied to this run."
    },
    "direction": {
      "status": "AVAILABLE",
      "method": "isotonic",
      "calibration_rows": 2636,
      "eval_rows": 2637,
      "brier_score_raw": 0.5001979127102023,
      "brier_score_calibrated": 0.5018382270506548,
      "log_loss_raw": 0.6933451074774488,
      "log_loss_calibrated": 0.6949921551822574,
      "improved": false,
      "recommended_for_production": false,
      "note": "Measured here; the caller (DirectionModel/EntryModel) swaps this stage onto the isotonic-calibrated estimator for live inference whenever `improved` is True, and leaves it on the raw estimator otherwise - see the model artifact's own `production_calibration` field for what was actually applied to this run."
    },
    "joint": {
      "status": "AVAILABLE",
      "method": "isotonic_per_class",
      "calibration_rows": 4456,
      "eval_rows": 4456,
      "brier_score_raw": 0.6251834329582705,
      "brier_score_calibrated": 0.6131196267408989,
      "log_loss_raw": 1.0411622867434611,
      "log_loss_calibrated": 1.1139434418468548,
      "improved": false,
      "recommended_for_production": false,
      "note": "Calibrates the joint long/short/no_trade probability directly, on top of whatever per-stage calibration already happened above - corrects residual miscalibration that the product of two independently-calibrated probabilities can still leave behind, which per-stage calibration alone cannot see. Applied only to the reported `probabilities` dict at inference (R3's consistency check, audit logging); trade_probability and direction_given_trade_probability, which R1a/R1b gate on, are never touched by this."
    }
  },
  "production_calibration": {
    "gate": "raw",
    "direction": "raw"
  },
  "per_symbol": {
    "SYM0/USDT:USDT": {
      "samples": 2228,
      "accuracy": 0.4151705565529623,
      "balanced_accuracy": 0.41861844569370454
    },
    "SYM1/USDT:USDT": {
      "samples": 2228,
      "accuracy": 0.41651705565529623,
      "balanced_accuracy": 0.42005455040048384
    },
    "SYM2/USDT:USDT": {
      "samples": 2228,
      "accuracy": 0.3509874326750449,
      "balanced_accuracy": 0.36166826903314014
    },
    "SYM3/USDT:USDT": {
      "samples": 2228,
      "accuracy": 0.38240574506283664,
      "balanced_accuracy": 0.394955541805124
    }
  },
  "architecture": "two_stage_cascade",
  "gate_threshold_sweep": [
    {
      "threshold": 0.3,
      "signals": 8808,
      "precision": 0.5986603088101726,
      "recall": 1.0,
      "f1": 0.7489524891698033,
      "meets_min_sample_size": true,
      "average_r": "NOT_AVAILABLE",
      "win_rate": "NOT_AVAILABLE",
      "profit_factor": "NOT_AVAILABLE",
      "expectancy": "NOT_AVAILABLE",
      "net_pnl": "NOT_AVAILABLE",
      "max_drawdown": "NOT_AVAILABLE"
    },
    {
      "threshold": 0.35,
      "signals": 8122,
      "precision": 0.6492243289830091,
      "recall": 1.0,
      "f1": 0.7873086972751027,
      "meets_min_sample_size": true,
      "average_r": "NOT_AVAILABLE",
      "win_rate": "NOT_AVAILABLE",
      "profit_factor": "NOT_AVAILABLE",
      "expectancy": "NOT_AVAILABLE",
      "net_pnl": "NOT_AVAILABLE",
      "max_drawdown": "NOT_AVAILABLE"
    },
    {
      "threshold": 0.4,
      "signals": 8122,
      "precision": 0.6492243289830091,
      "recall": 1.0,
      "f1": 0.7873086972751027,
      "meets_min_sample_size": true,
      "average_r": "NOT_AVAILABLE",
      "win_rate": "NOT_AVAILABLE",
      "profit_factor": "NOT_AVAILABLE",
      "expectancy": "NOT_AVAILABLE",
      "net_pnl": "NOT_AVAILABLE",
      "max_drawdown": "NOT_AVAILABLE"
    },
    {
      "threshold": 0.45,
      "signals": 8122,
      "precision": 0.6492243289830091,
      "recall": 1.0,
      "f1": 0.7873086972751027,
      "meets_min_sample_size": true,
      "average_r": "NOT_AVAILABLE",
      "win_rate": "NOT_AVAILABLE",
      "profit_factor": "NOT_AVAILABLE",
      "expectancy": "NOT_AVAILABLE",
      "net_pnl": "NOT_AVAILABLE",
      "max_drawdown": "NOT_AVAILABLE"
    },
    {
      "threshold": 0.5,
      "signals": 8114,
      "precision": 0.6490017254128666,
      "recall": 0.9986724824578039,
      "f1": 0.7867333980727571,
      "meets_min_sample_size": true,
      "average_r": "NOT_AVAILABLE",
      "win_rate": "NOT_AVAILABLE",
      "profit_factor": "NOT_AVAILABLE",
      "expectancy": "NOT_AVAILABLE",
      "net_pnl": "NOT_AVAILABLE",
      "max_drawdown": "NOT_AVAILABLE"
    },
    {
      "threshold": 0.55,
      "signals": 8111,
      "precision": 0.648995191714955,
      "recall": 0.9982931917314621,
      "f1": 0.7866108786610879,
      "meets_min_sample_size": true,
      "average_r": "NOT_AVAILABLE",
      "win_rate": "NOT_AVAILABLE",
      "profit_factor": "NOT_AVAILABLE",
      "expectancy": "NOT_AVAILABLE",
      "net_pnl": "NOT_AVAILABLE",
      "max_drawdown": "NOT_AVAILABLE"
    },
    {
      "threshold": 0.6,
      "signals": 8084,
      "precision": 0.6494309747649678,
      "recall": 0.99563815664707,
      "f1": 0.7861046642210077,
      "meets_min_sample_size": true,
      "average_r": "NOT_AVAILABLE",
      "win_rate": "NOT_AVAILABLE",
      "profit_factor": "NOT_AVAILABLE",
      "expectancy": "NOT_AVAILABLE",
      "net_pnl": "NOT_AVAILABLE",
      "max_drawdown": "NOT_AVAILABLE"
    },
    {
      "threshold": 0.65,
      "signals": 7641,
      "precision": 0.6529250098154692,
      "recall": 0.9461407168594728,
      "f1": 0.7726498373857829,
      "meets_min_sample_size": true,
      "average_r": "NOT_AVAILABLE",
      "win_rate": "NOT_AVAILABLE",
      "profit_factor": "NOT_AVAILABLE",
      "expectancy": "NOT_AVAILABLE",
      "net_pnl": "NOT_AVAILABLE",
      "max_drawdown": "NOT_AVAILABLE"
    },
    {
      "threshold": 0.7,
      "signals": 693,
      "precision": 0.6349206349206349,
      "recall": 0.083443959795183,
      "f1": 0.14750251424740193,
      "meets_min_sample_size": true,
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
  "recommended_gate_threshold": 0.35,
  "direction_threshold_sweep": [
    {
      "threshold": 0.3,
      "signals": 5273,
      "precision": 0.5057841835767115,
      "recall": 1.0,
      "f1": 0.6717884130982368,
      "meets_min_sample_size": true,
      "average_r": "NOT_AVAILABLE",
      "win_rate": "NOT_AVAILABLE",
      "profit_factor": "NOT_AVAILABLE",
      "expectancy": "NOT_AVAILABLE",
      "net_pnl": "NOT_AVAILABLE",
      "max_drawdown": "NOT_AVAILABLE"
    },
    {
      "threshold": 0.35,
      "signals": 5273,
      "precision": 0.5057841835767115,
      "recall": 1.0,
      "f1": 0.6717884130982368,
      "meets_min_sample_size": true,
      "average_r": "NOT_AVAILABLE",
      "win_rate": "NOT_AVAILABLE",
      "profit_factor": "NOT_AVAILABLE",
      "expectancy": "NOT_AVAILABLE",
      "net_pnl": "NOT_AVAILABLE",
      "max_drawdown": "NOT_AVAILABLE"
    },
    {
      "threshold": 0.4,
      "signals": 5273,
      "precision": 0.5057841835767115,
      "recall": 1.0,
      "f1": 0.6717884130982368,
      "meets_min_sample_size": true,
      "average_r": "NOT_AVAILABLE",
      "win_rate": "NOT_AVAILABLE",
      "profit_factor": "NOT_AVAILABLE",
      "expectancy": "NOT_AVAILABLE",
      "net_pnl": "NOT_AVAILABLE",
      "max_drawdown": "NOT_AVAILABLE"
    },
    {
      "threshold": 0.45,
      "signals": 5273,
      "precision": 0.5057841835767115,
      "recall": 1.0,
      "f1": 0.6717884130982368,
      "meets_min_sample_size": true,
      "average_r": "NOT_AVAILABLE",
      "win_rate": "NOT_AVAILABLE",
      "profit_factor": "NOT_AVAILABLE",
      "expectancy": "NOT_AVAILABLE",
      "net_pnl": "NOT_AVAILABLE",
      "max_drawdown": "NOT_AVAILABLE"
    },
    {
      "threshold": 0.5,
      "signals": 1891,
      "precision": 0.5108408249603384,
      "recall": 0.36220472440944884,
      "f1": 0.4238701184730145,
      "meets_min_sample_size": true,
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
  "recommended_direction_threshold": 0.3,
  "split": {
    "train": {
      "start": "2024-01-01T00:00:00+00:00",
      "end": "2024-01-16T12:05:00+00:00",
      "rows": 17864
    },
    "validation": {
      "start": "2024-01-16T13:00:00+00:00",
      "end": "2024-01-24T06:35:00+00:00",
      "rows": 8912
    },
    "test": {
      "start": "2024-01-24T07:30:00+00:00",
      "end": "2024-02-01T01:55:00+00:00",
      "rows": 8952
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
  "trained_at": "2026-08-22T21:16:52+00:00",
  "git_commit": "ebb1fb8c6bcf",
  "rows": 17864,
  "validation_rows": 8912,
  "positive_rate": 0.26739834673815904,
  "decision_threshold": 0.5,
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
    "threshold": 0.5,
    "accuracy": 0.4756508078994614,
    "precision": 0.4247318456417211,
    "recall": 0.9979721900347625,
    "f1": 0.5958661247081207,
    "roc_auc": 0.5639264438728518,
    "pr_auc": 0.4202800000520984,
    "confusion_matrix": {
      "labels": [
        0,
        1
      ],
      "raw": [
        [
          794,
          4666
        ],
        [
          7,
          3445
        ]
      ]
    },
    "class_distribution": {
      "positive": 3452,
      "negative": 5460
    },
    "predicted_positive_rate": 0.91012118491921,
    "probability_stats": {
      "mean": 0.5314136342433295,
      "median": 0.5355784818771747,
      "std": 0.011744402582818888,
      "min": 0.4949535499596524,
      "max": 0.5425726181954457
    }
  },
  "threshold_sweep": [
    {
      "threshold": 0.3,
      "signals": 8912,
      "precision": 0.387342908438061,
      "recall": 1.0,
      "f1": 0.5583953413134908,
      "meets_min_sample_size": true,
      "average_r": "NOT_AVAILABLE",
      "win_rate": "NOT_AVAILABLE",
      "profit_factor": "NOT_AVAILABLE",
      "expectancy": "NOT_AVAILABLE",
      "net_pnl": "NOT_AVAILABLE",
      "max_drawdown": "NOT_AVAILABLE"
    },
    {
      "threshold": 0.35,
      "signals": 8912,
      "precision": 0.387342908438061,
      "recall": 1.0,
      "f1": 0.5583953413134908,
      "meets_min_sample_size": true,
      "average_r": "NOT_AVAILABLE",
      "win_rate": "NOT_AVAILABLE",
      "profit_factor": "NOT_AVAILABLE",
      "expectancy": "NOT_AVAILABLE",
      "net_pnl": "NOT_AVAILABLE",
      "max_drawdown": "NOT_AVAILABLE"
    },
    {
      "threshold": 0.4,
      "signals": 8912,
      "precision": 0.387342908438061,
      "recall": 1.0,
      "f1": 0.5583953413134908,
      "meets_min_sample_size": true,
      "average_r": "NOT_AVAILABLE",
      "win_rate": "NOT_AVAILABLE",
      "profit_factor": "NOT_AVAILABLE",
      "expectancy": "NOT_AVAILABLE",
      "net_pnl": "NOT_AVAILABLE",
      "max_drawdown": "NOT_AVAILABLE"
    },
    {
      "threshold": 0.45,
      "signals": 8912,
      "precision": 0.387342908438061,
      "recall": 1.0,
      "f1": 0.5583953413134908,
      "meets_min_sample_size": true,
      "average_r": "NOT_AVAILABLE",
      "win_rate": "NOT_AVAILABLE",
      "profit_factor": "NOT_AVAILABLE",
      "expectancy": "NOT_AVAILABLE",
      "net_pnl": "NOT_AVAILABLE",
      "max_drawdown": "NOT_AVAILABLE"
    },
    {
      "threshold": 0.5,
      "signals": 8111,
      "precision": 0.4247318456417211,
      "recall": 0.9979721900347625,
      "f1": 0.5958661247081207,
      "meets_min_sample_size": true,
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
        "feature": "kama_slope",
        "importance": 2.0,
        "importance_pct": 0.125
      },
      {
        "feature": "dollar_volume_rank",
        "importance": 2.0,
        "importance_pct": 0.125
      },
      {
        "feature": "hour_cos",
        "importance": 2.0,
        "importance_pct": 0.125
      },
      {
        "feature": "kama_distance",
        "importance": 1.0,
        "importance_pct": 0.0625
      },
      {
        "feature": "log_return_48",
        "importance": 1.0,
        "importance_pct": 0.0625
      },
      {
        "feature": "realized_vol_48",
        "importance": 1.0,
        "importance_pct": 0.0625
      },
      {
        "feature": "garch_volatility",
        "importance": 1.0,
        "importance_pct": 0.0625
      },
      {
        "feature": "garch_vol_rank",
        "importance": 1.0,
        "importance_pct": 0.0625
      },
      {
        "feature": "garch_vol_ratio",
        "importance": 1.0,
        "importance_pct": 0.0625
      },
      {
        "feature": "wick_ratio",
        "importance": 1.0,
        "importance_pct": 0.0625
      },
      {
        "feature": "hmm_prob_bull",
        "importance": 1.0,
        "importance_pct": 0.0625
      },
      {
        "feature": "hmm_prob_bear",
        "importance": 1.0,
        "importance_pct": 0.0625
      },
      {
        "feature": "dow_sin",
        "importance": 1.0,
        "importance_pct": 0.0625
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
      }
    ],
    "shap": {
      "status": "NOT_AVAILABLE",
      "reason": "shap is not an installed project dependency"
    }
  },
  "calibration": {
    "status": "AVAILABLE",
    "method": "isotonic",
    "calibration_rows": 4456,
    "eval_rows": 4456,
    "brier_score_raw": 0.5113161849762184,
    "brier_score_calibrated": 0.44945115087299303,
    "log_loss_raw": 0.7044855668080364,
    "log_loss_calibrated": 0.6793054118866009,
    "improved": true,
    "recommended_for_production": true,
    "note": "Measured here; the caller (DirectionModel/EntryModel) swaps this stage onto the isotonic-calibrated estimator for live inference whenever `improved` is True, and leaves it on the raw estimator otherwise - see the model artifact's own `production_calibration` field for what was actually applied to this run."
  },
  "production_calibration": "isotonic",
  "split": {
    "train": {
      "start": "2024-01-01T00:00:00+00:00",
      "end": "2024-01-16T12:05:00+00:00",
      "rows": 17864
    },
    "validation": {
      "start": "2024-01-16T13:00:00+00:00",
      "end": "2024-01-24T06:35:00+00:00",
      "rows": 8912
    },
    "test": {
      "start": "2024-01-24T07:30:00+00:00",
      "end": "2024-02-01T01:55:00+00:00",
      "rows": 8952
    },
    "embargo_ms": 3000000,
    "scaled_down_from_nominal_months": true
  }
}
```

## Exit
```json
{
  "status": "TRAINED",
  "trained_at": "2026-08-22T21:16:52+00:00",
  "git_commit": "ebb1fb8c6bcf",
  "rows": 4041,
  "validation_rows": 5273,
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
    "target_tp_pct": {
      "mae": 0.011856655923235858,
      "rmse": 0.0152201858183846,
      "r2": -0.017157419261915496,
      "median_absolute_error": 0.01015828058115702,
      "target_stats": {
        "mean": 0.028407947290704057,
        "median": 0.02565493092930749,
        "std": 0.015091272773406727,
        "min": 0.00707004034928449,
        "max": 0.09008428825613032
      },
      "prediction_stats": {
        "mean": 0.02654159564233106,
        "median": 0.02629536782374029,
        "std": 0.0009794677286763345,
        "min": 0.024216345751574324,
        "max": 0.0312602239728999
      },
      "baseline_rule_based_mae": 0.016988696127542762,
      "beats_rule_based_baseline": true
    },
    "target_sl_pct": {
      "mae": 0.0016633617991903774,
      "rmse": 0.004306913581257664,
      "r2": -0.135823389980394,
      "median_absolute_error": 0.00024102525183391675,
      "target_stats": {
        "mean": 0.003999808366048064,
        "median": 0.002575485130900537,
        "std": 0.0040412038098741245,
        "min": 0.0016330634795190709,
        "max": 0.04075866587334107
      },
      "prediction_stats": {
        "mean": 0.002514333631839282,
        "median": 0.002474465866294144,
        "std": 0.00011468061377163781,
        "min": 0.0021556927372539995,
        "max": 0.002782060264755678
      },
      "baseline_rule_based_mae": 0.00370690388371049,
      "beats_rule_based_baseline": true
    },
    "target_trailing_pct": {
      "mae": 0.005928327961617929,
      "rmse": 0.0076100929091923,
      "r2": -0.017157419261915496,
      "median_absolute_error": 0.00507914029057851,
      "target_stats": {
        "mean": 0.014203973645352029,
        "median": 0.012827465464653746,
        "std": 0.0075456363867033634,
        "min": 0.003535020174642245,
        "max": 0.04504214412806516
      },
      "prediction_stats": {
        "mean": 0.01327079782116553,
        "median": 0.013147683911870146,
        "std": 0.0004897338643381672,
        "min": 0.012108172875787162,
        "max": 0.01563011198644995
      },
      "baseline_rule_based_mae": 0.008494348063771381,
      "beats_rule_based_baseline": true
    }
  },
  "feature_importance": {
    "target_tp_pct": {
      "status": "AVAILABLE",
      "method": "native_gain_or_split",
      "top_features": [
        {
          "feature": "garch_vol_rank",
          "importance": 9.0,
          "importance_pct": 0.075
        },
        {
          "feature": "ema_fast_slow_spread",
          "importance": 8.0,
          "importance_pct": 0.06666666666666667
        },
        {
          "feature": "realized_vol_48",
          "importance": 8.0,
          "importance_pct": 0.06666666666666667
        },
        {
          "feature": "vol_of_vol",
          "importance": 8.0,
          "importance_pct": 0.06666666666666667
        },
        {
          "feature": "hmm_regime_age",
          "importance": 8.0,
          "importance_pct": 0.06666666666666667
        },
        {
          "feature": "garch_volatility",
          "importance": 7.0,
          "importance_pct": 0.058333333333333334
        },
        {
          "feature": "hour_sin",
          "importance": 7.0,
          "importance_pct": 0.058333333333333334
        },
        {
          "feature": "dow_sin",
          "importance": 6.0,
          "importance_pct": 0.05
        },
        {
          "feature": "adx",
          "importance": 5.0,
          "importance_pct": 0.041666666666666664
        },
        {
          "feature": "volume_trend",
          "importance": 5.0,
          "importance_pct": 0.041666666666666664
        },
        {
          "feature": "bb_width",
          "importance": 4.0,
          "importance_pct": 0.03333333333333333
        },
        {
          "feature": "hmm_prob_bear",
          "importance": 4.0,
          "importance_pct": 0.03333333333333333
        },
        {
          "feature": "hmm_prob_sideways",
          "importance": 4.0,
          "importance_pct": 0.03333333333333333
        },
        {
          "feature": "fdi",
          "importance": 3.0,
          "importance_pct": 0.025
        },
        {
          "feature": "log_return_48",
          "importance": 3.0,
          "importance_pct": 0.025
        },
        {
          "feature": "garch_vol_ratio",
          "importance": 3.0,
          "importance_pct": 0.025
        },
        {
          "feature": "hmm_prob_bull",
          "importance": 3.0,
          "importance_pct": 0.025
        },
        {
          "feature": "hour_cos",
          "importance": 3.0,
          "importance_pct": 0.025
        },
        {
          "feature": "kama_slope_fast",
          "importance": 2.0,
          "importance_pct": 0.016666666666666666
        },
        {
          "feature": "di_spread",
          "importance": 2.0,
          "importance_pct": 0.016666666666666666
        }
      ],
      "shap": {
        "status": "NOT_AVAILABLE",
        "reason": "shap is not an installed project dependency"
      }
    },
    "target_sl_pct": {
      "status": "AVAILABLE",
      "method": "native_gain_or_split",
      "top_features": [
        {
          "feature": "atr_pct",
          "importance": 51.0,
          "importance_pct": 0.15088757396449703
        },
        {
          "feature": "atr_rank",
          "importance": 26.0,
          "importance_pct": 0.07692307692307693
        },
        {
          "feature": "garch_volatility",
          "importance": 23.0,
          "importance_pct": 0.06804733727810651
        },
        {
          "feature": "garch_vol_rank",
          "importance": 22.0,
          "importance_pct": 0.0650887573964497
        },
        {
          "feature": "hour_sin",
          "importance": 16.0,
          "importance_pct": 0.047337278106508875
        },
        {
          "feature": "vol_of_vol",
          "importance": 13.0,
          "importance_pct": 0.038461538461538464
        },
        {
          "feature": "bb_width",
          "importance": 12.0,
          "importance_pct": 0.03550295857988166
        },
        {
          "feature": "realized_vol_48",
          "importance": 11.0,
          "importance_pct": 0.03254437869822485
        },
        {
          "feature": "adx",
          "importance": 10.0,
          "importance_pct": 0.029585798816568046
        },
        {
          "feature": "fdi_delta",
          "importance": 9.0,
          "importance_pct": 0.026627218934911243
        },
        {
          "feature": "hmm_prob_bear",
          "importance": 9.0,
          "importance_pct": 0.026627218934911243
        },
        {
          "feature": "kama_distance",
          "importance": 8.0,
          "importance_pct": 0.023668639053254437
        },
        {
          "feature": "whipsaw_rate",
          "importance": 8.0,
          "importance_pct": 0.023668639053254437
        },
        {
          "feature": "hmm_prob_bull",
          "importance": 8.0,
          "importance_pct": 0.023668639053254437
        },
        {
          "feature": "volume_trend",
          "importance": 8.0,
          "importance_pct": 0.023668639053254437
        },
        {
          "feature": "dollar_volume_rank",
          "importance": 8.0,
          "importance_pct": 0.023668639053254437
        },
        {
          "feature": "hmm_prob_high_vol",
          "importance": 7.0,
          "importance_pct": 0.020710059171597635
        },
        {
          "feature": "kama_slope_fast",
          "importance": 6.0,
          "importance_pct": 0.01775147928994083
        },
        {
          "feature": "ema_fast_slow_spread",
          "importance": 6.0,
          "importance_pct": 0.01775147928994083
        },
        {
          "feature": "rsi_delta",
          "importance": 6.0,
          "importance_pct": 0.01775147928994083
        }
      ],
      "shap": {
        "status": "NOT_AVAILABLE",
        "reason": "shap is not an installed project dependency"
      }
    },
    "target_trailing_pct": {
      "status": "AVAILABLE",
      "method": "native_gain_or_split",
      "top_features": [
        {
          "feature": "garch_vol_rank",
          "importance": 9.0,
          "importance_pct": 0.075
        },
        {
          "feature": "ema_fast_slow_spread",
          "importance": 8.0,
          "importance_pct": 0.06666666666666667
        },
        {
          "feature": "realized_vol_48",
          "importance": 8.0,
          "importance_pct": 0.06666666666666667
        },
        {
          "feature": "vol_of_vol",
          "importance": 8.0,
          "importance_pct": 0.06666666666666667
        },
        {
          "feature": "hmm_regime_age",
          "importance": 8.0,
          "importance_pct": 0.06666666666666667
        },
        {
          "feature": "garch_volatility",
          "importance": 7.0,
          "importance_pct": 0.058333333333333334
        },
        {
          "feature": "hour_sin",
          "importance": 7.0,
          "importance_pct": 0.058333333333333334
        },
        {
          "feature": "dow_sin",
          "importance": 6.0,
          "importance_pct": 0.05
        },
        {
          "feature": "adx",
          "importance": 5.0,
          "importance_pct": 0.041666666666666664
        },
        {
          "feature": "volume_trend",
          "importance": 5.0,
          "importance_pct": 0.041666666666666664
        },
        {
          "feature": "bb_width",
          "importance": 4.0,
          "importance_pct": 0.03333333333333333
        },
        {
          "feature": "hmm_prob_bear",
          "importance": 4.0,
          "importance_pct": 0.03333333333333333
        },
        {
          "feature": "hmm_prob_sideways",
          "importance": 4.0,
          "importance_pct": 0.03333333333333333
        },
        {
          "feature": "fdi",
          "importance": 3.0,
          "importance_pct": 0.025
        },
        {
          "feature": "log_return_48",
          "importance": 3.0,
          "importance_pct": 0.025
        },
        {
          "feature": "garch_vol_ratio",
          "importance": 3.0,
          "importance_pct": 0.025
        },
        {
          "feature": "hmm_prob_bull",
          "importance": 3.0,
          "importance_pct": 0.025
        },
        {
          "feature": "hour_cos",
          "importance": 3.0,
          "importance_pct": 0.025
        },
        {
          "feature": "kama_slope_fast",
          "importance": 2.0,
          "importance_pct": 0.016666666666666666
        },
        {
          "feature": "di_spread",
          "importance": 2.0,
          "importance_pct": 0.016666666666666666
        }
      ],
      "shap": {
        "status": "NOT_AVAILABLE",
        "reason": "shap is not an installed project dependency"
      }
    }
  },
  "split": {
    "train": {
      "start": "2024-01-10T05:00:00+00:00",
      "end": "2024-01-16T12:05:00+00:00",
      "rows": 4041
    },
    "validation": {
      "start": "2024-01-16T13:00:00+00:00",
      "end": "2024-01-24T06:35:00+00:00",
      "rows": 5273
    },
    "test": {
      "start": "2024-01-24T07:30:00+00:00",
      "end": "2024-02-01T01:55:00+00:00",
      "rows": 5131
    },
    "embargo_ms": 3000000,
    "scaled_down_from_nominal_months": true
  }
}
```

## Risk
```json
{
  "status": "TRAINED",
  "trained_at": "2026-08-22T21:16:52+00:00",
  "git_commit": "ebb1fb8c6bcf",
  "rows": 4041,
  "validation_rows": 5273,
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
    "mae": 0.148126922256809,
    "rmse": 0.18832393119762456,
    "r2": 0.14279742602841405,
    "median_absolute_error": 0.12120701807380141,
    "target_stats": {
      "mean": 0.552771627321825,
      "median": 0.5570816226267464,
      "std": 0.20340600874513434,
      "min": 0.09258203633539419,
      "max": 0.9968814235842572
    },
    "prediction_stats": {
      "mean": 0.5816862536269525,
      "median": 0.5820464722381812,
      "std": 0.07761590397846015,
      "min": 0.41921869864547795,
      "max": 0.8003758917558668
    }
  },
  "feature_importance": {
    "status": "AVAILABLE",
    "method": "native_gain_or_split",
    "top_features": [
      {
        "feature": "garch_vol_rank",
        "importance": 199.0,
        "importance_pct": 0.07899960301707026
      },
      {
        "feature": "hour_sin",
        "importance": 101.0,
        "importance_pct": 0.04009527590313616
      },
      {
        "feature": "volume_trend",
        "importance": 100.0,
        "importance_pct": 0.03969829297340215
      },
      {
        "feature": "adx",
        "importance": 93.0,
        "importance_pct": 0.03691941246526399
      },
      {
        "feature": "wick_ratio",
        "importance": 87.0,
        "importance_pct": 0.03453751488685986
      },
      {
        "feature": "hour_cos",
        "importance": 86.0,
        "importance_pct": 0.034140531957125846
      },
      {
        "feature": "vol_of_vol",
        "importance": 76.0,
        "importance_pct": 0.03017070265978563
      },
      {
        "feature": "dollar_volume_rank",
        "importance": 76.0,
        "importance_pct": 0.03017070265978563
      },
      {
        "feature": "rsi_delta",
        "importance": 75.0,
        "importance_pct": 0.029773719730051607
      },
      {
        "feature": "realized_vol_48",
        "importance": 72.0,
        "importance_pct": 0.028582770940849545
      },
      {
        "feature": "fdi",
        "importance": 69.0,
        "importance_pct": 0.02739182215164748
      },
      {
        "feature": "log_return_3",
        "importance": 69.0,
        "importance_pct": 0.02739182215164748
      },
      {
        "feature": "fdi_delta",
        "importance": 68.0,
        "importance_pct": 0.026994839221913456
      },
      {
        "feature": "log_return_1",
        "importance": 68.0,
        "importance_pct": 0.026994839221913456
      },
      {
        "feature": "garch_volatility",
        "importance": 66.0,
        "importance_pct": 0.026200873362445413
      },
      {
        "feature": "hmm_prob_bear",
        "importance": 66.0,
        "importance_pct": 0.026200873362445413
      },
      {
        "feature": "di_spread",
        "importance": 65.0,
        "importance_pct": 0.025803890432711394
      },
      {
        "feature": "kama_distance",
        "importance": 63.0,
        "importance_pct": 0.02500992457324335
      },
      {
        "feature": "atr_pct",
        "importance": 61.0,
        "importance_pct": 0.02421595871377531
      },
      {
        "feature": "bb_position",
        "importance": 60.0,
        "importance_pct": 0.023818975784041286
      }
    ],
    "shap": {
      "status": "NOT_AVAILABLE",
      "reason": "shap is not an installed project dependency"
    }
  },
  "training_row_filter": "direction_target != NO_TRADE_OR_FAIL - see RiskModel.train docstring",
  "split": {
    "train": {
      "start": "2024-01-10T05:00:00+00:00",
      "end": "2024-01-16T12:05:00+00:00",
      "rows": 4041
    },
    "validation": {
      "start": "2024-01-16T13:00:00+00:00",
      "end": "2024-01-24T06:35:00+00:00",
      "rows": 5273
    },
    "test": {
      "start": "2024-01-24T07:30:00+00:00",
      "end": "2024-02-01T01:55:00+00:00",
      "rows": 5131
    },
    "embargo_ms": 3000000,
    "scaled_down_from_nominal_months": true
  }
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
    "NO_TRADE_OR_FAIL": 21322,
    "SHORT_SUCCESS": 7313,
    "LONG_SUCCESS": 7173
  }
}
```

## Walk-Forward
{'status': 'AVAILABLE', 'method': 'expanding_window', 'n_folds': 3, 'folds': [{'fold': 2, 'train_rows': 10700, 'validation_rows': 5371, 'accuracy': 0.4665797803016198, 'balanced_accuracy': 0.3333333333333333, 'log_loss': 1.356369247178182, 'macro_f1': 0.2120942829334349}, {'fold': 3, 'train_rows': 16072, 'validation_rows': 5371, 'accuracy': 0.37814187302178365, 'balanced_accuracy': 0.39786742979270534, 'log_loss': 1.1376005060041927, 'macro_f1': 0.36743030359114354}, {'fold': 4, 'train_rows': 21444, 'validation_rows': 5372, 'accuracy': 0.39538346984363365, 'balanced_accuracy': 0.41781858897188645, 'log_loss': 1.0585365125243709, 'macro_f1': 0.39521612646309806}], 'accuracy_mean': 0.41336837438901236, 'accuracy_std': 0.03827887541572602, 'balanced_accuracy_mean': 0.38300645069930833, 'balanced_accuracy_std': 0.03605621656791613, 'note': 'Each fold fits an independent two-stage cascade (not the production model) purely to measure how much accuracy varies across different time periods.'}

## Backtest
```json
{
  "start": "2024-01-13T21:10:00+00:00",
  "end": "2024-02-01T05:55:00+00:00",
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
  "signals_generated": 21160,
  "signals_rejected": 21160,
  "rejection_breakdown": {
    "R1B_DIRECTION_CONFIDENCE_TOO_LOW": 18955,
    "R1A_GATE_CONFIDENCE_TOO_LOW": 2205
  },
  "oos_disclosure": null,
  "trades": "[0 entries omitted from Markdown - see JSON export]",
  "equity_curve": "[2000 entries omitted from Markdown - see JSON export]"
}
```

## Backtest Reliability
```json
{
  "status": "AVAILABLE",
  "total_trades": 0,
  "minimum_trades_for_reliability": 30,
  "statistically_reliable": false,
  "signals_generated": 21160,
  "signals_rejected": 21160,
  "rejection_breakdown": {
    "R1B_DIRECTION_CONFIDENCE_TOO_LOW": 18955,
    "R1A_GATE_CONFIDENCE_TOO_LOW": 2205
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
    "SYM0/USDT:USDT": 8952,
    "SYM1/USDT:USDT": 8952,
    "SYM2/USDT:USDT": 8952,
    "SYM3/USDT:USDT": 8952
  },
  "per_symbol_direction_accuracy": {
    "SYM0/USDT:USDT": {
      "samples": 2228,
      "accuracy": 0.4151705565529623,
      "balanced_accuracy": 0.41861844569370454
    },
    "SYM1/USDT:USDT": {
      "samples": 2228,
      "accuracy": 0.41651705565529623,
      "balanced_accuracy": 0.42005455040048384
    },
    "SYM2/USDT:USDT": {
      "samples": 2228,
      "accuracy": 0.3509874326750449,
      "balanced_accuracy": 0.36166826903314014
    },
    "SYM3/USDT:USDT": {
      "samples": 2228,
      "accuracy": 0.38240574506283664,
      "balanced_accuracy": 0.394955541805124
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
  "CRITICAL": [],
  "HIGH": [
    "Backtest produced only 0 trade(s) from 21160 candidate signal(s) (21160 rejected, top rejection reason: R1B_DIRECTION_CONFIDENCE_TOO_LOW) - win rate/profit factor/Sharpe/expectancy are not statistically meaningful below 30 trades; widen the validation window or run a walk-forward-style backtest across multiple periods before trusting these numbers",
    "Microstructure/derivatives feed(s) sit at their neutral default for nearly every row (likely not being collected): funding_rate, long_short_ratio, open_interest_change, taker_buy_sell_ratio - check data collection for these sources before trusting Entry/Direction feature importance that involves them"
  ],
  "MEDIUM": [],
  "LOW": [
    "entry isotonic calibration measurably improves log loss and is wired into inference"
  ]
}
```