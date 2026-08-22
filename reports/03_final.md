# AI Diagnostic Summary

- Overall status: **CRITICAL**
- Strongest component: entry
- Weakest component: exit
- Biggest data problem: 5 feature(s) are effectively empty on every row (liquidation_imbalance, ob_imbalance, ob_imbalance_delta, ob_spread_bps, ob_spread_rank) - they occupy the feature contract and force a retrain on every change without contributing anything; remove them or fix the backfill that should be populating them
- Biggest ML problem: exit captures the least of its available headroom (0.0% lift over chance)
- Biggest validation problem: none measured (walk-forward across 4 folds, accuracy std=0.009)
- Biggest trading problem: backtest metrics are not statistically reliable: only 0 trade(s) generated from 21100 candidate signal(s) (21100 rejected) - treat win rate/profit factor/Sharpe as noise, not a performance estimate
- Most important improvement: NOT_AVAILABLE
- Most important degradation: NOT_AVAILABLE
- Recommended next action: Fix before anything else: 5 feature(s) are effectively empty on every row (liquidation_imbalance, ob_imbalance, ob_imbalance_delta, ob_spread_bps, ob_spread_rank) - they occupy the feature contract and force a retrain on every change without contributing anything; remove them or fix the backfill that should be populating them

# ML Diagnostic Report

Run `final` generated 2026-08-22T22:31:24+00:00 (git `eefd3f14e142`)

## Training Overview
- Timeframe: 5m
- Symbols: SYM0/USDT:USDT, SYM1/USDT:USDT, SYM2/USDT:USDT, SYM3/USDT:USDT
- Training period: {'start': 1704881700000, 'end': 1705813800000, 'rows': 12428}
- Validation period: {'start': 1705817100000, 'end': 1706281500000, 'rows': 6196}
- Model versions: {'direction': '2026-08-22T22:26:29+00:00', 'entry': '2026-08-22T22:26:30+00:00', 'exit': '2026-08-22T22:26:30+00:00', 'risk': '2026-08-22T22:26:30+00:00'}

## Dataset Health
```json
{
  "total_candidate_rows": 36000,
  "valid_samples": 24944,
  "rejected_invalid_label_rows": 192,
  "dropped_missing_or_inf_rows": 10864,
  "duplicate_feature_rows": 0,
  "training_samples": 12428,
  "validation_samples": 6196,
  "test_samples": 6240,
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
    "SYM2/USDT:USDT": 6237,
    "SYM3/USDT:USDT": 6237,
    "SYM0/USDT:USDT": 6235,
    "SYM1/USDT:USDT": 6235
  },
  "null_counts_by_feature": {
    "kama_distance": 0,
    "kama_slope": 0,
    "kama_slope_fast": 0,
    "ema_fast_slow_spread": 0,
    "close_ema_slow_ratio": 0,
    "adx": 0,
    "di_spread": 0,
    "fdi": 0,
    "fdi_trending": 0,
    "fdi_delta": 0,
    "bb_width": 0,
    "bb_position": 0,
    "rsi": 0,
    "rsi_delta": 0,
    "log_return_1": 0,
    "log_return_3": 0,
    "log_return_12": 0,
    "log_return_48": 0,
    "momentum_rank": 0,
    "atr_pct": 0,
    "atr_rank": 0,
    "realized_vol_12": 0,
    "realized_vol_48": 0,
    "garch_volatility": 0,
    "garch_vol_rank": 0,
    "garch_vol_ratio": 0,
    "vol_of_vol": 0,
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
    "volume_rank": 0,
    "volume_trend": 0,
    "dollar_volume_rank": 0,
    "ob_imbalance": 24944,
    "ob_imbalance_delta": 24944,
    "ob_spread_bps": 24944,
    "ob_spread_rank": 24944,
    "liquidation_imbalance": 24944,
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
      "rows": 12428,
      "span_days": 10.8,
      "capacity_rows": 12432,
      "coverage_pct": 0.9997
    },
    "validation": {
      "rows": 6196,
      "span_days": 5.4,
      "capacity_rows": 6196,
      "coverage_pct": 1.0
    },
    "test": {
      "rows": 6240,
      "span_days": 5.4,
      "capacity_rows": 6240,
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
  "trained_at": "2026-08-22T22:26:29+00:00",
  "git_commit": "eefd3f14e142",
  "rows": 12428,
  "validation_rows": 6196,
  "classes": [
    "LONG_SUCCESS",
    "SHORT_SUCCESS",
    "NO_TRADE_OR_FAIL"
  ],
  "distribution": {
    "train": {
      "NO_TRADE_OR_FAIL": 5341,
      "LONG_SUCCESS": 3578,
      "SHORT_SUCCESS": 3509
    },
    "validation": {
      "NO_TRADE_OR_FAIL": 2543,
      "SHORT_SUCCESS": 1934,
      "LONG_SUCCESS": 1719
    },
    "test": {
      "NO_TRADE_OR_FAIL": 2741,
      "SHORT_SUCCESS": 1781,
      "LONG_SUCCESS": 1718
    },
    "all": {
      "NO_TRADE_OR_FAIL": 10652,
      "SHORT_SUCCESS": 7263,
      "LONG_SUCCESS": 7029
    },
    "label_distribution_shift": 0.029790547718837268
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
    "accuracy": 0.4092963202065849,
    "balanced_accuracy": 0.3418638761708886,
    "log_loss": 1.010160780240958,
    "macro_precision": 0.34153749060049415,
    "macro_recall": 0.3418638761708886,
    "macro_f1": 0.25194286605401633,
    "weighted_f1": 0.2904212667340653,
    "per_class": {
      "LONG_SUCCESS": {
        "precision": 0.27155172413793105,
        "recall": 0.03664921465968586,
        "f1": 0.0645822655048693,
        "support": 1719
      },
      "SHORT_SUCCESS": {
        "precision": 0.3325,
        "recall": 0.0687693898655636,
        "f1": 0.11396743787489289,
        "support": 1934
      },
      "NO_TRADE_OR_FAIL": {
        "precision": 0.4205607476635514,
        "recall": 0.9201730239874164,
        "f1": 0.5772788947822869,
        "support": 2543
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
          63,
          142,
          1514
        ],
        [
          91,
          133,
          1710
        ],
        [
          78,
          125,
          2340
        ]
      ],
      "normalized": [
        [
          0.03664921465968586,
          0.08260616637579989,
          0.8807446189645143
        ],
        [
          0.04705274043433299,
          0.0687693898655636,
          0.8841778697001034
        ],
        [
          0.03067243413291388,
          0.04915454187966968,
          0.9201730239874164
        ]
      ]
    },
    "class_distribution": {
      "LONG_SUCCESS": 1719,
      "SHORT_SUCCESS": 1934,
      "NO_TRADE_OR_FAIL": 2543
    },
    "predicted_class_distribution": {
      "LONG_SUCCESS": 232,
      "SHORT_SUCCESS": 400,
      "NO_TRADE_OR_FAIL": 5564
    },
    "probability_stats": {
      "LONG_SUCCESS": {
        "mean": 0.29407490843408546,
        "median": 0.3194375892717486,
        "std": 0.08688738651866668,
        "min": 0.0,
        "max": 0.5051291608662501
      },
      "SHORT_SUCCESS": {
        "mean": 0.2947381855491369,
        "median": 0.3202136637012806,
        "std": 0.08708639167632114,
        "min": 0.0,
        "max": 0.5096119706238519
      },
      "NO_TRADE_OR_FAIL": {
        "mean": 0.41118690601677765,
        "median": 0.3596059113300494,
        "std": 0.17383404839807545,
        "min": 0.0,
        "max": 1.0
      }
    },
    "confidence_threshold_analysis": [
      {
        "confidence_threshold": 0.3,
        "n_predictions": 6196,
        "pct_of_samples": 1.0,
        "accuracy": 0.4092963202065849,
        "balanced_accuracy": 0.3418638761708886,
        "precision": 0.34153749060049415,
        "recall": 0.3418638761708886,
        "f1": 0.25194286605401633,
        "distinct_predicted_classes": 3,
        "is_degenerate": false,
        "per_class_recall": {
          "LONG_SUCCESS": 0.03664921465968586,
          "SHORT_SUCCESS": 0.0687693898655636,
          "NO_TRADE_OR_FAIL": 0.9201730239874164
        }
      },
      {
        "confidence_threshold": 0.35,
        "n_predictions": 3939,
        "pct_of_samples": 0.6357327307940607,
        "accuracy": 0.4523990860624524,
        "balanced_accuracy": 0.3399136510118102,
        "precision": 0.3644605799098348,
        "recall": 0.3399136510118102,
        "f1": 0.2327508088695542,
        "distinct_predicted_classes": 3,
        "is_degenerate": false,
        "per_class_recall": {
          "LONG_SUCCESS": 0.013579049466537343,
          "SHORT_SUCCESS": 0.026455026455026454,
          "NO_TRADE_OR_FAIL": 0.979706877113867
        }
      },
      {
        "confidence_threshold": 0.4,
        "n_predictions": 1197,
        "pct_of_samples": 0.19318915429309233,
        "accuracy": 0.6599832915622389,
        "balanced_accuracy": 0.338302524349036,
        "precision": 0.5260037348272643,
        "recall": 0.338302524349036,
        "f1": 0.2751634106322613,
        "distinct_predicted_classes": 3,
        "is_degenerate": false,
        "per_class_recall": {
          "LONG_SUCCESS": 0.010256410256410256,
          "SHORT_SUCCESS": 0.004651162790697674,
          "NO_TRADE_OR_FAIL": 1.0
        }
      },
      {
        "confidence_threshold": 0.45,
        "n_predictions": 725,
        "pct_of_samples": 0.11701097482246611,
        "accuracy": 0.8317241379310345,
        "balanced_accuracy": 0.34903070852633383,
        "precision": 0.5841070875889818,
        "recall": 0.34903070852633383,
        "f1": 0.3332904437902235,
        "distinct_predicted_classes": 3,
        "is_degenerate": false,
        "per_class_recall": {
          "LONG_SUCCESS": 0.029850746268656716,
          "SHORT_SUCCESS": 0.017241379310344827,
          "NO_TRADE_OR_FAIL": 1.0
        }
      },
      {
        "confidence_threshold": 0.5,
        "n_predictions": 659,
        "pct_of_samples": 0.10635894125242092,
        "accuracy": 0.8634294385432474,
        "balanced_accuracy": 0.3542452830188679,
        "precision": 0.5949216087252897,
        "recall": 0.3542452830188679,
        "f1": 0.34875852117231426,
        "distinct_predicted_classes": 3,
        "is_degenerate": false,
        "per_class_recall": {
          "LONG_SUCCESS": 0.03773584905660377,
          "SHORT_SUCCESS": 0.025,
          "NO_TRADE_OR_FAIL": 1.0
        }
      },
      {
        "confidence_threshold": 0.55,
        "n_predictions": 477,
        "pct_of_samples": 0.07698515171078114,
        "accuracy": 0.9958071278825996,
        "balanced_accuracy": 0.3333333333333333,
        "precision": 0.33193570929419985,
        "recall": 0.3333333333333333,
        "f1": 0.3326330532212885,
        "distinct_predicted_classes": 1,
        "is_degenerate": true,
        "per_class_recall": {
          "LONG_SUCCESS": 0.0,
          "SHORT_SUCCESS": 0.0,
          "NO_TRADE_OR_FAIL": 1.0
        }
      },
      {
        "confidence_threshold": 0.6,
        "n_predictions": 477,
        "pct_of_samples": 0.07698515171078114,
        "accuracy": 0.9958071278825996,
        "balanced_accuracy": 0.3333333333333333,
        "precision": 0.33193570929419985,
        "recall": 0.3333333333333333,
        "f1": 0.3326330532212885,
        "distinct_predicted_classes": 1,
        "is_degenerate": true,
        "per_class_recall": {
          "LONG_SUCCESS": 0.0,
          "SHORT_SUCCESS": 0.0,
          "NO_TRADE_OR_FAIL": 1.0
        }
      },
      {
        "confidence_threshold": 0.65,
        "n_predictions": 477,
        "pct_of_samples": 0.07698515171078114,
        "accuracy": 0.9958071278825996,
        "balanced_accuracy": 0.3333333333333333,
        "precision": 0.33193570929419985,
        "recall": 0.3333333333333333,
        "f1": 0.3326330532212885,
        "distinct_predicted_classes": 1,
        "is_degenerate": true,
        "per_class_recall": {
          "LONG_SUCCESS": 0.0,
          "SHORT_SUCCESS": 0.0,
          "NO_TRADE_OR_FAIL": 1.0
        }
      },
      {
        "confidence_threshold": 0.7,
        "n_predictions": 472,
        "pct_of_samples": 0.07617817947062622,
        "accuracy": 0.9978813559322034,
        "balanced_accuracy": 0.5,
        "precision": 0.3326271186440678,
        "recall": 0.3333333333333333,
        "f1": 0.3329798515376458,
        "distinct_predicted_classes": 1,
        "is_degenerate": true,
        "per_class_recall": {
          "LONG_SUCCESS": 0.0,
          "SHORT_SUCCESS": 0.0,
          "NO_TRADE_OR_FAIL": 1.0
        }
      },
      {
        "confidence_threshold": 0.75,
        "n_predictions": 472,
        "pct_of_samples": 0.07617817947062622,
        "accuracy": 0.9978813559322034,
        "balanced_accuracy": 0.5,
        "precision": 0.3326271186440678,
        "recall": 0.3333333333333333,
        "f1": 0.3329798515376458,
        "distinct_predicted_classes": 1,
        "is_degenerate": true,
        "per_class_recall": {
          "LONG_SUCCESS": 0.0,
          "SHORT_SUCCESS": 0.0,
          "NO_TRADE_OR_FAIL": 1.0
        }
      },
      {
        "confidence_threshold": 0.8,
        "n_predictions": 471,
        "pct_of_samples": 0.07601678502259522,
        "accuracy": 1.0,
        "balanced_accuracy": 1.0,
        "precision": 0.3333333333333333,
        "recall": 0.3333333333333333,
        "f1": 0.3333333333333333,
        "distinct_predicted_classes": 1,
        "is_degenerate": true,
        "per_class_recall": {
          "LONG_SUCCESS": 0.0,
          "SHORT_SUCCESS": 0.0,
          "NO_TRADE_OR_FAIL": 1.0
        }
      },
      {
        "confidence_threshold": 0.85,
        "n_predictions": 470,
        "pct_of_samples": 0.07585539057456424,
        "accuracy": 1.0,
        "balanced_accuracy": 1.0,
        "precision": 0.3333333333333333,
        "recall": 0.3333333333333333,
        "f1": 0.3333333333333333,
        "distinct_predicted_classes": 1,
        "is_degenerate": true,
        "per_class_recall": {
          "LONG_SUCCESS": 0.0,
          "SHORT_SUCCESS": 0.0,
          "NO_TRADE_OR_FAIL": 1.0
        }
      }
    ]
  },
  "metrics_source": "direction:LGBMClassifier:7ff3b4552f50|gate:CalibratedClassifierCV:7ff3b7fb8390",
  "metrics_provenance": "scored through the shipped (post-calibration) cascade on the validation block; that block also drove early stopping, threshold selection and calibration fitting",
  "metrics_raw_uncalibrated": {
    "accuracy": 0.41058747579083277,
    "balanced_accuracy": 0.33352724452200894,
    "log_loss": 1.0430314523542135,
    "macro_precision": 0.47016411084207693,
    "macro_recall": 0.33352724452200894,
    "macro_f1": 0.19440606031571983,
    "weighted_f1": 0.23921331417265054,
    "per_class": {
      "LONG_SUCCESS": {
        "precision": 1.0,
        "recall": 0.0005817335660267597,
        "f1": 0.0011627906976744186,
        "support": 1719
      },
      "SHORT_SUCCESS": {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "support": 1934
      },
      "NO_TRADE_OR_FAIL": {
        "precision": 0.41049233252623085,
        "recall": 1.0,
        "f1": 0.582055390249485,
        "support": 2543
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
          1,
          0,
          1718
        ],
        [
          0,
          0,
          1934
        ],
        [
          0,
          0,
          2543
        ]
      ],
      "normalized": [
        [
          0.0005817335660267597,
          0.0,
          0.9994182664339732
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
      "LONG_SUCCESS": 1719,
      "SHORT_SUCCESS": 1934,
      "NO_TRADE_OR_FAIL": 2543
    },
    "predicted_class_distribution": {
      "LONG_SUCCESS": 1,
      "SHORT_SUCCESS": 0,
      "NO_TRADE_OR_FAIL": 6195
    },
    "probability_stats": {
      "LONG_SUCCESS": {
        "mean": 0.2652792367724523,
        "median": 0.2800281484226493,
        "std": 0.047413037403113936,
        "min": 0.09112610225222831,
        "max": 0.33711225666171796
      },
      "SHORT_SUCCESS": {
        "mean": 0.2658703149753613,
        "median": 0.28043052712363414,
        "std": 0.047554428468100214,
        "min": 0.08853177820591286,
        "max": 0.336506551046166
      },
      "NO_TRADE_OR_FAIL": {
        "mean": 0.4688504482521864,
        "median": 0.43926116276765215,
        "std": 0.09476959033581417,
        "min": 0.32870523629246806,
        "max": 0.8180865876744181
      }
    },
    "confidence_threshold_analysis": [
      {
        "confidence_threshold": 0.3,
        "n_predictions": 6196,
        "pct_of_samples": 1.0,
        "accuracy": 0.41058747579083277,
        "balanced_accuracy": 0.33352724452200894,
        "precision": 0.47016411084207693,
        "recall": 0.33352724452200894,
        "f1": 0.19440606031571983,
        "distinct_predicted_classes": 2,
        "is_degenerate": false,
        "per_class_recall": {
          "LONG_SUCCESS": 0.0005817335660267597,
          "SHORT_SUCCESS": 0.0,
          "NO_TRADE_OR_FAIL": 1.0
        }
      },
      {
        "confidence_threshold": 0.35,
        "n_predictions": 6180,
        "pct_of_samples": 0.9974176888315042,
        "accuracy": 0.41067961165048544,
        "balanced_accuracy": 0.3333333333333333,
        "precision": 0.13689320388349516,
        "recall": 0.3333333333333333,
        "f1": 0.19408121128699243,
        "distinct_predicted_classes": 1,
        "is_degenerate": true,
        "per_class_recall": {
          "LONG_SUCCESS": 0.0,
          "SHORT_SUCCESS": 0.0,
          "NO_TRADE_OR_FAIL": 1.0
        }
      },
      {
        "confidence_threshold": 0.4,
        "n_predictions": 6003,
        "pct_of_samples": 0.9688508715300194,
        "accuracy": 0.41329335332333833,
        "balanced_accuracy": 0.3333333333333333,
        "precision": 0.13776445110777943,
        "recall": 0.3333333333333333,
        "f1": 0.19495520980669498,
        "distinct_predicted_classes": 1,
        "is_degenerate": true,
        "per_class_recall": {
          "LONG_SUCCESS": 0.0,
          "SHORT_SUCCESS": 0.0,
          "NO_TRADE_OR_FAIL": 1.0
        }
      },
      {
        "confidence_threshold": 0.45,
        "n_predictions": 2112,
        "pct_of_samples": 0.3408650742414461,
        "accuracy": 0.5260416666666666,
        "balanced_accuracy": 0.3333333333333333,
        "precision": 0.1753472222222222,
        "recall": 0.3333333333333333,
        "f1": 0.22980659840728102,
        "distinct_predicted_classes": 1,
        "is_degenerate": true,
        "per_class_recall": {
          "LONG_SUCCESS": 0.0,
          "SHORT_SUCCESS": 0.0,
          "NO_TRADE_OR_FAIL": 1.0
        }
      },
      {
        "confidence_threshold": 0.5,
        "n_predictions": 875,
        "pct_of_samples": 0.14122014202711428,
        "accuracy": 0.7565714285714286,
        "balanced_accuracy": 0.3333333333333333,
        "precision": 0.2521904761904762,
        "recall": 0.3333333333333333,
        "f1": 0.2871394491433528,
        "distinct_predicted_classes": 1,
        "is_degenerate": true,
        "per_class_recall": {
          "LONG_SUCCESS": 0.0,
          "SHORT_SUCCESS": 0.0,
          "NO_TRADE_OR_FAIL": 1.0
        }
      },
      {
        "confidence_threshold": 0.55,
        "n_predictions": 567,
        "pct_of_samples": 0.09151065203357005,
        "accuracy": 0.9171075837742504,
        "balanced_accuracy": 0.3333333333333333,
        "precision": 0.3057025279247501,
        "recall": 0.3333333333333333,
        "f1": 0.3189205765102729,
        "distinct_predicted_classes": 1,
        "is_degenerate": true,
        "per_class_recall": {
          "LONG_SUCCESS": 0.0,
          "SHORT_SUCCESS": 0.0,
          "NO_TRADE_OR_FAIL": 1.0
        }
      },
      {
        "confidence_threshold": 0.6,
        "n_predictions": 474,
        "pct_of_samples": 0.07650096836668818,
        "accuracy": 0.9957805907172996,
        "balanced_accuracy": 0.3333333333333333,
        "precision": 0.33192686357243323,
        "recall": 0.3333333333333333,
        "f1": 0.33262861169837915,
        "distinct_predicted_classes": 1,
        "is_degenerate": true,
        "per_class_recall": {
          "LONG_SUCCESS": 0.0,
          "SHORT_SUCCESS": 0.0,
          "NO_TRADE_OR_FAIL": 1.0
        }
      },
      {
        "confidence_threshold": 0.65,
        "n_predictions": 469,
        "pct_of_samples": 0.07569399612653324,
        "accuracy": 1.0,
        "balanced_accuracy": 1.0,
        "precision": 0.3333333333333333,
        "recall": 0.3333333333333333,
        "f1": 0.3333333333333333,
        "distinct_predicted_classes": 1,
        "is_degenerate": true,
        "per_class_recall": {
          "LONG_SUCCESS": 0.0,
          "SHORT_SUCCESS": 0.0,
          "NO_TRADE_OR_FAIL": 1.0
        }
      },
      {
        "confidence_threshold": 0.7,
        "n_predictions": 469,
        "pct_of_samples": 0.07569399612653324,
        "accuracy": 1.0,
        "balanced_accuracy": 1.0,
        "precision": 0.3333333333333333,
        "recall": 0.3333333333333333,
        "f1": 0.3333333333333333,
        "distinct_predicted_classes": 1,
        "is_degenerate": true,
        "per_class_recall": {
          "LONG_SUCCESS": 0.0,
          "SHORT_SUCCESS": 0.0,
          "NO_TRADE_OR_FAIL": 1.0
        }
      },
      {
        "confidence_threshold": 0.75,
        "n_predictions": 468,
        "pct_of_samples": 0.07553260167850226,
        "accuracy": 1.0,
        "balanced_accuracy": 1.0,
        "precision": 0.3333333333333333,
        "recall": 0.3333333333333333,
        "f1": 0.3333333333333333,
        "distinct_predicted_classes": 1,
        "is_degenerate": true,
        "per_class_recall": {
          "LONG_SUCCESS": 0.0,
          "SHORT_SUCCESS": 0.0,
          "NO_TRADE_OR_FAIL": 1.0
        }
      },
      {
        "confidence_threshold": 0.8,
        "n_predictions": 36,
        "pct_of_samples": 0.005810200129115558,
        "accuracy": 1.0,
        "balanced_accuracy": 1.0,
        "precision": 0.3333333333333333,
        "recall": 0.3333333333333333,
        "f1": 0.3333333333333333,
        "distinct_predicted_classes": 1,
        "is_degenerate": true,
        "per_class_recall": {
          "LONG_SUCCESS": 0.0,
          "SHORT_SUCCESS": 0.0,
          "NO_TRADE_OR_FAIL": 1.0
        }
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
          "feature": "garch_vol_rank",
          "importance": 33.0,
          "importance_pct": 0.06470588235294118
        },
        {
          "feature": "garch_volatility",
          "importance": 31.0,
          "importance_pct": 0.060784313725490195
        },
        {
          "feature": "realized_vol_48",
          "importance": 26.0,
          "importance_pct": 0.050980392156862744
        },
        {
          "feature": "log_return_48",
          "importance": 22.0,
          "importance_pct": 0.043137254901960784
        },
        {
          "feature": "hour_cos",
          "importance": 21.0,
          "importance_pct": 0.041176470588235294
        },
        {
          "feature": "realized_vol_12",
          "importance": 19.0,
          "importance_pct": 0.03725490196078431
        },
        {
          "feature": "hmm_prob_bull",
          "importance": 18.0,
          "importance_pct": 0.03529411764705882
        },
        {
          "feature": "ema_fast_slow_spread",
          "importance": 16.0,
          "importance_pct": 0.03137254901960784
        },
        {
          "feature": "fdi_delta",
          "importance": 16.0,
          "importance_pct": 0.03137254901960784
        },
        {
          "feature": "garch_vol_ratio",
          "importance": 16.0,
          "importance_pct": 0.03137254901960784
        },
        {
          "feature": "hour_sin",
          "importance": 16.0,
          "importance_pct": 0.03137254901960784
        },
        {
          "feature": "kama_slope",
          "importance": 14.0,
          "importance_pct": 0.027450980392156862
        },
        {
          "feature": "kama_slope_fast",
          "importance": 14.0,
          "importance_pct": 0.027450980392156862
        },
        {
          "feature": "adx",
          "importance": 14.0,
          "importance_pct": 0.027450980392156862
        },
        {
          "feature": "atr_rank",
          "importance": 14.0,
          "importance_pct": 0.027450980392156862
        },
        {
          "feature": "wick_ratio",
          "importance": 13.0,
          "importance_pct": 0.025490196078431372
        },
        {
          "feature": "volume_trend",
          "importance": 13.0,
          "importance_pct": 0.025490196078431372
        },
        {
          "feature": "bb_width",
          "importance": 12.0,
          "importance_pct": 0.023529411764705882
        },
        {
          "feature": "log_return_1",
          "importance": 12.0,
          "importance_pct": 0.023529411764705882
        },
        {
          "feature": "vol_of_vol",
          "importance": 12.0,
          "importance_pct": 0.023529411764705882
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
          "feature": "realized_vol_48",
          "importance": 5.0,
          "importance_pct": 0.12195121951219512
        },
        {
          "feature": "ema_fast_slow_spread",
          "importance": 3.0,
          "importance_pct": 0.07317073170731707
        },
        {
          "feature": "adx",
          "importance": 3.0,
          "importance_pct": 0.07317073170731707
        },
        {
          "feature": "garch_vol_rank",
          "importance": 3.0,
          "importance_pct": 0.07317073170731707
        },
        {
          "feature": "hour_sin",
          "importance": 3.0,
          "importance_pct": 0.07317073170731707
        },
        {
          "feature": "kama_slope_fast",
          "importance": 2.0,
          "importance_pct": 0.04878048780487805
        },
        {
          "feature": "di_spread",
          "importance": 2.0,
          "importance_pct": 0.04878048780487805
        },
        {
          "feature": "log_return_12",
          "importance": 2.0,
          "importance_pct": 0.04878048780487805
        },
        {
          "feature": "garch_volatility",
          "importance": 2.0,
          "importance_pct": 0.04878048780487805
        },
        {
          "feature": "garch_vol_ratio",
          "importance": 2.0,
          "importance_pct": 0.04878048780487805
        },
        {
          "feature": "dow_cos",
          "importance": 2.0,
          "importance_pct": 0.04878048780487805
        },
        {
          "feature": "kama_slope",
          "importance": 1.0,
          "importance_pct": 0.024390243902439025
        },
        {
          "feature": "close_ema_slow_ratio",
          "importance": 1.0,
          "importance_pct": 0.024390243902439025
        },
        {
          "feature": "fdi",
          "importance": 1.0,
          "importance_pct": 0.024390243902439025
        },
        {
          "feature": "bb_width",
          "importance": 1.0,
          "importance_pct": 0.024390243902439025
        },
        {
          "feature": "momentum_rank",
          "importance": 1.0,
          "importance_pct": 0.024390243902439025
        },
        {
          "feature": "atr_pct",
          "importance": 1.0,
          "importance_pct": 0.024390243902439025
        },
        {
          "feature": "atr_rank",
          "importance": 1.0,
          "importance_pct": 0.024390243902439025
        },
        {
          "feature": "vol_of_vol",
          "importance": 1.0,
          "importance_pct": 0.024390243902439025
        },
        {
          "feature": "whipsaw_rate",
          "importance": 1.0,
          "importance_pct": 0.024390243902439025
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
      "calibration_rows": 3098,
      "eval_rows": 3098,
      "brier_score_raw": 0.4501650175154183,
      "brier_score_calibrated": 0.431137486568291,
      "log_loss_raw": 0.6405181405733252,
      "log_loss_calibrated": 0.6117463674452751,
      "log_loss_relative_gain": 0.044919528902486086,
      "minimum_relative_gain": 0.01,
      "improved": true,
      "recommended_for_production": true,
      "note": "Measured here; the caller (DirectionModel/EntryModel) swaps this stage onto the isotonic-calibrated estimator for live inference whenever `improved` is True, and leaves it on the raw estimator otherwise - see the model artifact's own `production_calibration` field for what was actually applied to this run."
    },
    "direction": {
      "status": "AVAILABLE",
      "method": "isotonic",
      "calibration_rows": 1826,
      "eval_rows": 1827,
      "brier_score_raw": 0.4999031137308359,
      "brier_score_calibrated": 0.5073130436091113,
      "log_loss_raw": 0.6930502798931198,
      "log_loss_calibrated": 0.7006614473056038,
      "log_loss_relative_gain": -0.010982128762227298,
      "minimum_relative_gain": 0.01,
      "improved": false,
      "recommended_for_production": false,
      "note": "stage-2 output spans only 0.029 of probability; calibrating a score with no usable range amplifies noise into displayed confidence rather than correcting miscalibration, so the raw estimator is kept",
      "raw_probability_span": 0.029043675828260118,
      "minimum_probability_span": 0.2,
      "degenerate_score": true
    },
    "joint": {
      "status": "AVAILABLE",
      "method": "isotonic_per_class",
      "log_loss_relative_gain": 0.0025638324697937176,
      "minimum_relative_gain": 0.01,
      "calibration_rows": 3098,
      "eval_rows": 3098,
      "brier_score_raw": 0.6373075543886576,
      "brier_score_calibrated": 0.6262716645678856,
      "log_loss_raw": 1.0559355328864424,
      "log_loss_calibrated": 1.0532282910812192,
      "improved": false,
      "recommended_for_production": false,
      "note": "Calibrates the joint long/short/no_trade probability directly, on top of whatever per-stage calibration already happened above - corrects residual miscalibration that the product of two independently-calibrated probabilities can still leave behind, which per-stage calibration alone cannot see. Applied only to the reported `probabilities` dict at inference (R3's consistency check, audit logging); trade_probability and direction_given_trade_probability, which R1a/R1b gate on, are never touched by this."
    }
  },
  "production_calibration": {
    "gate": "isotonic",
    "direction": "raw"
  },
  "direction_probability_span": 0.029043675828260118,
  "per_symbol": {
    "SYM0/USDT:USDT": {
      "samples": 1549,
      "accuracy": 0.3763718528082634,
      "balanced_accuracy": 0.34481266282736867
    },
    "SYM1/USDT:USDT": {
      "samples": 1549,
      "accuracy": 0.41252420916720467,
      "balanced_accuracy": 0.3372265153540213
    },
    "SYM2/USDT:USDT": {
      "samples": 1549,
      "accuracy": 0.4176888315041963,
      "balanced_accuracy": 0.34242054167920727
    },
    "SYM3/USDT:USDT": {
      "samples": 1549,
      "accuracy": 0.4306003873466753,
      "balanced_accuracy": 0.34237393379946646
    }
  },
  "architecture": "two_stage_cascade",
  "gate_threshold_sweep": [
    {
      "threshold": 0.3,
      "signals": 5724,
      "precision": 0.6380153738644304,
      "recall": 0.9997262523952916,
      "f1": 0.7789271622053962,
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
      "signals": 5719,
      "precision": 0.6383983213848575,
      "recall": 0.9994525047905831,
      "f1": 0.7791293213828425,
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
      "signals": 5719,
      "precision": 0.6383983213848575,
      "recall": 0.9994525047905831,
      "f1": 0.7791293213828425,
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
      "signals": 5719,
      "precision": 0.6383983213848575,
      "recall": 0.9994525047905831,
      "f1": 0.7791293213828425,
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
      "signals": 5603,
      "precision": 0.6412636087810102,
      "recall": 0.9835751437174924,
      "f1": 0.776361279170268,
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
      "signals": 5478,
      "precision": 0.6453085067542899,
      "recall": 0.9676977826444019,
      "f1": 0.7742854013799145,
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
      "signals": 5006,
      "precision": 0.6492209348781462,
      "recall": 0.8896797153024911,
      "f1": 0.7506640489663934,
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
      "signals": 2392,
      "precision": 0.6634615384615384,
      "recall": 0.43443744867232414,
      "f1": 0.5250620347394541,
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
      "signals": 135,
      "precision": 0.7333333333333333,
      "recall": 0.027101012866137423,
      "f1": 0.05227032734952482,
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
      "signals": 61,
      "precision": 0.7704918032786885,
      "recall": 0.012866137421297564,
      "f1": 0.025309639203015617,
      "meets_min_sample_size": true,
      "average_r": "NOT_AVAILABLE",
      "win_rate": "NOT_AVAILABLE",
      "profit_factor": "NOT_AVAILABLE",
      "expectancy": "NOT_AVAILABLE",
      "net_pnl": "NOT_AVAILABLE",
      "max_drawdown": "NOT_AVAILABLE"
    },
    {
      "threshold": 0.8,
      "signals": 7,
      "precision": 1.0,
      "recall": 0.0019162332329592116,
      "f1": 0.003825136612021858,
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
      "signals": 7,
      "precision": 1.0,
      "recall": 0.0019162332329592116,
      "f1": 0.003825136612021858,
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
  "direction_threshold_sweep": [
    {
      "threshold": 0.5,
      "signals": 3653,
      "signal_rate": 1.0,
      "precision": 0.5130030112236518,
      "recall": 1.0,
      "f1": 0.6781255654061878,
      "side_accuracy": 0.5130030112236518,
      "long_share": 0.42485628250752805,
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
      "signal_rate": 0.0,
      "precision": 0.0,
      "recall": 0.0,
      "f1": 0.0,
      "side_accuracy": 0.0,
      "long_share": 0.0,
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
      "signal_rate": 0.0,
      "precision": 0.0,
      "recall": 0.0,
      "f1": 0.0,
      "side_accuracy": 0.0,
      "long_share": 0.0,
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
      "signal_rate": 0.0,
      "precision": 0.0,
      "recall": 0.0,
      "f1": 0.0,
      "side_accuracy": 0.0,
      "long_share": 0.0,
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
      "signal_rate": 0.0,
      "precision": 0.0,
      "recall": 0.0,
      "f1": 0.0,
      "side_accuracy": 0.0,
      "long_share": 0.0,
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
      "signal_rate": 0.0,
      "precision": 0.0,
      "recall": 0.0,
      "f1": 0.0,
      "side_accuracy": 0.0,
      "long_share": 0.0,
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
      "signal_rate": 0.0,
      "precision": 0.0,
      "recall": 0.0,
      "f1": 0.0,
      "side_accuracy": 0.0,
      "long_share": 0.0,
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
      "signal_rate": 0.0,
      "precision": 0.0,
      "recall": 0.0,
      "f1": 0.0,
      "side_accuracy": 0.0,
      "long_share": 0.0,
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
      "signal_rate": 0.0,
      "precision": 0.0,
      "recall": 0.0,
      "f1": 0.0,
      "side_accuracy": 0.0,
      "long_share": 0.0,
      "meets_min_sample_size": false,
      "average_r": "NOT_AVAILABLE",
      "win_rate": "NOT_AVAILABLE",
      "profit_factor": "NOT_AVAILABLE",
      "expectancy": "NOT_AVAILABLE",
      "net_pnl": "NOT_AVAILABLE",
      "max_drawdown": "NOT_AVAILABLE"
    },
    {
      "threshold": 0.95,
      "signals": 0,
      "signal_rate": 0.0,
      "precision": 0.0,
      "recall": 0.0,
      "f1": 0.0,
      "side_accuracy": 0.0,
      "long_share": 0.0,
      "meets_min_sample_size": false,
      "average_r": "NOT_AVAILABLE",
      "win_rate": "NOT_AVAILABLE",
      "profit_factor": "NOT_AVAILABLE",
      "expectancy": "NOT_AVAILABLE",
      "net_pnl": "NOT_AVAILABLE",
      "max_drawdown": "NOT_AVAILABLE"
    }
  ],
  "recommended_direction_threshold": 0.6,
  "split": {
    "train": {
      "start": "2024-01-10T10:15:00+00:00",
      "end": "2024-01-21T05:10:00+00:00",
      "rows": 12428
    },
    "validation": {
      "start": "2024-01-21T06:05:00+00:00",
      "end": "2024-01-26T15:05:00+00:00",
      "rows": 6196
    },
    "test": {
      "start": "2024-01-26T16:00:00+00:00",
      "end": "2024-02-01T01:55:00+00:00",
      "rows": 6240
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
  "trained_at": "2026-08-22T22:26:30+00:00",
  "git_commit": "eefd3f14e142",
  "rows": 12428,
  "validation_rows": 6196,
  "positive_rate": 0.3760824246311738,
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
    "accuracy": 0.5974822466107166,
    "precision": 0.4166666666666667,
    "recall": 0.11974789915966387,
    "f1": 0.1860313315926893,
    "roc_auc": 0.5695923180592992,
    "pr_auc": 0.42196598806481855,
    "confusion_matrix": {
      "labels": [
        0,
        1
      ],
      "raw": [
        [
          3417,
          399
        ],
        [
          2095,
          285
        ]
      ]
    },
    "class_distribution": {
      "positive": 2380,
      "negative": 3816
    },
    "predicted_positive_rate": 0.1103938024531956,
    "probability_stats": {
      "mean": 0.5080632593424395,
      "median": 0.5303468939082334,
      "std": 0.0842234649400364,
      "min": 0.18112961496235858,
      "max": 0.6832309050216261
    }
  },
  "threshold_sweep": [
    {
      "threshold": 0.3,
      "signals": 5727,
      "precision": 0.41557534485769165,
      "recall": 1.0,
      "f1": 0.5871469100777106,
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
      "signals": 5727,
      "precision": 0.41557534485769165,
      "recall": 1.0,
      "f1": 0.5871469100777106,
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
      "signals": 5727,
      "precision": 0.41557534485769165,
      "recall": 1.0,
      "f1": 0.5871469100777106,
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
      "signals": 5725,
      "precision": 0.4155458515283843,
      "recall": 0.9995798319327731,
      "f1": 0.587045033929673,
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
      "signals": 5467,
      "precision": 0.41832815072251694,
      "recall": 0.9609243697478992,
      "f1": 0.5828979227730343,
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
      "signals": 684,
      "precision": 0.4166666666666667,
      "recall": 0.11974789915966387,
      "f1": 0.1860313315926893,
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
      "signals": 22,
      "precision": 0.5454545454545454,
      "recall": 0.005042016806722689,
      "f1": 0.009991673605328892,
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
      "signals": 3,
      "precision": 0.6666666666666666,
      "recall": 0.0008403361344537816,
      "f1": 0.001678556441460344,
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
        "feature": "garch_vol_rank",
        "importance": 30.0,
        "importance_pct": 0.07042253521126761
      },
      {
        "feature": "garch_volatility",
        "importance": 24.0,
        "importance_pct": 0.056338028169014086
      },
      {
        "feature": "volume_trend",
        "importance": 24.0,
        "importance_pct": 0.056338028169014086
      },
      {
        "feature": "hmm_prob_bull",
        "importance": 20.0,
        "importance_pct": 0.046948356807511735
      },
      {
        "feature": "hour_cos",
        "importance": 19.0,
        "importance_pct": 0.04460093896713615
      },
      {
        "feature": "fdi_delta",
        "importance": 18.0,
        "importance_pct": 0.04225352112676056
      },
      {
        "feature": "kama_slope_fast",
        "importance": 15.0,
        "importance_pct": 0.035211267605633804
      },
      {
        "feature": "fdi",
        "importance": 14.0,
        "importance_pct": 0.03286384976525822
      },
      {
        "feature": "wick_ratio",
        "importance": 14.0,
        "importance_pct": 0.03286384976525822
      },
      {
        "feature": "hmm_prob_sideways",
        "importance": 14.0,
        "importance_pct": 0.03286384976525822
      },
      {
        "feature": "hour_sin",
        "importance": 13.0,
        "importance_pct": 0.03051643192488263
      },
      {
        "feature": "kama_distance",
        "importance": 12.0,
        "importance_pct": 0.028169014084507043
      },
      {
        "feature": "adx",
        "importance": 12.0,
        "importance_pct": 0.028169014084507043
      },
      {
        "feature": "bb_width",
        "importance": 12.0,
        "importance_pct": 0.028169014084507043
      },
      {
        "feature": "atr_pct",
        "importance": 12.0,
        "importance_pct": 0.028169014084507043
      },
      {
        "feature": "realized_vol_12",
        "importance": 11.0,
        "importance_pct": 0.025821596244131457
      },
      {
        "feature": "kama_slope",
        "importance": 10.0,
        "importance_pct": 0.023474178403755867
      },
      {
        "feature": "close_ema_slow_ratio",
        "importance": 10.0,
        "importance_pct": 0.023474178403755867
      },
      {
        "feature": "di_spread",
        "importance": 10.0,
        "importance_pct": 0.023474178403755867
      },
      {
        "feature": "garch_vol_ratio",
        "importance": 10.0,
        "importance_pct": 0.023474178403755867
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
    "calibration_rows": 3098,
    "eval_rows": 3098,
    "brier_score_raw": 0.4861729075193694,
    "brier_score_calibrated": 0.45452028277682976,
    "log_loss_raw": 0.6767651286222772,
    "log_loss_calibrated": 0.6356915320363999,
    "log_loss_relative_gain": 0.06069106525847859,
    "minimum_relative_gain": 0.01,
    "improved": true,
    "recommended_for_production": true,
    "note": "Measured here; the caller (DirectionModel/EntryModel) swaps this stage onto the isotonic-calibrated estimator for live inference whenever `improved` is True, and leaves it on the raw estimator otherwise - see the model artifact's own `production_calibration` field for what was actually applied to this run."
  },
  "production_calibration": "isotonic",
  "split": {
    "train": {
      "start": "2024-01-10T10:15:00+00:00",
      "end": "2024-01-21T05:10:00+00:00",
      "rows": 12428
    },
    "validation": {
      "start": "2024-01-21T06:05:00+00:00",
      "end": "2024-01-26T15:05:00+00:00",
      "rows": 6196
    },
    "test": {
      "start": "2024-01-26T16:00:00+00:00",
      "end": "2024-02-01T01:55:00+00:00",
      "rows": 6240
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
  "trained_at": "2026-08-22T22:26:30+00:00",
  "git_commit": "eefd3f14e142",
  "rows": 7087,
  "validation_rows": 3653,
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
      "mae": 0.010672130482988693,
      "rmse": 0.01335871469839956,
      "r2": -0.013923797444616248,
      "median_absolute_error": 0.009340064921413642,
      "target_stats": {
        "mean": 0.027232459807076344,
        "median": 0.025137053229131498,
        "std": 0.013266672752096106,
        "min": 0.0072481443400179354,
        "max": 0.07867920672903188
      },
      "prediction_stats": {
        "mean": 0.025712677979695044,
        "median": 0.025714318028185516,
        "std": 0.0004125460906344822,
        "min": 0.024894848144613204,
        "max": 0.027082942007109524
      },
      "baseline_rule_based_mae": 0.015782861521266615,
      "beats_rule_based_baseline": true,
      "rail_override_rate": 0.16890227210511907
    },
    "target_sl_pct": {
      "mae": 0.0018607172494949622,
      "rmse": 0.0045958606733586945,
      "r2": -0.1546383639132607,
      "median_absolute_error": 0.00024381470697653522,
      "target_stats": {
        "mean": 0.004205391258373875,
        "median": 0.002574299662628862,
        "std": 0.004277045423977248,
        "min": 0.0017672940407270623,
        "max": 0.04075866587334107
      },
      "prediction_stats": {
        "mean": 0.0025215382759972383,
        "median": 0.0024925155241499147,
        "std": 0.00011356725895281278,
        "min": 0.0023139133522059076,
        "max": 0.0028492700655995563
      },
      "baseline_rule_based_mae": 0.0037978786950372194,
      "beats_rule_based_baseline": true,
      "rail_override_rate": 1.0,
      "beats_rule_based_baseline_note": "the model's prediction is overridden by a hard rail on 100% of validation rows, so this comparison describes an output that rarely reaches the exchange"
    }
  },
  "metrics_source": "target_sl_pct:LGBMRegressor:7ff3b7f37dd0|target_tp_pct:LGBMRegressor:7ff3bb92c550",
  "rail_override_rates": {
    "target_sl_pct": 1.0,
    "target_tp_pct": 0.16890227210511907
  },
  "feature_importance": {
    "target_tp_pct": {
      "status": "AVAILABLE",
      "method": "native_gain_or_split",
      "top_features": [
        {
          "feature": "realized_vol_48",
          "importance": 9.0,
          "importance_pct": 0.09782608695652174
        },
        {
          "feature": "hour_cos",
          "importance": 9.0,
          "importance_pct": 0.09782608695652174
        },
        {
          "feature": "garch_vol_rank",
          "importance": 7.0,
          "importance_pct": 0.07608695652173914
        },
        {
          "feature": "dow_sin",
          "importance": 6.0,
          "importance_pct": 0.06521739130434782
        },
        {
          "feature": "fdi",
          "importance": 5.0,
          "importance_pct": 0.05434782608695652
        },
        {
          "feature": "hour_sin",
          "importance": 5.0,
          "importance_pct": 0.05434782608695652
        },
        {
          "feature": "log_return_48",
          "importance": 4.0,
          "importance_pct": 0.043478260869565216
        },
        {
          "feature": "garch_volatility",
          "importance": 4.0,
          "importance_pct": 0.043478260869565216
        },
        {
          "feature": "garch_vol_ratio",
          "importance": 4.0,
          "importance_pct": 0.043478260869565216
        },
        {
          "feature": "ema_fast_slow_spread",
          "importance": 3.0,
          "importance_pct": 0.03260869565217391
        },
        {
          "feature": "close_ema_slow_ratio",
          "importance": 3.0,
          "importance_pct": 0.03260869565217391
        },
        {
          "feature": "di_spread",
          "importance": 3.0,
          "importance_pct": 0.03260869565217391
        },
        {
          "feature": "fdi_delta",
          "importance": 3.0,
          "importance_pct": 0.03260869565217391
        },
        {
          "feature": "vol_of_vol",
          "importance": 3.0,
          "importance_pct": 0.03260869565217391
        },
        {
          "feature": "hmm_prob_bull",
          "importance": 3.0,
          "importance_pct": 0.03260869565217391
        },
        {
          "feature": "hmm_regime_age",
          "importance": 3.0,
          "importance_pct": 0.03260869565217391
        },
        {
          "feature": "adx",
          "importance": 2.0,
          "importance_pct": 0.021739130434782608
        },
        {
          "feature": "hmm_prob_sideways",
          "importance": 2.0,
          "importance_pct": 0.021739130434782608
        },
        {
          "feature": "dow_cos",
          "importance": 2.0,
          "importance_pct": 0.021739130434782608
        },
        {
          "feature": "kama_slope",
          "importance": 1.0,
          "importance_pct": 0.010869565217391304
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
          "importance": 81.0,
          "importance_pct": 0.1478102189781022
        },
        {
          "feature": "atr_rank",
          "importance": 47.0,
          "importance_pct": 0.08576642335766424
        },
        {
          "feature": "realized_vol_48",
          "importance": 39.0,
          "importance_pct": 0.07116788321167883
        },
        {
          "feature": "garch_vol_rank",
          "importance": 29.0,
          "importance_pct": 0.05291970802919708
        },
        {
          "feature": "vol_of_vol",
          "importance": 28.0,
          "importance_pct": 0.051094890510948905
        },
        {
          "feature": "garch_volatility",
          "importance": 27.0,
          "importance_pct": 0.04927007299270073
        },
        {
          "feature": "kama_distance",
          "importance": 20.0,
          "importance_pct": 0.0364963503649635
        },
        {
          "feature": "fdi",
          "importance": 20.0,
          "importance_pct": 0.0364963503649635
        },
        {
          "feature": "bb_width",
          "importance": 16.0,
          "importance_pct": 0.029197080291970802
        },
        {
          "feature": "hour_cos",
          "importance": 16.0,
          "importance_pct": 0.029197080291970802
        },
        {
          "feature": "realized_vol_12",
          "importance": 14.0,
          "importance_pct": 0.025547445255474453
        },
        {
          "feature": "hour_sin",
          "importance": 13.0,
          "importance_pct": 0.023722627737226276
        },
        {
          "feature": "fdi_delta",
          "importance": 12.0,
          "importance_pct": 0.021897810218978103
        },
        {
          "feature": "dow_sin",
          "importance": 12.0,
          "importance_pct": 0.021897810218978103
        },
        {
          "feature": "hmm_prob_bull",
          "importance": 11.0,
          "importance_pct": 0.020072992700729927
        },
        {
          "feature": "kama_slope",
          "importance": 10.0,
          "importance_pct": 0.01824817518248175
        },
        {
          "feature": "ema_fast_slow_spread",
          "importance": 10.0,
          "importance_pct": 0.01824817518248175
        },
        {
          "feature": "rsi_delta",
          "importance": 10.0,
          "importance_pct": 0.01824817518248175
        },
        {
          "feature": "hmm_prob_sideways",
          "importance": 10.0,
          "importance_pct": 0.01824817518248175
        },
        {
          "feature": "volume_trend",
          "importance": 10.0,
          "importance_pct": 0.01824817518248175
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
      "start": "2024-01-10T10:35:00+00:00",
      "end": "2024-01-21T05:10:00+00:00",
      "rows": 7087
    },
    "validation": {
      "start": "2024-01-21T06:05:00+00:00",
      "end": "2024-01-26T15:05:00+00:00",
      "rows": 3653
    },
    "test": {
      "start": "2024-01-26T16:00:00+00:00",
      "end": "2024-02-01T01:55:00+00:00",
      "rows": 3499
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
  "trained_at": "2026-08-22T22:26:30+00:00",
  "git_commit": "eefd3f14e142",
  "rows": 12428,
  "validation_rows": 6196,
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
    "mae": 0.2645259783487494,
    "rmse": 0.2990021960706067,
    "r2": 0.01831381364023199,
    "median_absolute_error": 0.27453355877661584,
    "target_stats": {
      "mean": 0.3490037573044527,
      "median": 0.3683719183666159,
      "std": 0.301778321129606,
      "min": 0.0,
      "max": 0.9961008428031449
    },
    "prediction_stats": {
      "mean": 0.3933527255208163,
      "median": 0.3876432309474668,
      "std": 0.070926239276973,
      "min": 0.20124097210317354,
      "max": 0.6140430013477226
    },
    "clipped_prediction_rate": 0.0
  },
  "feature_importance": {
    "status": "AVAILABLE",
    "method": "native_gain_or_split",
    "top_features": [
      {
        "feature": "fdi_delta",
        "importance": 133.0,
        "importance_pct": 0.043822075782537065
      },
      {
        "feature": "garch_vol_rank",
        "importance": 127.0,
        "importance_pct": 0.04184514003294893
      },
      {
        "feature": "kama_distance",
        "importance": 115.0,
        "importance_pct": 0.03789126853377265
      },
      {
        "feature": "bb_width",
        "importance": 110.0,
        "importance_pct": 0.036243822075782535
      },
      {
        "feature": "volume_trend",
        "importance": 107.0,
        "importance_pct": 0.035255354200988465
      },
      {
        "feature": "hour_cos",
        "importance": 106.0,
        "importance_pct": 0.034925864909390446
      },
      {
        "feature": "log_return_48",
        "importance": 104.0,
        "importance_pct": 0.0342668863261944
      },
      {
        "feature": "hour_sin",
        "importance": 100.0,
        "importance_pct": 0.032948929159802305
      },
      {
        "feature": "hmm_prob_bull",
        "importance": 98.0,
        "importance_pct": 0.03228995057660626
      },
      {
        "feature": "adx",
        "importance": 97.0,
        "importance_pct": 0.031960461285008235
      },
      {
        "feature": "realized_vol_12",
        "importance": 92.0,
        "importance_pct": 0.030313014827018123
      },
      {
        "feature": "realized_vol_48",
        "importance": 89.0,
        "importance_pct": 0.029324546952224053
      },
      {
        "feature": "wick_ratio",
        "importance": 89.0,
        "importance_pct": 0.029324546952224053
      },
      {
        "feature": "kama_slope",
        "importance": 83.0,
        "importance_pct": 0.027347611202635916
      },
      {
        "feature": "kama_slope_fast",
        "importance": 81.0,
        "importance_pct": 0.026688632619439868
      },
      {
        "feature": "fdi",
        "importance": 81.0,
        "importance_pct": 0.026688632619439868
      },
      {
        "feature": "garch_volatility",
        "importance": 80.0,
        "importance_pct": 0.026359143327841845
      },
      {
        "feature": "vol_of_vol",
        "importance": 80.0,
        "importance_pct": 0.026359143327841845
      },
      {
        "feature": "close_ema_slow_ratio",
        "importance": 78.0,
        "importance_pct": 0.025700164744645797
      },
      {
        "feature": "garch_vol_ratio",
        "importance": 78.0,
        "importance_pct": 0.025700164744645797
      }
    ],
    "shap": {
      "status": "NOT_AVAILABLE",
      "reason": "shap is not an installed project dependency"
    }
  },
  "training_row_filter": "none - every bar, matching the population predict() is asked about",
  "split": {
    "train": {
      "start": "2024-01-10T10:15:00+00:00",
      "end": "2024-01-21T05:10:00+00:00",
      "rows": 12428
    },
    "validation": {
      "start": "2024-01-21T06:05:00+00:00",
      "end": "2024-01-26T15:05:00+00:00",
      "rows": 6196
    },
    "test": {
      "start": "2024-01-26T16:00:00+00:00",
      "end": "2024-02-01T01:55:00+00:00",
      "rows": 6240
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
    "NO_TRADE_OR_FAIL": 10652,
    "SHORT_SUCCESS": 7263,
    "LONG_SUCCESS": 7029
  }
}
```

## Walk-Forward
{'status': 'AVAILABLE', 'method': 'expanding_window', 'n_folds': 4, 'folds': [{'fold': 1, 'train_rows': 3700, 'validation_rows': 3741, 'accuracy': 0.39427960438385456, 'balanced_accuracy': 0.3689388977468448, 'log_loss': 1.1701206374323354, 'macro_f1': 0.35243941922531336}, {'fold': 2, 'train_rows': 7440, 'validation_rows': 3741, 'accuracy': 0.4012296177492649, 'balanced_accuracy': 0.34630703147024183, 'log_loss': 1.102832475246724, 'macro_f1': 0.27868169549911176}, {'fold': 3, 'train_rows': 11180, 'validation_rows': 3741, 'accuracy': 0.41512964448008555, 'balanced_accuracy': 0.3425071842730758, 'log_loss': 1.0538836432692436, 'macro_f1': 0.256842427551877}, {'fold': 4, 'train_rows': 14920, 'validation_rows': 3741, 'accuracy': 0.3918738305265972, 'balanced_accuracy': 0.3352857477342864, 'log_loss': 1.1004010222068765, 'macro_f1': 0.23520389461243996}], 'accuracy_mean': 0.40062817428495057, 'accuracy_std': 0.009049823844828107, 'balanced_accuracy_mean': 0.3482597153061122, 'balanced_accuracy_std': 0.012578324693901088, 'note': 'Each fold fits an independent two-stage cascade (not the production model) purely to measure how much accuracy varies across different time periods.'}

## Backtest
```json
{
  "start": "2024-01-13T22:25:00+00:00",
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
    "trades_per_year": 0.0,
    "per_trade_sharpe": 0.0,
    "annualised_sharpe_from_trades": 0.0,
    "equity_curve_points": 5275.0,
    "nonzero_return_bars": 0.0,
    "calmar_ratio": 0.0,
    "total_fees": 0.0,
    "total_funding": 0.0,
    "liquidations": 0.0
  },
  "signals_generated": 21100,
  "signals_rejected": 21100,
  "rejection_breakdown": {
    "R1B_DIRECTION_CONFIDENCE_TOO_LOW": 18036,
    "R1A_GATE_CONFIDENCE_TOO_LOW": 3064
  },
  "rejection_breakdown_independent": {
    "R1B_DIRECTION_CONFIDENCE_TOO_LOW": 18036,
    "R1A_GATE_CONFIDENCE_TOO_LOW": 3064
  },
  "rule_evaluation_counts": {
    "R0_SYSTEM_HALTED": {
      "reached": 21100,
      "passed": 21100
    },
    "R0_TRADING_DISABLED": {
      "reached": 21100,
      "passed": 21100
    },
    "R0_PORTFOLIO_FULL": {
      "reached": 21100,
      "passed": 21100
    },
    "R0_SYMBOL_ALREADY_OPEN": {
      "reached": 21100,
      "passed": 21100
    },
    "R1A_GATE_CONFIDENCE_TOO_LOW": {
      "reached": 21100,
      "passed": 18036
    },
    "R1B_DIRECTION_CONFIDENCE_TOO_LOW": {
      "reached": 18036,
      "passed": 0
    }
  },
  "oos_disclosure": null,
  "settings_delta": null,
  "risk_guard_transitions": [],
  "halted_at": null,
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
  "signals_generated": 21100,
  "signals_rejected": 21100,
  "rejection_breakdown": {
    "R1B_DIRECTION_CONFIDENCE_TOO_LOW": 18036,
    "R1A_GATE_CONFIDENCE_TOO_LOW": 3064
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
    "SYM2/USDT:USDT": 6237,
    "SYM3/USDT:USDT": 6237,
    "SYM0/USDT:USDT": 6235,
    "SYM1/USDT:USDT": 6235
  },
  "per_symbol_direction_accuracy": {
    "SYM0/USDT:USDT": {
      "samples": 1549,
      "accuracy": 0.3763718528082634,
      "balanced_accuracy": 0.34481266282736867
    },
    "SYM1/USDT:USDT": {
      "samples": 1549,
      "accuracy": 0.41252420916720467,
      "balanced_accuracy": 0.3372265153540213
    },
    "SYM2/USDT:USDT": {
      "samples": 1549,
      "accuracy": 0.4176888315041963,
      "balanced_accuracy": 0.34242054167920727
    },
    "SYM3/USDT:USDT": {
      "samples": 1549,
      "accuracy": 0.4306003873466753,
      "balanced_accuracy": 0.34237393379946646
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
    "5 feature(s) are effectively empty on every row (liquidation_imbalance, ob_imbalance, ob_imbalance_delta, ob_spread_bps, ob_spread_rank) - they occupy the feature contract and force a retrain on every change without contributing anything; remove them or fix the backfill that should be populating them"
  ],
  "HIGH": [
    "Backtest produced only 0 trade(s) from 21100 candidate signal(s) (21100 rejected, top rejection reason: R1B_DIRECTION_CONFIDENCE_TOO_LOW) - win rate/profit factor/Sharpe/expectancy are not statistically meaningful below 30 trades; widen the validation window or run a walk-forward-style backtest across multiple periods before trusting these numbers",
    "Microstructure/derivatives feed(s) sit at their neutral default for nearly every row (likely not being collected): funding_rate, long_short_ratio, open_interest_change, taker_buy_sell_ratio - check data collection before trusting feature importance involving them",
    "the confidence sweep reaches 100.0% accuracy at threshold 0.8 only by predicting a single class - that number is the surviving subset's base rate, not skill, and raising the live threshold toward it would produce a system that never trades"
  ],
  "MEDIUM": [],
  "LOW": [
    "entry isotonic calibration measurably improves log loss and is wired into inference",
    "direction gate (trade vs no-trade) isotonic calibration measurably improves log loss and is wired into inference"
  ]
}
```