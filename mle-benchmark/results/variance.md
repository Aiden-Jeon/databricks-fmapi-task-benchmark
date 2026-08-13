# Run-to-run variance (M-track repeats)

Unit = **relative std** (std / |mean|, %) so tasks with different metric scales are comparable. Only cells with n≥2 appear.

## By model — consistency is a model property

| Model | cells (n≥2) | median rel-std | max rel-std |
|---|---|---|---|
| GPT-5.6-sol | 24 | **1.54%** | 9.74% |
| Opus 5 | 24 | **3.20%** | 103.67% |
| GLM 5.2 | 21 | **3.51%** | 100.70% |

## By output shape — a weak predictor, with big exceptions

| Output shape | cells | median rel-std | max rel-std |
|---|---|---|---|
| structured parse | 6 | 6.64% | 18.03% |
| numeric regression | 12 | 4.91% | 103.67% |
| free-text span | 4 | 3.86% | 9.06% |
| closed-set label | 47 | 1.80% | 23.33% |

Structured-output tasks carry the highest median, but the single most variable cell is a plain tabular regression — so output shape explains part of the spread, not all of it.

## Most variable cells

| rel-std | model | task | shape | runs |
|---|---|---|---|---|
| 103.67% | Opus 5 | t1_pubg | numeric regression | 0.02278 → 0.1257 (n=3) |
| 100.70% | GLM 5.2 | t1_pubg | numeric regression | 0.04472 → 0.2673 (n=3) |
| 23.33% | Opus 5 | t2_spooky | closed-set label | 0.2582 → 0.4126 (n=3) |
| 19.52% | GLM 5.2 | t5_bike | numeric regression | 209.8 → 303.4 (n=3) |
| 18.03% | Opus 5 | t25_klue_dp | structured parse | 0.5731 → 0.8199 (n=3) |
| 14.18% | Opus 5 | t20_klue_ner | structured parse | 0.6573 → 0.8506 (n=3) |
| 13.70% | GLM 5.2 | t6_klue_nli | closed-set label | 0.4356 → 0.5644 (n=3) |
| 9.74% | GPT-5.6-sol | t5_bike | numeric regression | 203.5 → 246.5 (n=3) |
| 9.61% | GLM 5.2 | t2_spooky | closed-set label | 0.3557 → 0.426 (n=3) |
| 9.38% | GLM 5.2 | t25_klue_dp | structured parse | 0.6243 → 0.7491 (n=3) |

11 of 69 cells are near-deterministic (rel-std < 0.5%), so high variance is concentrated, not universal.
