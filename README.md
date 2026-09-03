# S&P 555

**Do the ten largest constituents move the S&P 500?**

Every trading day from 1 January 2016 to 31 December 2025, the S&P 500 members
are ranked by market capitalisation. The top ten become the independent
variables; the target is the index close on the **following** trading day.

---

## Setup

> The `python` on this machine's PATH is a Microsoft Store stub that cannot run.
> The virtual environment must be created with the real interpreter, once.

```powershell
& "C:\Users\jojoj\AppData\Roaming\uv\python\cpython-3.12-windows-x86_64-none\python.exe" -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The environment is already built and populated. Activating it is enough.

If PowerShell blocks the activation script:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

---

## Running the pipeline

Always run from the project root; the modules import as `src.*`.

```powershell
python -m scripts.run_data          # download, rank, assemble the dataset
python -m scripts.build_race        # render the home-page animation
python -m scripts.run_train         # expanding-window ablation A / B / C
python -m scripts.run_experiments   # N sweep and weight vs importance
pytest -q                           # leakage and alignment tests

streamlit run app/Home.py           # dashboard
```

`run_data` caches every download under `data/raw/`, so later runs are fast.
Pass `--force-download` to refetch.

---

## Design

### Variables

| | |
|---|---|
| **X** | market cap of the top ten, columns `x1`…`x10` |
| **y** | S&P 500 close at `t + 1` |
| **Excluded** | the index level at `t`, and anything derived from it |

`x1` is **whatever company is largest on that date**, not a fixed ticker. Across
this window roughly two dozen different firms pass through a block ten wide;
pinning the columns to today's leaders would rewrite the early years.

### Why market cap rather than share price

Ranking by market cap while valuing the column by share price creates a
discontinuity. Two firms crossing in size have no particular relationship
between their share prices, so a swap could jump a column from $150 to $600 for
no economic reason — and with daily re-ranking, the boundary slots swap
constantly.

Valued by market cap, a swap changes the column by almost nothing, because the
two firms are near-identical in size at the moment they cross. The slots are
also monotonic by construction: `x1 >= x2 >= … >= x10`.

### Market cap construction

```
mcap(t) = shares_outstanding_now * Close(t)
```

`Close`, never `Adj Close`. Yahoo's `Close` is split-adjusted but not
dividend-adjusted, which is exactly right here: both factors then sit on today's
share basis. `Adj Close` additionally removes dividends, pushing historical
prices down by roughly a fifth across ten years and corrupting the ranking.
There is a test asserting this.

### Feature sets (ablation)

| Set | Columns | Count |
|---|---|---|
| **A** | `x1`…`x10` | 10 |
| **B** | A + `mom_5`, `mom_20`, `ma_ratio_20`, `share_1`, `hhi` | 15 |
| **C** | B + `DGS10`, `DGS2`, `T10Y2Y`, `gold_return` | 19 |

Trend features are built from `S(t)`, the combined market cap of the block, so
they stay inside the scope of the research question. Rates enter as levels; gold
enters as a return, because its price roughly doubled over the window and a
second trending series would manufacture correlation with an index that also
trended up.

### Validation

Expanding window, one fold per validation year:

| Fold | Train | Validate |
|---|---|---|
| 1 | 2016–2019 | 2020 |
| 2 | 2016–2020 | 2021 |
| 3 | 2016–2021 | 2022 |
| 4 | 2016–2022 | 2023 |
| 5 | 2016–2023 | 2024 |
| 6 | 2016–2024 | 2025 |

A one-row gap separates train from validation so the last training row's target
cannot reach into the validation block. There is no random splitter anywhere in
`src/models/split.py`, deliberately.

---

## Reading the results

### R² is not the headline

The index level carries an enormous trend, so almost any model scores well
against total variance. **The number that matters is `skill_mae`:**

```
skill_mae = model MAE / naive-forecast MAE
```

where the naive forecast is "tomorrow equals today".

- `skill_mae` **< 1** — the model beats doing nothing
- `skill_mae` **> 1** — a constant no-change guess was more accurate

### The extrapolation ceiling

Tree models predict by averaging training targets inside a leaf, so they cannot
return a value above the highest target they were trained on. The index roughly
triples across the window, which means most folds ask them for a number they
cannot produce:

| Validation year | Rows above the training maximum |
|---|---|
| 2020 | 53% |
| 2021 | 98% |
| **2022** | **0%** |
| 2023 | 0% |
| 2024 | 96% |
| 2025 | 56% |

2022 is the control: the one bear market in the window, and the only year where
the target stays inside the range already seen. Model quality tracks this column
more closely than it tracks anything about the features.

This is why `LR` is pinned into `model.included_model_types` and why
`scripts/run_train.py` scores **every model individually on the validation
window**, not just the ensemble AutoGluon selected. AutoGluon picks its ensemble
weights on a tuning split carved from the end of the training block — which sits
inside the range the trees have already seen, so trees look strongest exactly
where the choice is made.

The `Model Comparison` dashboard page exists to show that directly.

---

## Findings

Numbers below are from the run of 3 September 2026 over the full window.

### The top ten explain the index

Regressing the index level on the ten market caps **inside each year**:

| Year | R² | MAPE | index ÷ block |
|---|---|---|---|
| 2016 | 0.954 | 0.79% | 0.592 |
| 2017 | 0.983 | 0.44% | 0.557 |
| 2018 | 0.916 | 0.81% | 0.499 |
| 2019 | 0.970 | 0.67% | 0.485 |
| 2020 | 0.938 | 1.99% | 0.392 |
| 2021 | 0.976 | 0.77% | 0.349 |
| 2022 | 0.943 | 1.36% | 0.357 |
| 2023 | 0.940 | 1.06% | 0.342 |
| 2024 | 0.967 | 0.99% | 0.293 |
| 2025 | 0.978 | 0.84% | 0.267 |

Never below 0.92. The answer to the research question is yes.

### But the relationship drifts

The last column is the coefficient a model has to learn, and it falls
**monotonically by a factor of 2.2** across the decade. The top-ten block grew
6.65× while the index grew 2.97× — the block outgrew the index 2.24 times over.

Consequence: any model fitted on earlier years expects more index per unit of
top-ten market cap than later years deliver, and **over-predicts**. Mean error on
the expanding folds runs +424, +380, +318, +513 points. The single exception is
2022, the only validation year whose values stay inside the range already seen,
where R² reaches 0.74–0.78 while every other fold is negative.

### Two hypotheses that turned out wrong

Both were in the original design and both were falsified by the run:

1. **"A linear model will extrapolate, since the index is a weighted sum."**
   `LinearModel` scored MAE 782 — worse than every tree. One fixed slope cannot
   span a 2.2× drift; the trees at least clamp to their training range, which
   bounds the error instead of projecting the wrong slope indefinitely.
2. **"Normalising the target by the block total will fix the range problem."**
   It made things far worse (skill 60+). Dividing the features by `S(t)` turns
   them into within-block weights that sum to one, destroying the level
   information the target depends on.

Both are recorded here rather than quietly removed, because the design document
asserted them and the data did not agree.

### How many constituents?

| N | Mean R² | SD | Mean MAE | Skill |
|---|---|---|---|---|
| 5 | −0.652 | 1.94 | 327.6 | 9.67 |
| 10 | −0.931 | 1.71 | 343.6 | 10.58 |
| 20 | **−0.095** | 1.82 | **241.4** | **7.35** |

No knee. Accuracy improves with N, and the spread across folds is larger than
the gap between values of N — the validation year matters more than the count.
Ten is defensible as a reporting choice, not as an optimum the data selects.

### Weight against importance

Spearman rank correlation between market-cap weight and permutation importance,
per fold: **−0.17, −0.18, −0.40, n/a, −0.26, −0.06**, with p-values from 0.25 to
0.88.

Not merely non-linear — statistically indistinguishable from no relationship at
all, and negative in sign every time it is measurable. A constituent's size
carries essentially no information about how much the model relies on its slot.

### Forecasting

Every configuration loses to the naive forecast, by factors of roughly 6 to 10.
This is unfavourable by construction: the naive rule is handed today's index
level, which the design deliberately withholds from the model. The result says
the top ten do not pin down the index level as precisely as yesterday's index
does — which was never in doubt — and should not be read as evidence that the
constituents do not matter.

---

## Layout

```
config/config.yaml       every tunable value; nothing is hard-coded in src/
data/raw/                cached downloads
data/processed/          dataset, rankings, concentration -- what the app reads
reports/                 metrics, predictions, importance
assets/top10_race.mp4    home-page animation
logs/                    rotating run logs

src/
  config.py              paths and config loading
  logger.py              console + rotating file logger
  utils.py               seeding, artifact and table IO
  pipeline.py            end-to-end dataset construction
  data/universe.py       constituent list and shares outstanding
  data/load.py           prices, index, gold, FRED
  data/preprocess.py     calendar alignment (forward fill only)
  features/marketcap.py  market caps and the daily ranking
  features/trend.py      momentum and concentration features
  features/macro.py      rate and gold controls
  features/build_features.py  assembly, target, ablation sets
  models/split.py        expanding-window folds
  models/train.py        AutoGluon, per-model scoring
  models/evaluate.py     MAE / MSE / MAPE / MSPE / R² plus baseline
  models/importance.py   weight vs permutation importance

scripts/                 entry points
app/                     Streamlit dashboard (Home + 5 pages)
tests/                   leakage and alignment tests
```

---

## Configuration

Change `config/config.yaml` only.

| To change | Edit |
|---|---|
| Number of constituents | `features.top_n` |
| N sweep values | `features.top_n_sweep` |
| Study window | `data.start_date`, `data.end_date` |
| Prediction horizon | `target.horizon` |
| Validation years | `split.validation_years` |
| Model pool / time budget | `model.included_model_types`, `model.time_limit` |
| Animation smoothness | `dashboard.race_freq`, `race_fps` |

---

## Known limitations

1. **Shares outstanding are a current snapshot.** Companies that repurchased a
   large share of their float have their historical market cap understated.
   Apple is the clearest case: its computed January 2016 market cap comes out
   near \$375bn against an actual figure closer to \$540bn, which distorts the
   ranking in the early years.
2. **Survivorship bias.** The constituent list is today's membership applied
   backwards. Firms that left the index before today are never candidates, even
   though some held top-ten slots during the window.
3. **Dual share classes occupy separate slots.** Alphabet appears as both
   `GOOGL` and `GOOG`, so the "top ten" is really the top ten share classes.
   This matches how the index itself treats them, but it means fewer than ten
   distinct companies are represented on many dates.
4. **No transaction costs, slippage, or trading logic.** This is an explanatory
   study, not a strategy.

---

## Notes

This is coursework, **not investment advice**.

All code, comments, log output and dashboard text are in English by request.
