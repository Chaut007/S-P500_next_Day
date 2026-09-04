# S&P 555

**Do the ten largest constituents move the S&P 500?**

Every trading day from 1 January 2016 to 31 December 2025, the S&P 500 members
are ranked by market capitalisation. The top ten become the independent
variables; the target is the index close on the **following** trading day.

---

## Setup

Python 3.12 is required. AutoGluon does not yet publish wheels for 3.13+.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Or with [uv](https://github.com/astral-sh/uv):

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
```

<details>
<summary>Windows notes</summary>

If `python` resolves to the Microsoft Store stub, `python -m venv` fails with a
prompt to install from the Store. Create the environment with a real interpreter
instead — `uv venv --python 3.12` is the least painful route.

If PowerShell refuses to run the activation script:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

</details>

---

## Running the pipeline

Always run from the project root; the modules import as `src.*`.

```powershell
python -m scripts.run_data          # download, rank, assemble the dataset
python -m scripts.build_logos       # company marks for the home-page leaderboard
python -m scripts.run_train         # expanding-window ablation A / B / C
python -m scripts.run_experiments   # explanatory power and weight vs importance
python -m scripts.run_tuning        # grid search SVR / XGBoost / LSTM
python -m scripts.run_models        # four models on the 80/20 split + SHAP
python -m scripts.run_models --tuned   # the same run using the tuned settings
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

### Splitting

Two splits, for two different jobs.

**Chronological 80/20** is the headline. The leading 80% of trading days trains,
the trailing 20% tests, and nothing about the test block reaches the models —
not the scalers, not AutoGluon's tuning split, not the LSTM's early stopping.
This is the split the four-model comparison runs on.

**Expanding window** is used only to choose the feature set, one fold per
validation year:

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
| **2023** | **0%** |
| 2024 | 96% |
| 2025 | 56% |

2022 and 2023 are the controls: the two years where the target stays inside the
range already seen. Model quality tracks this column more closely than it tracks
anything about the features — but not perfectly, and the imperfection is
informative. 2022 is the best fold in the study; 2023 is one of the worst. The
ceiling is one of two failure modes, not the whole story; the other is the
coefficient drift documented below.

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

Never below 0.91. The answer to the research question is yes.

### But the relationship drifts

The last column is the coefficient a model has to learn, and it falls **by a
factor of 2.2** across the decade — in nine of the ten year-on-year steps, the
exception being 2021 to 2022, when the bear market lifted it briefly. The
top-ten block grew 6.65× while the index grew 2.97×; the block outgrew the index
2.24 times over.

Consequence: any model fitted on earlier years expects more index per unit of
top-ten market cap than later years deliver, and **over-predicts**. That is the
dominant error wherever the ceiling is not active — 2023 lies entirely inside
the training range and is over-predicted by +249 to +325 points depending on the
feature set, and 2020 by +224 to +331.

Where the index instead runs far above the training range, the extrapolation
ceiling pulls the other way and wins: 2021, 2024 and 2025 come out net
**under**-predicted, by as much as −697 points in 2024. Mean signed error
(predicted − actual) by validation year:

| Set | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|
| A | +331 | −365 | −41 | +325 | −697 | −110 |
| B | +316 | −322 | −24 | +249 | −677 | −108 |
| C | +224 | −311 | +30 | +250 | −631 | −69 |

The two mechanisms are what make the sign flip across folds. R² is positive on
2022 and 2025 for every feature set (0.74–0.80) and on 2020 for set C alone; it
is negative on 2021, 2023 and 2024 throughout.

### Two hypotheses that turned out wrong

Both were in the original design and both were falsified by the run:

1. **"A linear model will extrapolate, since the index is a weighted sum."**
   `LinearModel` scored mean MAE 782, second from last — behind `RandomForest`
   (678), `LightGBM` (685) and `XGBoost` (714), ahead of only `CatBoost` (794).
   It does escape the ceiling: its highest prediction *overshoots* the year's
   actual high in every fold, where every tree model undershoots in every fold.
   Escaping the ceiling only trades undershooting for overshooting. One fixed
   slope cannot span a 2.2× drift, while the trees at least clamp to their
   training range, which bounds the error instead of projecting a wrong slope
   indefinitely.
2. **"Normalising the target by the block total will fix the range problem."**
   It made things far worse (skill 60+). Dividing the features by `S(t)` turns
   them into within-block weights that sum to one, destroying the level
   information the target depends on.

Both are recorded here rather than quietly removed, because the design document
asserted them and the data did not agree.

### Four model families on a chronological 80/20 split

Feature set C at N = 10. Train is the leading 1,994 trading days
(Feb 2016 – Jan 2024); test is the trailing 499 (Jan 2024 – Dec 2025). No model
sees a test row while fitting, scaling or early stopping.

| Model | Test MAE | R² | Skill | Highest prediction | Shortfall |
|---|---|---|---|---|---|
| AutoGluon | **940.8** | −2.37 | 24.6 | 5,133 | **1,799** |
| LSTM | 978.3 | −2.70 | 25.6 | 4,971 | 1,961 |
| XGBoost | 1,147.2 | −4.03 | 30.1 | 4,760 | 2,172 |
| SVR | 2,181.9 | −15.92 | 57.2 | 4,686 | 2,246 |

The training block tops out at **4,796.6** and the test block reaches
**6,932.0**, so **98.2%** of the test rows sit above anything the models were
fitted on. That single number explains the whole table: the MAE ranking is
exactly the ranking of how far each model can reach past its training maximum.

- **SVR** (4,686) and **XGBoost** (4,760) never even reach the training maximum.
  A tree averages the training targets in a leaf; an RBF kernel decays toward
  its bias term as inputs leave the support vectors. Both are capped by
  construction.
- **LSTM** (4,971) and **AutoGluon** (5,133) clear it. The LSTM does so because
  its target is standardised rather than min-max scaled — a `[0, 1]` target
  would have capped the output layer at the training maximum and made 98% of
  the test set unreachable by arithmetic rather than by modelling. AutoGluon
  does so through the linear member of its ensemble.

None of them is remotely useful: the best still loses to the naive forecast by
**24.6×**. The comparison measures the extrapolation ceiling, not fit quality.

### Hyperparameter tuning

Grid search over 88 combinations, scored by MAE on `TimeSeriesSplit` folds
**inside the training block**. The test block takes no part in the search.

| Model | Combinations | Folds | Best settings | CV MAE |
|---|---|---|---|---|
| SVR | 36 | 5 | `C=100, gamma=0.01, epsilon=0.01` | 338.5 |
| XGBoost | 36 | 5 | `max_depth=3, lr=0.03, n_estimators=600, subsample=1.0` | 247.0 |
| LSTM | 16 | 3 | `window=10, hidden=64, layers=2, lr=0.001` | 502.1 |

Scored on the test block, tuning helps two of the three and changes nothing
that matters:

| Model | Default MAE | Tuned MAE | Δ | Tuned skill |
|---|---|---|---|---|
| LSTM | 978.3 | 994.0 | **+15.8** | 26.0× |
| XGBoost | 1,147.2 | 1,087.2 | −59.9 | 28.5× |
| SVR | 2,181.9 | **1,707.2** | **−474.7** | 44.7× |

SVR gains the most — widening `gamma` from `scale` to `0.01` slows the decay
toward its training mean, and its shortfall falls from 2,246 to 1,821. The LSTM
gets marginally worse, which is what a 3-fold CV on 16 candidates buys.

The conclusion does not move. The best tuned model still loses to the naive
forecast by 26×, and the shortfalls stay within a few per cent of where they
were. No value of `max_depth` lets a tree predict above the largest target it
was trained on, and no kernel width stops an RBF from decaying toward its
training mean: the ceiling belongs to the model family, not to its
hyperparameters. AutoGluon is not in this table because it searches its own
model pool inside the time budget it is given.

### Feature importance

TreeSHAP on XGBoost ranks the middle slots above the largest: `x9` (mean |SHAP|
457.7), `x4` (261.2), `x10` (190.5), `x5` (166.1). `x1` does not reach the top
eight. This is the same conclusion the Spearman analysis reaches from the other
direction — a constituent's size says little about how much the model leans on
its slot.

Permutation importance, run across all four models, is **uninformative here and
that is itself the result**. The largest effect any single feature has on any
model is 4.0% of its baseline error, and for AutoGluon the top feature moves MAE
by 0.33 points out of 940.85. Shuffling a column cannot matter much when the
error is made almost entirely of the gap between where a model can reach and
where the index actually went. SVR shows the only visible response, because
decaying toward the training mean makes it more input-sensitive than models that
simply flatten.

### Weight against importance

Spearman rank correlation between market-cap weight and permutation importance,
per fold: **−0.17, −0.18, −0.40, n/a, −0.26, −0.06**, with p-values from 0.25 to
0.88.

Not merely non-linear — statistically indistinguishable from no relationship at
all, and negative in sign every time it is measurable. A constituent's size
carries essentially no information about how much the model relies on its slot.

### Forecasting

Every configuration loses to the naive forecast, by factors ranging from 2.1
(set B on 2022) to 21.7 (set A on 2024). This is unfavourable by construction:
the naive rule is handed today's index
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
app/static/logos/        company marks, served to the home-page leaderboard
logs/                    rotating run logs
```

The marks come from [Simple Icons](https://simpleicons.org) (CC0) via
`scripts.build_logos`, recoloured to the dashboard's foreground. Sixteen of the
twenty-three companies that have held a top-ten slot have a glyph there; the
rest are identified by ticker alone. The marks are trademarks of their owners
and appear here only to identify the companies under study.

```
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
  models/split.py        expanding-window folds and the 80/20 split
  models/train.py        AutoGluon, per-model scoring
  models/zoo.py          AutoGluon / XGBoost / SVR / LSTM behind one interface
  models/explain.py      permutation importance and TreeSHAP
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
| Feature set for the model comparison | `features.best_set` |
| Study window | `data.start_date`, `data.end_date` |
| Prediction horizon | `target.horizon` |
| Train/test proportion | `split.test_ratio` |
| Validation years (ablation) | `split.validation_years` |
| AutoGluon pool / time budget | `model.included_model_types`, `model.time_limit` |
| SVR / XGBoost / LSTM settings | `zoo.svr`, `zoo.xgboost`, `zoo.lstm` |
| Grid search space | `tuning.svr.grid`, `tuning.xgboost.grid`, `tuning.lstm.grid` |
| SHAP sample sizes | `explain.shap_sample`, `explain.shap_background` |
| Leaderboard frames / bars | `dashboard.race_freq`, `dashboard.race_top_n` |

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
