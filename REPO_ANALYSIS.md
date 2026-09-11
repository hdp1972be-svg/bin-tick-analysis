# Bin Tick Analysis – Codebase Walkthrough & Improvement Plan

## What this repository is doing

This repo is experimenting with **tick-by-tick microstructure analysis** centered on direction streaks (consecutive up/down price ticks), and using those streaks to derive non-standard pseudo-HFT metrics.

At a high level, the scripts:

1. Read CSV tick data (`timestamp`, `price`).
2. Convert price changes into directional ticks (`UP`/`DOWN`).
3. Segment the stream into streaks (runs of same direction).
4. Compute per-streak metrics such as:
   - streak length frequency,
   - cumulative move per streak,
   - directional asymmetry,
   - burstiness,
   - entropy,
   - transition dynamics between streak lengths.
5. Visualize distributions and regime boundaries to guide strategy ideas (e.g., grid spacing).

## Key files and roles

### `streak_analyzer.py`
Primary analysis script. It combines:

- Data ingestion and streak extraction (`read_csv`).
- Statistical helper functions (burstiness, entropy, mass ratios, asymmetry, etc.).
- Sequence-level analysis (`analyze_streak_runs`, `calculate_transition_matrix`).
- Strategy-oriented heuristics (`calculate_optimal_grid_params`).
- Plotting helpers (ECDF, split violins, annotation overlays).

This is the most complete “core” implementation.

### `streak_analyzer_helpers.py`
Appears to be a **partial extraction** of helper functions from `streak_analyzer.py`, but currently incomplete as a standalone module (uses NumPy/Matplotlib symbols without imports). This suggests refactoring was started but not finalized.

### `test_price_bin_dist_dir11.py` and related `test_price_bin_dist_*`
Prototype scripts that iterate through directories of CSVs and print diagnostics for:

- top-N longest streaks,
- first tick displacement vs total streak displacement,
- simple rebound patterns (A-B-C sequence where B is short counter-streak).

These look like exploratory analysis notebooks converted into scripts.

### `test_direction-analysis.py`
Older simple histogram of streak-length counts with split up/down buckets.

### `download_binance_data.py` (+ `download_random_*.py` variants)
Data acquisition helpers that download Binance daily trade CSVs and then run local streak analysis.

### `websocket2mqqt_bridge.py`
Standalone market-data bridge prototype (CCXT → MQTT). Not integrated with analysis pipeline and has async usage issues for CCXT.

## Observations about code quality and behavior

1. **Strong exploratory value, low consolidation**
   - Many scripts repeat near-identical streak logic with slight variations.
   - “test_*.py” names are used for analysis scripts rather than tests.

2. **Core idea is coherent**
   - The streak-based features are internally consistent and useful for intraday microstructure profiling.
   - Transition matrix + burstiness + directional asymmetry gives a reasonable pseudo-HFT feature set.

3. **Refactor drift is visible**
   - “old” and new versions of functions coexist.
   - Helper extraction is incomplete.
   - Mixed language comments (EN/NL) and placeholder docstrings indicate WIP status.

4. **Validation/reproducibility gaps**
   - No formal unit/integration tests.
   - Few input/output schemas.
   - Hardcoded paths in several scripts reduce portability.

## Suggested improvement plan (practical order)

### Phase 1 – Stabilize structure (quick win)

1. Create package layout:
   - `src/bin_tick_analysis/io.py`
   - `src/bin_tick_analysis/streaks.py`
   - `src/bin_tick_analysis/metrics.py`
   - `src/bin_tick_analysis/plots.py`
   - `src/bin_tick_analysis/cli.py`

2. Keep one canonical implementation of:
   - streak extraction,
   - metric computation,
   - transition analysis.

3. Convert exploratory scripts into CLI entry points:
   - `analyze-file`,
   - `analyze-dir`,
   - `plot-streaks`,
   - `detect-rebounds`.

### Phase 2 – Improve data correctness

1. Standardize timestamp parsing:
   - support timezone-aware ISO and custom formats consistently.
2. Enforce schema checks:
   - required columns,
   - numeric coercion + NaN handling,
   - monotonic timestamp validation.
3. Explicitly define handling for flat ticks (`price == prev_price`) and boundary conditions for final streak closure.

### Phase 3 – Make metrics more robust

1. Add confidence intervals/bootstrap for key metrics.
2. Replace histogram-entropy defaults with adaptive bins (or KDE-based alternatives).
3. Normalize metrics by instrument tick size and volatility regime.
4. Add optional rolling-window analysis to detect regime changes over time.

### Phase 4 – Testing and reproducibility

1. Add synthetic-data unit tests for streak segmentation edge cases.
2. Add regression tests with tiny fixture CSVs.
3. Save outputs in a structured format (Parquet/JSON summaries) for repeatable comparisons.
4. Introduce configuration via YAML/TOML instead of hardcoded constants.

### Phase 5 – Strategy-facing outputs

1. Emit feature tables per time bucket:
   - burstiness,
   - asymmetry,
   - transition persistence,
   - rebound propensity.
2. Score/flag “market mode” labels (trend/range/noisy).
3. Backtest-ready export interface (timestamped features aligned to tradable bars/ticks).

## Immediate concrete fixes I’d prioritize

1. Finalize `streak_analyzer_helpers.py` as a real importable module (or remove it).
2. Remove duplicated `*_old` functions once confidence is established.
3. Rename `test_*.py` analysis scripts to `analysis_*.py` to avoid pytest confusion.
4. Introduce a single reusable `extract_streaks()` function and use it everywhere.
5. Add `README.md` with:
   - expected CSV schema,
   - run commands,
   - metric definitions.

## Bottom line

You already have a useful experimental engine for non-standard microstructure metrics. The main opportunity now is not inventing more metrics, but **consolidating one clean pipeline** so results become comparable, testable, and strategy-usable.
