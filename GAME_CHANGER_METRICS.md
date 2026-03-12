# Game-Changer Ideas: Non-Traditional Tick Metrics

This list is tailored to the current codebase patterns in:

- `streak_analyzer.py` (streak sequence, transition matrix, interarrival, burstiness, asymmetry)
- `test_price_bin_dist_dir11.py` (A-B-C rebound motif logic)
- `download_binance_data.py` and `test_direction-analysis.py` (baseline streak counting)

---

## 1) Reversal Friction Index (RFI)
**What it captures:** How much movement is "wasted" before trend continuation after a short pullback.

**Definition (using your A-B-C motif):**
\[
RFI = \operatorname{median}\left(\frac{|\Delta_B|}{|\Delta_C|+\epsilon}\right)
\]
for motifs where B is short counter-streak and C resumes A direction.

**Why this can matter:** High RFI implies expensive continuation (choppy); low RFI implies cheap continuation (clean momentum).

---

## 2) Continuation Convexity (CCX)
**What it captures:** Whether longer streaks produce superlinear or sublinear move growth.

**Definition:** Fit
\[
\log(E[\Delta \mid L=s]) = a + b\log(s)
\]
Then `b>1` indicates convex continuation, `b<1` diminishing returns.

**Why this can matter:** A direct gauge for whether adding depth to trend-following entries is rewarded.

---

## 3) Streak Hazard Curve (SHC)
**What it captures:** Probability that streak ends at next tick given it already reached length `k`.

**Definition:**
\[
h(k)=P(L=k\mid L\ge k)
\]

**Why this can matter:** Real-time stop/exit adjustment per current streak age.

---

## 4) Directional Entropy Gap (DEG)
**What it captures:** Structural difference in uncertainty between up and down streak move distributions.

**Definition:**
\[
DEG = H(\Delta_{up}) - H(\Delta_{down})
\]
where `H` can reuse your entropy function per side.

**Why this can matter:** Signals asymmetric order-flow quality and side-specific predictability.

---

## 5) Tail Regime Persistence (TRP)
**What it captures:** How sticky extreme-move regimes are once entered.

**Definition:** classify each streak move into normal/burst/extreme (p90/p99). Build transition matrix over regime labels and track:
\[
TRP = P(\text{extreme}_{t+1}\mid \text{extreme}_{t})
\]

**Why this can matter:** Helps gate risk during shock clusters.

---

## 6) Interarrival-Conditioned Drift (ICD)
**What it captures:** Whether direction persistence depends on event pace.

**Definition:** bucket interarrival times (fast/medium/slow quantiles) and compute continuation probability within each bucket.

**Why this can matter:** Distinguishes "fast toxic flow" from low-urgency flow.

---

## 7) Elasticity of Move-per-Tick (EMT)
**What it captures:** Efficiency of additional streak ticks.

**Definition:**
\[
EMT(s)=\frac{E[\Delta/L\mid L=s+1]-E[\Delta/L\mid L=s]}{E[\Delta/L\mid L=s] + \epsilon}
\]

**Why this can matter:** Identifies where continuation gets less efficient and profit-taking should dominate.

---

## 8) Anti-Persistence Shock Score (APSS)
**What it captures:** Mean-reverting snap risk after unusually long streaks.

**Definition:** for top-q% longest streaks, measure immediate counter-streak move ratio:
\[
APSS = \operatorname{median}\left(\frac{|\Delta_{next,opp}|}{|\Delta_{long}|+\epsilon}\right)
\]

**Why this can matter:** Pairs naturally with your top-longest streak analysis in `test_price_bin_dist_dir11.py`.

---

## 9) Regime-Weighted Asymmetry (RWA)
**What it captures:** Directional asymmetry that only appears in specific volatility/move regimes.

**Definition:** compute asymmetry separately in normal/burst/extreme bins and aggregate with time-varying regime weights.

**Why this can matter:** Avoids averaging away edge that exists only in tails.

---

## 10) Streak Sequence Compressibility (SSC)
**What it captures:** Hidden repeat structure in streak-length sequence.

**Definition:** compress the streak-length sequence (e.g., LZ complexity proxy). Lower complexity = higher repeatability.

**Why this can matter:** Model selection switch: deterministic/templated flow vs near-random flow.

---

## 11) Directional Liquidity Stress Proxy (DLSP)
**What it captures:** Per-direction price displacement normalized by event tempo.

**Definition:**
\[
DLSP_{dir} = \operatorname{median}\left(\frac{\Delta_{dir}}{\sum iat_{streak}+\epsilon}\right)
\]

**Why this can matter:** Fast large displacements imply thin book/urgent flow on that side.

---

## 12) Transition Surprise Index (TSI)
**What it captures:** Regime breaks in streak-length transitions.

**Definition:** Compare rolling transition matrix to long-run baseline via KL/Jensen-Shannon divergence.

**Why this can matter:** Early warning of microstructure change before PnL drift is obvious.

---

## A practical "starter 4" to implement first

1. **SHC** (easy, directly from `streak_sequence`)
2. **RFI** (already close to your rebound logic)
3. **ICD** (you already parse interarrival)
4. **TSI** (extends existing transition matrix)

These four are high signal-per-effort and can be built without changing your current data model.

---

## Minimal implementation map vs current code

- Use `read_csv()` outputs in `streak_analyzer.py`:
  - `streak_sequence` for SHC/TSI/SSC
  - `price_moves_up` + `price_moves_down` for DEG/RWA
  - `interarrival` + streak boundaries for ICD/DLSP
- Reuse existing helpers:
  - `entropy_of_deltas()` for DEG
  - `calculate_transition_matrix()` as base for TSI
  - percentile regime logic (`compute_regimes`, `calculate_mass_ratios`) for TRP/RWA
- Reuse A-B-C pattern pass from `test_price_bin_dist_dir11.py` for RFI/APSS.

---

## Why these are "non-traditional"

They are not classical TA indicators (RSI/MACD/etc.). They are **microstructure-native**, built from:
- event-time behavior,
- run-length dynamics,
- conditional transition structure,
- rebound path geometry,
- entropy/complexity shifts.

That makes them suitable as *state variables* for execution timing, inventory skew, or dynamic grid spacing in the spirit of your existing work.
