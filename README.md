# TradeGAN

**A reproduction of the Fin-GAN methodology of Vuletić & Cont (2023), applied to half-day
TCS excess returns against the Nifty IT sector index.**

> ### Attribution
> The model, the name *Fin-GAN*, the ForGAN-based architecture and the economics-driven
> generator loss are **not mine**. They are the work of Milena Vuletić and Rama Cont,
> *"Fin-GAN: Forecasting and Classifying Financial Time Series via Generative Adversarial
> Networks"* (2023). `src/TradeGAN.py` began as their reference implementation
> (`src/FinGAN.py`) and was renamed in commit `2396561`, which also stripped the original
> attribution header — that was a mistake and this file exists to correct it.
>
> What is mine is the application to a new market (TCS vs `^CNXIT`), the run in
> `Trade-GAN_results/`, and the reading of those results below — including the parts that
> do not support the method.

---

## What it does

A conditional GAN (LSTM generator and discriminator, ForGAN architecture) is trained to
forecast **half-day excess log-returns**: TCS's open-to-close and close-to-open returns
minus the Nifty IT index's, on an interleaved open/close price series, each leg clipped to
±15% before subtraction (`src/TradeGAN.py:50-95`).

The point of Fin-GAN is the generator's objective. Instead of only minimising forecast
error, it adds **economics-motivated terms** on top of the adversarial BCE loss:

| term | what it rewards |
| --- | --- |
| `PnL` | realised profit of trading the forecast's sign |
| `SR` | Sharpe ratio of that PnL |
| `STD` | penalises PnL volatility |
| `MSE` | ordinary squared forecast error |

Trading on a *sign* is not differentiable, so the sign is replaced by a scaled `tanh`
surrogate — `torch.tanh(100 * predicted_return)` (`src/TradeGAN.py:800`) — which keeps the
PnL and Sharpe terms trainable.

The four term weights (α, β, γ, δ) are meant to be set automatically by **gradient-norm
matching** against the BCE gradient, rather than hand-tuned (`src/TradeGAN.py:723-863`).
See [Known issues](#known-issues) — the implementation of this is buggy.

## Results

One completed run: TCS, 100 GAN epochs / 500 LSTM epochs, all 10 permitted loss-term
combinations. Every figure below is a column of
`Trade-GAN_results/Results/results.csv`, on the **same held-out 261-sample test set**.

| objective | test Sharpe | test RMSE | vs MSE baseline |
| --- | --- | --- | --- |
| `MSE` (baseline) | **−0.96** | 0.00966 | — |
| `BCE` | −0.94 | 0.00965 | ~0% |
| `PnL SR` | +0.76 | 0.00743 | −23% |
| `PnL MSE` | +0.82 | 0.00648 | **−33%** |
| `PnL MSE SR` | +0.85 | 0.00685 | −29% |
| `PnL` | +1.15 | 0.00653 | −32% |
| **`PnL MSE STD`** | **+1.31** | **0.00648** | **−33%** |
| `PnL STD` | +1.67 | 0.12479 | 13× worse |
| `SR` | +1.67 | 0.47092 | 49× worse |
| `SR MSE` | **+2.00** | 0.10053 | 10× worse |

**The finding.** Training for profit rather than accuracy is what moves Sharpe: both
error-only objectives (`MSE`, `BCE`) end up **negative**, and every objective containing a
`PnL` or `SR` term is positive. That is the paper's claim, and it reproduces here.

**The catch, and it matters.** The best Sharpe (`SR MSE`, +2.00) costs a test RMSE **10×
worse than the baseline** — a model that trades well while forecasting badly. The
objectives that win on *both* axes are the PnL-plus-MSE family, of which
**`PnL MSE STD` is the best compromise: Sharpe +1.31 with RMSE 33% below baseline.**
Quoting the +2.00 Sharpe and the −33% RMSE together would be a cherry-pick across two
different, non-comparable runs; they are different rows of the table.

## Known issues

Disclosed rather than quietly fixed, because they bound what the numbers above mean.

1. **The gradient-norm weights are not correct L2 norms.** Of the five per-loss gradient
   accumulators, only `SR_norm` takes its square root *after* summing squared
   per-parameter norms (`src/TradeGAN.py:806-813`). `BCE_norm`, `PnL_norm`, `MSE_norm` and
   `STD_norm` all take it *inside* the per-parameter loop (`:815-851`) — a nested running
   square root. Since `BCE_norm` is the shared numerator of all four ratios, **none of α,
   β, γ, δ is a correctly computed gradient-norm ratio.** The ratio *structure* matches
   the paper; the norms feeding it do not. The weights were still derived from gradient
   magnitudes rather than hand-picked, which is the claim the results rest on.
2. **No directional-accuracy metric exists in this repo.** The `Pos mn` / `Neg mn` columns
   are the fraction of the *generated* distribution that is positive — a sign-bias
   statistic about the model's own output, not a hit rate against realised signs. Do not
   read them as accuracy.
3. **Nothing runs out of the box.** `src/app.py` hardcodes absolute paths from the
   original author's machine, `requirements.txt` pins no versions, and there are **no
   tests and no CI**.
4. `trash2.ipynb` is the pre-rename development notebook and still carries the upstream
   author's `@author:` docstring; `src/__pycache__/FinGAN.cpython-312.pyc` carries it in
   compiled form. Both are retained deliberately for now as the provenance record.

## Layout

```
src/TradeGAN.py        3,778 lines: data prep, generator/discriminator,
                       10 loss-combination training loops, LSTM baseline, evaluation
src/data_maker.py      yfinance price fetch
src/app.py             driver (hardcoded paths — edit before running)
TradeGAN_steps/        10 numbered files: the incremental development log
Trade-GAN_results/     results.csv, per-objective PnL series, plots, .pth checkpoints
stocks-etfs-list.csv   201 rows of ticker -> sector-index metadata (no prices)
TradeGAN.pdf           write-up; predates this README and its framing is corrected above
```

## Data

`stocks-etfs-list.csv` is ticker → sector-index metadata only (200 data rows, no price
columns), and prices are fetched at runtime via `yfinance` into a gitignored `data/`
directory that is absent from git's index.

One exception, disclosed rather than glossed: **`trash.ipynb` does redistribute a small
fragment of real market data.** It holds only 143 characters of source but 3.3 KB of
saved cell *output*, including a truncated DataFrame repr with genuine ZENSARTECH.NS
dates and Adj Open/Close prices spanning 2010-01-04 to 2021-12-30. The notebook is
tracked and not gitignored, so "this repo redistributes no market data" would be false
as an absolute claim.

## Reference

Vuletić, M. and Cont, R. (2023). *Fin-GAN: Forecasting and Classifying Financial Time
Series via Generative Adversarial Networks.* Please cite their work, not this repository,
for the method.
