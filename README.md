# ash-stats

Statistics for product analytics in plain Rust: confidence intervals and significance tests for
changes in counts and rates, robust anomaly detection for daily metrics that grow, and a seasonal
forecast. No dependencies beyond an optional `serde`.

It is the statistics module of **ASH**, the analytics engine I wrote for
[StrinovaHub](https://strinovahub.com) (an esports platform). There it backs every "is this change
real?" badge, the unusual-days report and the forecasts, over ~117k page views held in memory and
answered in milliseconds.

## What's inside

| Function | Answers |
|---|---|
| `wilson(k, n)` | 95% Wilson score interval for a rate (stable at small `n` and at 0 or 100%) |
| `proportion_test(k1, n1, k2, n2)` | Two-proportion z-test: did the conversion rate change? |
| `count_test(now, before)` | Conditional binomial test for two Poisson counts over equal periods: did visits, signups or messages really change? |
| `mean_test(a, b)` | Welch's test on two sample summaries (e.g. time on page) |
| `describe(values)` | n, mean, sd, min, p25, p50, p75, p90, p99, max |
| `median`, `mad`, `quantile` | Robust building blocks |
| `weekly_anomalies(series, threshold, from)` | Days far from what they should have been, trend- and weekday-aware |
| `forecast(series, horizon, period)` | Additive Holt–Winters with grid-searched parameters and an 80% band |
| `trend(series)` | Least-squares slope, absolute and relative to the mean |
| `stars(p)` | `*`, `**`, `***` for p < 0.05, 0.01, 0.001 |

## Example

```rust
// Signups: 48 of 600 visitors this week, 30 of 590 the week before.
let rate = ash_stats::wilson(48.0, 600.0);          // 8.0%, 95% CI 6.1%–10.4%
let test = ash_stats::proportion_test(48.0, 600.0, 30.0, 590.0);
println!("{:.1}% [{:.1}–{:.1}], p = {:.3}", rate.p * 100.0, rate.lo * 100.0, rate.hi * 100.0, test.p);

// Daily visitors: find the unusual days, then forecast two weeks.
let base = ash_stats::weekly_anomalies(&daily, 3.5, 14);
for a in &base.anomalies {
    println!("day {}: {} vs {:.0} expected ({})", a.index, a.value, a.expected, a.kind);
}
let next = ash_stats::forecast(&daily, 14, 7);
```

`cargo run --example daily` builds a synthetic metric with steady growth, quiet weekends and one
launch-day spike:

```
day 60:  836 vs  396 expected (spike, robust z 22.1)
next 14 days: about 6201 (80% band on day 1: 364-524), alpha 0.1 beta 0 gamma 0.1
last week 2782 vs 2487 two weeks before: z 4.05, p 0.0001 ***
```

The spike is found and the growth is not mistaken for one.

## How the anomaly detector works

A plain "mean ± 3σ of the last 28 days" flags every day of a growing product as a spike, and a
fixed weekday profile breaks as soon as traffic grows. Instead, each day gets its own expectation:

- **level**: the median of the seven days before it, which follows the trend and ignores single
  outliers;
- **weekday factor**: the median, over the four weeks before, of that weekday's value against its
  week's mean (at least two weeks are needed, otherwise the day has no expectation);
- **expected** = level × weekday factor;
- **spread**: the MAD of the residuals over the four weeks before (×1.4826, so it estimates σ),
  floored by the Poisson spread √expected, so small numbers don't fire on noise.

A day is an anomaly when it is `threshold` spreads away **and** at least 50% above (or 50% below)
the expectation. In ASH consecutive anomalous days are then grouped into one episode and explained
by the segment that moved most.

## Forecast

Additive Holt–Winters with a weekly season. α, β and γ are chosen by grid search on the one-step
in-sample error; the band widens with the horizon from the residual spread. Series shorter than
two seasons fall back to their recent average. The in-sample MAPE is returned so a UI can say how
much to trust it.

## Tests

```
cargo test
cargo test --no-default-features   # without serde
```

The tests include `growth_is_not_an_anomaly`: a steadily growing series with a weekly rhythm must
produce no anomalies.

## License

MIT © 2026 Daniil Neretin
