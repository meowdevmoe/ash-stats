//! Statistics for product analytics, from ASH (the analytics engine of StrinovaHub): intervals and
//! tests for changes and rates, robust baselines for anomalies in daily metrics, a seasonal
//! forecast, and distribution summaries. Plain functions on slices, no allocation-heavy machinery.
//!
//! ```
//! // Signups: 48 of 600 visitors this week, 30 of 590 the week before. Is it a real change?
//! let now = ash_stats::wilson(48.0, 600.0);
//! let test = ash_stats::proportion_test(48.0, 600.0, 30.0, 590.0);
//! assert!(now.lo < 0.08 && now.hi > 0.08);
//! assert!(test.p < 0.05);
//! ```

/// Standard normal CDF (Abramowitz–Stegun 7.1.26 through erf, |error| < 1.5e-7).
pub fn normal_cdf(z: f64) -> f64 {
    let x = z.abs() / std::f64::consts::SQRT_2;
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    if z >= 0.0 {
        (1.0 + y) / 2.0
    } else {
        (1.0 - y) / 2.0
    }
}

fn two_sided(z: f64) -> f64 {
    (2.0 * (1.0 - normal_cdf(z.abs()))).clamp(0.0, 1.0)
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct Interval {
    pub p: f64,
    pub lo: f64,
    pub hi: f64,
}

/// Wilson score interval for a proportion k/n at 95%.
pub fn wilson(k: f64, n: f64) -> Interval {
    if n <= 0.0 {
        return Interval {
            p: 0.0,
            lo: 0.0,
            hi: 0.0,
        };
    }
    let z = 1.959964;
    let p = k / n;
    let d = 1.0 + z * z / n;
    let c = (p + z * z / (2.0 * n)) / d;
    let h = z * ((p * (1.0 - p)) / n + z * z / (4.0 * n * n)).sqrt() / d;
    Interval {
        p,
        lo: (c - h).max(0.0),
        hi: (c + h).min(1.0),
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct Test {
    pub z: f64,
    pub p: f64,
}

/// Two-proportion z-test: is k1/n1 different from k2/n2? Two-sided.
pub fn proportion_test(k1: f64, n1: f64, k2: f64, n2: f64) -> Test {
    if n1 <= 0.0 || n2 <= 0.0 {
        return Test { z: 0.0, p: 1.0 };
    }
    let pool = (k1 + k2) / (n1 + n2);
    let se = (pool * (1.0 - pool) * (1.0 / n1 + 1.0 / n2)).sqrt();
    if se == 0.0 {
        return Test { z: 0.0, p: 1.0 };
    }
    let z = (k1 / n1 - k2 / n2) / se;
    Test { z, p: two_sided(z) }
}

/// Two counts over periods of the same length (Poisson rates): the conditional binomial test,
/// normal approximation with continuity correction.
pub fn count_test(now: f64, before: f64) -> Test {
    let n = now + before;
    if n < 4.0 {
        return Test { z: 0.0, p: 1.0 };
    }
    let diff = ((now - n / 2.0).abs() - 0.5).max(0.0);
    let z = (now - n / 2.0).signum() * diff / (n / 4.0).sqrt();
    Test { z, p: two_sided(z) }
}

/// Welch's test on two samples' summaries, normal approximation (large samples).
pub fn mean_test(a: &Summary, b: &Summary) -> Test {
    if a.n < 2 || b.n < 2 {
        return Test { z: 0.0, p: 1.0 };
    }
    let se = (a.sd * a.sd / a.n as f64 + b.sd * b.sd / b.n as f64).sqrt();
    if se == 0.0 {
        return Test { z: 0.0, p: 1.0 };
    }
    let z = (a.mean - b.mean) / se;
    Test { z, p: two_sided(z) }
}

pub fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let pos = (sorted.len() - 1) as f64 * q;
    let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
    sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f64)
}

pub fn median(values: &[f64]) -> f64 {
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    quantile(&v, 0.5)
}

/// Median absolute deviation, scaled to estimate a standard deviation.
pub fn mad(values: &[f64]) -> f64 {
    let m = median(values);
    1.4826 * median(&values.iter().map(|v| (v - m).abs()).collect::<Vec<_>>())
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct Summary {
    pub n: usize,
    pub mean: f64,
    pub sd: f64,
    pub min: f64,
    pub p25: f64,
    pub p50: f64,
    pub p75: f64,
    pub p90: f64,
    pub p99: f64,
    pub max: f64,
}

pub fn describe(values: &[f64]) -> Summary {
    let mut s = values.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let n = s.len();
    let mean = if n > 0 {
        s.iter().sum::<f64>() / n as f64
    } else {
        0.0
    };
    let sd = if n > 1 {
        (s.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt()
    } else {
        0.0
    };
    Summary {
        n,
        mean,
        sd,
        min: s.first().copied().unwrap_or(0.0),
        p25: quantile(&s, 0.25),
        p50: quantile(&s, 0.5),
        p75: quantile(&s, 0.75),
        p90: quantile(&s, 0.9),
        p99: quantile(&s, 0.99),
        max: s.last().copied().unwrap_or(0.0),
    }
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct Anomaly {
    pub index: usize,
    pub value: f64,
    pub expected: f64,
    pub low: f64,
    pub high: f64,
    pub score: f64,
    pub kind: &'static str,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct Baseline {
    pub expected: Vec<Option<f64>>,
    pub low: Vec<Option<f64>>,
    pub high: Vec<Option<f64>>,
    pub anomalies: Vec<Anomaly>,
}

/// Days far from what they should have been. The expectation follows the trend: the level is the
/// median of the seven days before, scaled by the weekday's usual share (the median, over the four
/// weeks before, of that weekday's value against its week's mean). The spread is the MAD of the
/// residuals over the four weeks before, floored by the Poisson spread so small numbers do not fire
/// on noise. An anomaly is `threshold` spreads away and at least half again (or half) the expected.
pub fn weekly_anomalies(series: &[f64], threshold: f64, from: usize) -> Baseline {
    let n = series.len();
    let mut expected: Vec<Option<f64>> = vec![None; n];
    for i in 7..n {
        let level = median(&series[i - 7..i]);
        let factors: Vec<f64> = (1..=4)
            .filter_map(|k| {
                let j = i.checked_sub(7 * k)?;
                let lo = j.saturating_sub(3);
                let hi = (j + 4).min(n);
                let mean = series[lo..hi].iter().sum::<f64>() / (hi - lo) as f64;
                (mean > 0.0).then(|| series[j] / mean)
            })
            .collect();
        // without two weeks behind it a day has no expectation
        if factors.len() >= 2 {
            expected[i] = Some(level * median(&factors));
        }
    }
    let mut b = Baseline {
        expected: vec![],
        low: vec![],
        high: vec![],
        anomalies: vec![],
    };
    for i in 0..n {
        let Some(base) = expected[i] else {
            b.expected.push(None);
            b.low.push(None);
            b.high.push(None);
            continue;
        };
        let residuals: Vec<f64> = (i.saturating_sub(28)..i)
            .filter_map(|j| expected[j].map(|e| series[j] - e))
            .collect();
        let spread = if residuals.len() >= 7 {
            mad(&residuals)
        } else {
            0.0
        }
        .max(base.max(1.0).sqrt())
        .max(1.0);
        let (low, high) = (
            (base - threshold * spread).max(0.0),
            base + threshold * spread,
        );
        b.expected.push(Some(base));
        b.low.push(Some(low));
        b.high.push(Some(high));
        let score = (series[i] - base) / spread;
        let relative = if base > 0.0 {
            (series[i] - base).abs() / base
        } else {
            f64::INFINITY
        };
        if i >= from && score.abs() >= threshold && relative >= 0.5 {
            b.anomalies.push(Anomaly {
                index: i,
                value: series[i],
                expected: base,
                low,
                high,
                score,
                kind: if score > 0.0 { "spike" } else { "drop" },
            });
        }
    }
    b
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct Forecast {
    pub mean: Vec<f64>,
    pub low: Vec<f64>,
    pub high: Vec<f64>,
    pub alpha: f64,
    pub beta: f64,
    pub gamma: f64,
    /// in-sample mean absolute percentage error of one-step predictions
    pub mape: Option<f64>,
}

struct Fit {
    level: f64,
    trend: f64,
    seasons: Vec<f64>,
    sse: f64,
    residuals: usize,
    mape: Option<f64>,
}

fn holt_winters(series: &[f64], period: usize, alpha: f64, beta: f64, gamma: f64) -> Fit {
    let c1 = &series[..period];
    let c2 = &series[period..2 * period];
    let avg = (c1.iter().sum::<f64>() + c2.iter().sum::<f64>()) / (2 * period) as f64;
    let mut seasons: Vec<f64> = (0..period)
        .map(|i| ((c1[i] - avg) + (c2[i] - avg)) / 2.0)
        .collect();
    let mut level = c1.iter().sum::<f64>() / period as f64;
    let mut trend = (c2.iter().sum::<f64>() - c1.iter().sum::<f64>()) / (period * period) as f64;
    let (mut sse, mut residuals, mut ape, mut ape_n) = (0.0, 0usize, 0.0, 0usize);
    for (t, &y) in series.iter().enumerate() {
        let s = seasons[t % period];
        let predicted = level + trend + s;
        if t >= period {
            sse += (y - predicted).powi(2);
            residuals += 1;
            if y > 0.0 {
                ape += (y - predicted).abs() / y;
                ape_n += 1;
            }
        }
        let prev = level;
        level = alpha * (y - s) + (1.0 - alpha) * (level + trend);
        trend = beta * (level - prev) + (1.0 - beta) * trend;
        seasons[t % period] = gamma * (y - level) + (1.0 - gamma) * s;
    }
    Fit {
        level,
        trend,
        seasons,
        sse,
        residuals,
        mape: (ape_n > 0).then(|| ape / ape_n as f64),
    }
}

/// Additive Holt–Winters with a weekly season, parameters by grid search on one-step error, and an
/// approximate 80% band from the in-sample residuals widening with the horizon. Series shorter than
/// two seasons carry their recent average forward.
pub fn forecast(series: &[f64], horizon: usize, period: usize) -> Forecast {
    let n = series.len();
    if n < period * 2 {
        let tail = &series[n.saturating_sub(7)..];
        let s = describe(tail);
        return Forecast {
            mean: vec![s.mean; horizon],
            low: vec![(s.mean - 1.28 * s.sd).max(0.0); horizon],
            high: vec![s.mean + 1.28 * s.sd; horizon],
            alpha: 0.0,
            beta: 0.0,
            gamma: 0.0,
            mape: None,
        };
    }
    let mut best = (0.3, 0.05, 0.2);
    let mut fit = holt_winters(series, period, best.0, best.1, best.2);
    for alpha in [0.1, 0.2, 0.3, 0.5, 0.7] {
        for beta in [0.0, 0.02, 0.05, 0.1] {
            for gamma in [0.05, 0.1, 0.2, 0.4] {
                let f = holt_winters(series, period, alpha, beta, gamma);
                if f.sse < fit.sse {
                    best = (alpha, beta, gamma);
                    fit = f;
                }
            }
        }
    }
    let sd = (fit.sse / (fit.residuals.saturating_sub(3).max(1)) as f64).sqrt();
    let (mut mean, mut low, mut high) = (vec![], vec![], vec![]);
    for h in 1..=horizon {
        let m = (fit.level + h as f64 * fit.trend + fit.seasons[(n + h - 1) % period]).max(0.0);
        let band = 1.28 * sd * (1.0 + (h - 1) as f64 * best.0 * best.0).sqrt();
        mean.push(m);
        low.push((m - band).max(0.0));
        high.push(m + band);
    }
    Forecast {
        mean,
        low,
        high,
        alpha: best.0,
        beta: best.1,
        gamma: best.2,
        mape: fit.mape,
    }
}

/// Least-squares slope per step, and the slope as a share of the mean.
pub fn trend(series: &[f64]) -> (f64, f64) {
    let n = series.len();
    if n < 3 {
        return (0.0, 0.0);
    }
    let mx = (n - 1) as f64 / 2.0;
    let my = series.iter().sum::<f64>() / n as f64;
    let (mut num, mut den) = (0.0, 0.0);
    for (i, y) in series.iter().enumerate() {
        num += (i as f64 - mx) * (y - my);
        den += (i as f64 - mx).powi(2);
    }
    let slope = if den > 0.0 { num / den } else { 0.0 };
    (slope, if my != 0.0 { slope / my } else { 0.0 })
}

/// How a significance reads: 3 for p < 0.001, 2 for p < 0.01, 1 for p < 0.05, else 0.
pub fn stars(p: f64) -> u8 {
    if p < 0.001 {
        3
    } else if p < 0.01 {
        2
    } else if p < 0.05 {
        1
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normal_and_tests() {
        assert!((normal_cdf(1.96) - 0.975).abs() < 1e-3);
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-9);
        let w = wilson(10.0, 100.0);
        assert!(
            (w.lo - 0.0552).abs() < 1e-3 && (w.hi - 0.1744).abs() < 1e-3,
            "{w:?}"
        );
        assert!(proportion_test(60.0, 100.0, 40.0, 100.0).p < 0.01);
        assert!(proportion_test(51.0, 100.0, 49.0, 100.0).p > 0.5);
        assert!(count_test(200.0, 100.0).p < 0.001);
        assert!(count_test(10.0, 9.0).p > 0.5);
    }

    #[test]
    fn anomalies_and_forecast() {
        // a weekly pattern with one spike on day 40
        let mut s: Vec<f64> = (0..56)
            .map(|i| [100.0, 110.0, 120.0, 115.0, 105.0, 60.0, 50.0][i % 7])
            .collect();
        s[40] = 400.0;
        let b = weekly_anomalies(&s, 3.5, 0);
        assert_eq!(
            b.anomalies.iter().map(|a| a.index).collect::<Vec<_>>(),
            vec![40]
        );
        assert_eq!(b.anomalies[0].index, 40);
        let f = forecast(&s[..35], 7, 7);
        for (i, m) in f.mean.iter().enumerate() {
            let want = [100.0, 110.0, 120.0, 115.0, 105.0, 60.0, 50.0][(35 + i) % 7];
            assert!((m - want).abs() < 15.0, "day {i}: {m} vs {want}");
        }
    }

    #[test]
    fn growth_is_not_an_anomaly() {
        let s: Vec<f64> = (0..60)
            .map(|i| 20.0 * 1.05f64.powi(i) * [1.0, 1.1, 1.2, 1.1, 1.0, 0.7, 0.6][i as usize % 7])
            .collect();
        assert!(weekly_anomalies(&s, 3.5, 0).anomalies.is_empty());
    }

    #[test]
    fn summaries() {
        let d = describe(&[1.0, 2.0, 3.0, 4.0, 100.0]);
        assert_eq!(d.p50, 3.0);
        assert_eq!(d.max, 100.0);
        assert!((median(&[5.0, 1.0, 3.0]) - 3.0).abs() < 1e-9);
        assert_eq!(trend(&[1.0, 2.0, 3.0, 4.0]).0, 1.0);
    }
}
