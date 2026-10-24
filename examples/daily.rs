//! A synthetic daily metric: steady growth, a weekly rhythm (quiet weekends) and one launch-day
//! spike. Finds the spike without flagging the growth, then forecasts the next two weeks.
//!     cargo run --example daily
fn main() {
    let weekday = [1.10, 1.15, 1.10, 1.05, 1.00, 0.75, 0.85];
    let mut series: Vec<f64> = (0..70)
        .map(|d| ((200.0 + 3.0 * d as f64) * weekday[d % 7]).round())
        .collect();
    series[60] *= 2.2; // launch day

    let base = ash_stats::weekly_anomalies(&series, 3.5, 14);
    for a in &base.anomalies {
        println!(
            "day {:>2}: {:>4.0} vs {:>4.0} expected ({}, robust z {:.1})",
            a.index, a.value, a.expected, a.kind, a.score
        );
    }

    let f = ash_stats::forecast(&series, 14, 7);
    let total: f64 = f.mean.iter().sum();
    println!(
        "next 14 days: about {:.0} (80% band on day 1: {:.0}-{:.0}), alpha {} beta {} gamma {}",
        total, f.low[0], f.high[0], f.alpha, f.beta, f.gamma
    );

    // the week before holds the launch spike, so compare with the one before it
    let (last, prev): (f64, f64) = (series[63..70].iter().sum(), series[49..56].iter().sum());
    let t = ash_stats::count_test(last, prev);
    println!(
        "last week {last} vs {prev} two weeks before: z {:.2}, p {:.4} {}",
        t.z,
        t.p,
        "*".repeat(ash_stats::stars(t.p) as usize)
    );
}
