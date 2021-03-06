use criterion::{criterion_group, criterion_main, Criterion};
use dcg::Dcg;

fn bench_additions(c: &mut Criterion) {
    // Set up cells and computations
    let dcg = Dcg::new();

    let a = dcg.cell("something".to_string());
    let b = dcg.cell("borrowed".to_string());

    let concat_ab = || format!("{} {}", dcg.get(a), dcg.get(b));
    let memo = dcg.memo(&concat_ab, &[a, b]);
    let thunk = dcg.thunk(&concat_ab, &[a, b]);

    // Do benchmarking
    let mut group = c.benchmark_group("Concatenation");
    group.bench_function("Memo Concat", |b| b.iter(|| dcg.get(memo)));
    group.bench_function("Thunk Concat", |b| b.iter(|| dcg.get(thunk)));
    group.bench_function("Raw Concat", |b| {
        b.iter(|| format!("{} {}", "something".to_string(), "borrowed".to_string()))
    });
}

criterion_group!(benches, bench_additions);
criterion_main!(benches);
