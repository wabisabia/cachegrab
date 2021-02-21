use criterion::{black_box, criterion_group, criterion_main, Criterion};
use dcg::{Cell, Comp, Thunk};

fn bench_additions(c: &mut Criterion) {
    // Set up cells and computations
    let a = Cell::new(&1);
    let b = Cell::new(&2);
    let ab = (&a, &b);
    let add = Comp::new(&ab, &|(x, y)| x.value() + y.value());

    // Do benchmarking
    let mut group = c.benchmark_group("Addition");
    group.bench_function("DCG Addition", |b| b.iter(|| add.value()));
    group.bench_function("Normal Addition", |b| {
        b.iter(|| black_box(1) + black_box(2))
    });
}

criterion_group!(benches, bench_additions);
criterion_main!(benches);
