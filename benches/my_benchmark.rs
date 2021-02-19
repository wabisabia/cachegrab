use criterion::{black_box, criterion_group, criterion_main, Criterion};
use dcg::{Thunk, Cell, Comp};

fn bench_additions(c: &mut Criterion) {
    let a = Cell::new(&1);
    let b = Cell::new(&2);
    let ab = (&a, &b);
    let add = Comp::new(&ab, &|(x, y)| x.value() + y.value());
    let mut group = c.benchmark_group("Addition");
    group.warm_up_time(std::time::Duration::from_secs(10));
    group.sample_size(1000);
    group.measurement_time(std::time::Duration::from_secs(10));
    group.bench_function("DCG Addition", |b| b.iter(|| add.value()));
    group.bench_function("Normal Addition", |b| b.iter(|| black_box(1) + black_box(2)));
}

criterion_group!(benches, bench_additions);
criterion_main!(benches);
