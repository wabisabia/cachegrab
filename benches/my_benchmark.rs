use criterion::{black_box, criterion_group, criterion_main, Criterion};
use dcg::Dcg;
use rand::{rngs::SmallRng, seq::IteratorRandom, SeedableRng};

fn bench_additions(c: &mut Criterion) {
    // Set up cells and computations
    let dcg = Dcg::new();

    let needle = dcg.cell('c');
    let haystack = dcg.cell("the quick brown fox jumped over the lazy dog");

    let remove_needles = || {
        let needle = needle.get();
        haystack
            .get()
            .chars()
            .filter(|c| *c == needle)
            .collect::<String>()
    };

    let memo = dcg.memo(&remove_needles, &[needle.idx, haystack.idx]);
    let thunk = dcg.thunk(&remove_needles, &[needle.idx, haystack.idx]);

    // Do benchmarking

    {
        let mut group = c.benchmark_group("internal");

        let mut rng = SmallRng::seed_from_u64(123);

        group.bench_function("query cell", |b| b.iter(|| needle.query()));
        group.bench_function("get cell", |b| b.iter(|| needle.get()));
        group.bench_function("set cell", |b| {
            b.iter(|| needle.set(('a'..'z').choose(&mut rng).unwrap()))
        });
        group.bench_function("query thunk", |b| b.iter(|| thunk.query()));
        group.bench_function("get thunk", |b| b.iter(|| thunk.get()));
        group.bench_function("query memo", |b| b.iter(|| memo.query()));
        group.bench_function("get memo", |b| b.iter(|| memo.get()));
    }

    {
        let mut group = c.benchmark_group("filter random letter");

        let population = "aaaaab".chars();

        let mut rng = SmallRng::seed_from_u64(123);

        group.bench_function("memo'd", |b| {
            b.iter(|| {
                needle.set(population.clone().choose(&mut rng).unwrap());
                memo.query();
            })
        });
        group.bench_function("thunk", |b| {
            b.iter(|| {
                needle.set(population.clone().choose(&mut rng).unwrap());
                thunk.query();
            })
        });
        group.bench_function("raw", |b| {
            b.iter(|| {
                let needle = population.clone().choose(&mut rng).unwrap();
                black_box(
                    "the quick brown fox jumped over the lazy dog"
                        .chars()
                        .filter(|c| *c == needle)
                        .collect::<String>(),
                );
            })
        });
    }
}

criterion_group!(benches, bench_additions);
criterion_main!(benches);
