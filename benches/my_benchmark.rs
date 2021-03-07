use criterion::{black_box, criterion_group, criterion_main, Criterion};
use dcg::Dcg;
use rand::seq::IteratorRandom;

fn bench_additions(c: &mut Criterion) {
    // Set up cells and computations
    let dcg = Dcg::new();

    let needle = dcg.cell("a".to_string());
    let haystack = dcg.cell("the quick brown fox jumped over the lazy dog".to_string());

    let remove_needles = || {
        let needle = needle.get().chars().next().unwrap();
        haystack.get().chars().filter(|c| *c == needle).collect()
    };

    let memo = dcg.memo(&remove_needles, &[needle, haystack]);
    let thunk = dcg.thunk(&remove_needles, &[needle, haystack]);

    // Do benchmarking
    let mut group = c.benchmark_group("filter random letter");

    let population = "aaaaaaaaaaaaaaaaaaaab";

    group.bench_function("memo'd", |b| {
        b.iter(|| {
            needle.set(String::from(
                population.chars().choose(&mut rand::thread_rng()).unwrap(),
            ));
            memo.query();
        })
    });

    group.bench_function("thunk", |b| {
        b.iter(|| {
            needle.set(String::from(
                population.chars().choose(&mut rand::thread_rng()).unwrap(),
            ));
            thunk.query();
        })
    });

    group.bench_function("raw", |b| {
        b.iter(|| {
            let needle = black_box(
                String::from(population.chars().choose(&mut rand::thread_rng()).unwrap())
                    .chars()
                    .next()
                    .unwrap(),
            );
            black_box(
                "the quick brown fox jumped over the lazy dog"
                    .to_string()
                    .chars()
                    .filter(|c| *c == needle)
                    .collect::<String>(),
            );
        })
    });
}

criterion_group!(benches, bench_additions);
criterion_main!(benches);
