use std::{
    collections::{HashMap, HashSet},
    rc::Rc,
};

use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use dcg::Dcg;
use rand::{
    distributions::Uniform, prelude::Distribution, prelude::SliceRandom, rngs::SmallRng,
    SeedableRng,
};

fn internals(c: &mut Criterion) {
    let dcg = Dcg::new();

    let cell = dcg.cell(1);

    let cell1 = cell.clone();
    let get_cell = Rc::new(move || cell1.get());
    let memo = dcg.memo(get_cell.clone(), &[cell.idx]);
    let thunk = dcg.thunk(get_cell, &[cell.idx]);

    let mut internals = c.benchmark_group("Internals");

    internals.bench_function("query cell", |b| b.iter(|| cell.query()));
    internals.bench_function("get cell", |b| b.iter(|| cell.get()));
    internals.bench_function("set cell same", |b| b.iter(|| cell.set(1)));
    internals.bench_function("set cell changed", |b| {
        b.iter_batched(
            || {
                cell.set(1);
            },
            |()| cell.set(2),
            BatchSize::SmallInput,
        )
    });

    internals.bench_function("query thunk", |b| b.iter(|| thunk.query()));
    internals.bench_function("get thunk", |b| b.iter(|| thunk.get()));

    internals.bench_function("query memo same", |b| b.iter(|| memo.query()));
    internals.bench_function("query memo changed", |b| {
        b.iter_batched(
            || {
                cell.set(if cell.query() == 1 { 2 } else { 2 });
            },
            |()| memo.query(),
            BatchSize::SmallInput,
        )
    });
    internals.bench_function("get memo same", |b| b.iter(|| memo.get()));
    internals.bench_function("get memo changed", |b| {
        b.iter_batched(
            || {
                cell.set(if cell.query() == 1 { 2 } else { 1 });
            },
            |()| memo.get(),
            BatchSize::SmallInput,
        )
    });
}

fn filter_random_letter(c: &mut Criterion) {
    let dcg = Dcg::new();

    let needle = dcg.cell('c');
    let haystack = dcg.cell("the quick brown fox jumped over the lazy dog");

    let needle1 = needle.clone();
    let haystack1 = haystack.clone();
    let remove_needles = Rc::new(move || {
        let needle = needle1.get();
        haystack1
            .get()
            .chars()
            .filter(|c| *c == needle)
            .collect::<String>()
    });

    let sizes = [2, 10, 100, 1000];
    let max_size = sizes[sizes.len() - 1];
    let mut population = vec!['a'; max_size];
    population[max_size - 1] = 'b';

    {
        let mut memo_group = c.benchmark_group("Filter Random Letter: Memo'd");

        let mut rng = SmallRng::seed_from_u64(123);

        let memo = dcg.memo(remove_needles.clone(), &[needle.idx, haystack.idx]);

        for size in sizes.iter() {
            memo_group.bench_function(BenchmarkId::from_parameter(size), |b| {
                b.iter_batched(
                    || {
                        population[(max_size - size)..max_size]
                            .choose(&mut rng)
                            .unwrap()
                    },
                    |c| {
                        needle.set(*c);
                        memo.query();
                    },
                    BatchSize::SmallInput,
                )
            });
        }
    }

    {
        let mut thunk_group = c.benchmark_group("Filter Random Letter: Thunk");

        let mut rng = SmallRng::seed_from_u64(123);

        let thunk = dcg.thunk(remove_needles, &[needle.idx, haystack.idx]);

        for size in sizes.iter() {
            thunk_group.bench_function(BenchmarkId::from_parameter(size), |b| {
                b.iter_batched(
                    || {
                        population[(max_size - size)..max_size]
                            .choose(&mut rng)
                            .unwrap()
                    },
                    |c| {
                        needle.set(*c);
                        thunk.query();
                    },
                    BatchSize::SmallInput,
                )
            });
        }
    }

    {
        let mut raw_group = c.benchmark_group("Filter Random Letter: Raw");

        let mut rng = SmallRng::seed_from_u64(123);

        for size in sizes.iter() {
            raw_group.bench_function(BenchmarkId::from_parameter(size), |b| {
                b.iter_batched(
                    || {
                        population[(max_size - size)..max_size]
                            .choose(&mut rng)
                            .unwrap()
                    },
                    |needle| {
                        black_box(
                            "the quick brown fox jumped over the lazy dog"
                                .chars()
                                .filter(|c| *c == *needle)
                                .collect::<String>(),
                        );
                    },
                    BatchSize::SmallInput,
                )
            });
        }
    }
}

fn depth_first_search() {
    let dcg = Dcg::new();
    let v = 100;
    let mut rng = SmallRng::seed_from_u64(123);
    let precision = 2;
    let scale = 10u32.pow(precision);
    let dist = Uniform::from(0..scale);
    let density = 0.5;
    let mut graph = HashMap::<_, HashSet<_>>::with_capacity(v);
    for i in 0..v {
        for j in 0..v {
            if dist.sample(&mut rng) <= (density * scale as f32) as u32 {
                graph
                    .entry(i)
                    .and_modify(|neighbours| {
                        neighbours.insert(j);
                    })
                    .or_default();
            }
        }
    }
}

criterion_group!(benches, filter_random_letter, internals);
criterion_main!(benches);
