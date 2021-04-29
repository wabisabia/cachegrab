use cachegrab::{memo, thunk, Dcg, Incremental};
use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use rand::{prelude::SliceRandom, rngs::SmallRng, SeedableRng};

fn internals(c: &mut Criterion) {
    let dcg = Dcg::new();
    let cell = dcg.var(1);
    let memo = memo!(dcg, cell, cell);
    let thunk = thunk!(dcg, cell, cell);

    let mut internals = c.benchmark_group("Internals");

    internals.bench_function("read cell", |b| b.iter(|| cell.read()));
    internals.bench_function("write cell same", |b| b.iter(|| cell.write(1)));
    internals.bench_function("write cell changed", |b| {
        b.iter_batched(
            || {
                cell.write(1);
            },
            |_| cell.write(2),
            BatchSize::SmallInput,
        )
    });
    internals.bench_function("modify cell same", |b| b.iter(|| cell.modify(|x| *x)));
    internals.bench_function("modify cell changed", |b| {
        b.iter_batched(
            || {
                cell.write(1);
            },
            |_| cell.modify(|x| *x + 1),
            BatchSize::SmallInput,
        )
    });
    internals.bench_function("modify cell vs read-write", |b| {
        b.iter_batched(
            || {
                cell.write(1);
            },
            |_| cell.write(cell.read() + 1),
            BatchSize::SmallInput,
        )
    });

    internals.bench_function("read thunk", |b| b.iter(|| thunk.read()));

    internals.bench_function("read memo same", |b| b.iter(|| memo.read()));
    internals.bench_function("read memo changed", |b| {
        b.iter_batched(
            || {
                cell.write(if cell.read() == 1 { 2 } else { 1 });
            },
            |_| memo.read(),
            BatchSize::SmallInput,
        )
    });
}

fn filter_random_letter(c: &mut Criterion) {
    let dcg = Dcg::new();

    let needle = dcg.var('a');
    let haystack = dcg.var("the quick brown fox jumped over the lazy dog");

    let sizes = [2, 10, 100, 1000];
    let max_size = sizes[sizes.len() - 1];
    let mut population = vec!['a'; max_size];
    population[max_size - 1] = 'b';

    {
        let mut memo_group = c.benchmark_group("Filter Random Letter: Memo'd");

        let mut rng = SmallRng::seed_from_u64(123);

        let memo = memo!(
            dcg,
            haystack
                .chars()
                .filter(|c| *c == needle)
                .collect::<String>(),
            needle,
            haystack
        );

        for size in sizes.iter() {
            memo_group.bench_function(BenchmarkId::from_parameter(size), |b| {
                b.iter_batched(
                    || {
                        population[(max_size - size)..max_size]
                            .choose(&mut rng)
                            .unwrap()
                    },
                    |c| {
                        needle.write(*c);
                        memo.read();
                    },
                    BatchSize::SmallInput,
                )
            });
        }
    }

    {
        let mut thunk_group = c.benchmark_group("Filter Random Letter: Thunk");

        let mut rng = SmallRng::seed_from_u64(123);

        let thunk = thunk!(
            dcg,
            haystack
                .chars()
                .filter(|c| *c == needle)
                .collect::<String>(),
            needle,
            haystack
        );

        for size in sizes.iter() {
            thunk_group.bench_function(BenchmarkId::from_parameter(size), |b| {
                b.iter_batched(
                    || {
                        population[(max_size - size)..max_size]
                            .choose(&mut rng)
                            .unwrap()
                    },
                    |c| {
                        needle.write(*c);
                        thunk.read();
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

// fn depth_first_search() {
//     let dcg = Dcg::new();
//     let v = 100;
//     let mut rng = SmallRng::seed_from_u64(123);
//     let precision = 2;
//     let scale = 10u32.pow(precision);
//     let dist = Uniform::from(0..scale);
//     let density = 0.5;
//     let mut graph = HashMap::<_, HashSet<_>>::with_capacity(v);
//     for i in 0..v {
//         for j in 0..v {
//             if dist.sample(&mut rng) <= (density * scale as f32) as u32 {
//                 graph
//                     .entry(i)
//                     .and_modify(|neighbours| {
//                         neighbours.insert(j);
//                     })
//                     .or_default();
//             }
//         }
//     }
//     println!("{:?}", graph);
// }

criterion_group!(benches, internals, filter_random_letter);
criterion_main!(benches);
