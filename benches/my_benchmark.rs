use cachegrab::{buffer, incremental::Incremental, memo, thunk, Dcg};
use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use rand::{prelude::SliceRandom, rngs::SmallRng, SeedableRng};

fn internals(c: &mut Criterion) {
    let dcg = Dcg::new();
    let var = dcg.var(1);
    let thunk = thunk!(dcg, var);
    let memo = memo!(dcg, var);
    let buffer = buffer!(dcg, var);

    let mut internals = c.benchmark_group("Internals");

    internals.bench_function("read var", |b| b.iter(|| var.read()));
    internals.bench_function("write var same", |b| b.iter(|| var.write(1)));
    internals.bench_function("write var changed", |b| {
        b.iter_batched(
            || {
                var.write(1);
            },
            |_| var.write(2),
            BatchSize::SmallInput,
        )
    });
    internals.bench_function("modify var same", |b| b.iter(|| var.modify(|x| *x)));
    internals.bench_function("modify var changed", |b| {
        b.iter_batched(
            || {
                var.write(1);
            },
            |_| var.modify(|x| *x + 1),
            BatchSize::SmallInput,
        )
    });
    internals.bench_function("modify var vs read-write", |b| {
        b.iter_batched(
            || {
                var.write(1);
            },
            |_| var.write(var.read() + 1),
            BatchSize::SmallInput,
        )
    });

    internals.bench_function("read thunk", |b| b.iter(|| thunk.read()));
    internals.bench_function("read memo same", |b| b.iter(|| memo.read()));
    internals.bench_function("read memo changed", |b| {
        b.iter_batched(
            || {
                var.modify(|&mut x| if x == 1 { 2 } else { 1 });
            },
            |_| memo.read(),
            BatchSize::SmallInput,
        )
    });
    internals.bench_function("read buffer same", |b| b.iter(|| buffer.read()));
    internals.bench_function("read buffer changed", |b| {
        b.iter_batched(
            || {
                var.modify(|&mut x| if x == 1 { 2 } else { 1 });
            },
            |_| buffer.read(),
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
        let mut thunk_group = c.benchmark_group("Filter Random Letter: Thunk");

        let mut rng = SmallRng::seed_from_u64(123);

        let thunk = thunk!(dcg, (needle, haystack) => {
            haystack
                .chars()
                .filter(|c| *c == needle)
                .collect::<String>()
        });

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
        let mut memo_group = c.benchmark_group("Filter Random Letter: Memo");

        let mut rng = SmallRng::seed_from_u64(123);

        let memo = memo!(dcg, (needle, haystack) => {
            haystack
                .chars()
                .filter(|c| *c == needle)
                .collect::<String>()
        });

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
        let mut buffer_group = c.benchmark_group("Filter Random Letter: Buffer");

        let mut rng = SmallRng::seed_from_u64(123);

        let buffer = buffer!(dcg, (needle, haystack) => {
            haystack
                .chars()
                .filter(|c| *c == needle)
                .collect::<String>()
        });

        for size in sizes.iter() {
            buffer_group.bench_function(BenchmarkId::from_parameter(size), |b| {
                b.iter_batched(
                    || {
                        population[(max_size - size)..max_size]
                            .choose(&mut rng)
                            .unwrap()
                    },
                    |c| {
                        needle.write(*c);
                        buffer.read();
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
