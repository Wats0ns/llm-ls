use criterion::criterion_main;

mod benchmarks;

criterion_main! {
    benchmarks::retriever::generation_speed,
}
