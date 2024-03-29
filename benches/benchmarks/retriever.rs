use std::{path::PathBuf, sync::Arc};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
// #[path = "../../src/retrieval.rs"]
// mod retrieval;
use retrieval::{Snippet, SnippetRetriever, build_model_and_tokenizer};
use tokenizers::{
    Encoding, PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer, TruncationDirection,
};

async fn setup() -> SnippetRetriever {
    let cache_dir = PathBuf::from(r"idontexist");
    let snippet_retriever = SnippetRetriever::new(cache_dir, ModelConfig::default(), 32, 32).await.unwrap();
    snippet_retriever
}

async fn bench_generate_embeddings(b: &mut Criterion) {
    let retriever = setup().await;
    let (model, tokenizer) = build_model_and_tokenizer(
        "intfloat/multilingual-e5-small".to_string(),
        "main".to_string(),
    )
    .await
    .unwrap();
    let mut encoding = tokenizer.encode("hello word".to_string(), true)?;
    encoding.truncate(
        312,
        1,
        TruncationDirection::Right,
    );
    b.bench_function("one text", |b| b.iter(|| retriever.generate_embeddings(vec![encoding], Arc::new(model))));
    // b.iter(|| retriever.generate_embeddings(vec![encoding], Arc::new(model)));
}

criterion_group! {
    name = generation_speed;
    config = Criterion::default();
    targets = bench_generate_embeddings
}
