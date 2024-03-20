use crate::error::{Error, Result};
use config::ModelConfig;
use ropey::Rope;
use std::fs;
use std::time::Duration;
use std::{path::PathBuf, sync::Arc};
use tracing::debug;
use std::cmp;
use retrieval::{build_model_and_tokenizer, SnippetRetriever};
mod backend;
mod config;
mod document;
mod error;
use tokio::time::Instant;

mod language_id;
pub mod retrieval;
fn get_position_idx(rope: &Rope, row: usize, col: usize) -> Result<usize> {
    Ok(rope.try_line_to_char(row)?
        + col.min(
            rope.get_line(row.min(rope.len_lines().saturating_sub(1)))
                .ok_or(Error::OutOfBoundLine(row))?
                .len_chars()
                .saturating_sub(1),
        ))
}
const MAX_WARNING_REPEAT: Duration = Duration::from_secs(3_600);
pub const NAME: &str = "llm-ls";
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

use tokenizers::TruncationDirection;

async fn setup() -> SnippetRetriever {
    let cache_dir = PathBuf::from(r"idontexist");
    let snippet_retriever = SnippetRetriever::new(cache_dir, ModelConfig::default(), 32, 32)
        .await
        .unwrap();
    snippet_retriever
}

async fn bench_generate_embeddings(content_len: usize) {
    let retriever = setup().await;
    let (model, tokenizer) = build_model_and_tokenizer(
        "intfloat/multilingual-e5-small".to_string(),
        "main".to_string(),
    )
    .await
    .unwrap();
    // let content = "hello world".to_string();
    let contents = fs::read_to_string("./crates/llm-ls/src/retrieval.rs")
        .expect("Should have been able to read the file");
    let repeated_content = &contents[0..cmp::min(content_len, contents.len())];
    let test_sizes = vec![1, 10];
    for test_size in test_sizes.into_iter() {
        let start = Instant::now();
        for _ in 0..test_size {
            let mut encoding = tokenizer.encode(repeated_content, true).unwrap();
            encoding.truncate(512, 1, TruncationDirection::Right);
            retriever
                .generate_embeddings(vec![encoding.clone()], retriever.model.clone())
                .await
                .unwrap();
        }
        println!(
            "Embeded {test_size} text of len {content_len} in {:?}",
            start.elapsed() / test_size

        );
    }
    // b.bench_function("one text", |b| b.iter(|| retriever.generate_embeddings(vec![encoding.clone()], retriever.model.clone())));
    // b.iter(|| retriever.generate_embeddings(vec![encoding], Arc::new(model)));
}

#[tokio::main]
async fn main() {
    bench_generate_embeddings(100).await;
    bench_generate_embeddings(1000).await;
    bench_generate_embeddings(10000).await;
    bench_generate_embeddings(30000).await;

}
