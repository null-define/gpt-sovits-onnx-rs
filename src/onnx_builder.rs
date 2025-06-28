use std::path::Path;

use ort::{
    execution_providers::CPUExecutionProvider,
    inputs,
    session::{Session, builder::GraphOptimizationLevel},
    value::TensorRef,
};

use crate::error::GSVError;

pub fn create_onnx_cpu_session<P: AsRef<Path>>(path: P) -> Result<Session, GSVError>{
    Ok(Session::builder()?
        .with_execution_providers([CPUExecutionProvider::default()
            .with_arena_allocator(true)
            .build()])?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(8)?
        .commit_from_file(path)?)
}
