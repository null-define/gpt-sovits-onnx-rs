use crate::cpu_info::get_hw_big_cores;
use lazy_static::lazy_static;
use std::path::Path;

use ort::{
    execution_providers::CPUExecutionProvider,
    session::{Session, builder::GraphOptimizationLevel},
};

use crate::error::GSVError;
lazy_static! {
    pub static ref BIG_CORES: Vec<(usize, u64)> =
        get_hw_big_cores().unwrap_or((0..8).map(|id| (id, 0)).collect());
}

pub fn create_onnx_cpu_session<P: AsRef<Path>>(path: P) -> Result<Session, GSVError> {
    Ok(Session::builder()?
        .with_execution_providers([CPUExecutionProvider::default()
            .with_arena_allocator(true)
            .build()])?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(BIG_CORES.len())?
        .with_memory_pattern(true)?
        .with_prepacking(true)?
        .with_config_entry("session.enable_mem_reuse", "1")?
        .with_independent_thread_pool()?
        .with_intra_op_spinning(true)?
        .commit_from_file(path)?)
}
