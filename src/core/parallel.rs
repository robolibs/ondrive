//! The thread pool the sampling controllers spread their rollouts over.

use rayon::prelude::*;
use std::sync::OnceLock;

/// Most threads the pool runs unless `ONDRIVE_THREADS` asks for more.
const MAX_THREADS: usize = 8;

static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();

/// The pool: `ONDRIVE_THREADS` threads if set, else the cores available up
/// to `MAX_THREADS`, leaving the rest to the process the controller runs in.
pub fn pool() -> &'static rayon::ThreadPool {
    POOL.get_or_init(|| {
        let threads = std::env::var("ONDRIVE_THREADS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|n| *n > 0)
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()).min(MAX_THREADS));
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|i| format!("ondrive-{i}"))
            .build()
            .unwrap_or_else(|_| rayon::ThreadPoolBuilder::new().num_threads(1).build().expect("a one-thread pool"))
    })
}

/// `f` of every index in `0..n` on the pool, returned in index order, so
/// the result does not depend on how the work was shared out.
pub fn map_indexed<T: Send>(n: usize, f: impl Fn(usize) -> T + Sync + Send) -> Vec<T> {
    let pool = pool();
    let chunk = (n / (4 * pool.current_num_threads())).max(1);
    pool.install(|| (0..n).into_par_iter().with_min_len(chunk).map(f).collect())
}
