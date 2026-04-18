use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("path is empty")]
    EmptyPath,

    #[error("invalid constraints: {0}")]
    InvalidConstraints(String),

    #[error("solver failed: {0}")]
    SolverFailed(String),
}

pub type Result<T> = std::result::Result<T, Error>;
