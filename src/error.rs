use hound::Error as HoundError;
use ndarray::ShapeError;
use ort::Error as OrtError;
use std::{
    error::Error,
    fmt::{Display, Formatter, Result as FmtResult},
    io::Error as IoError,
    time::SystemTimeError,
};
#[derive(Debug)]
pub enum GSVError {
    Io(IoError),
    Ort(OrtError),
    Shape(ShapeError),
    SystemTime(SystemTimeError),
    Hound(HoundError),
    AnyHow(anyhow::Error),
    Common(String),
}

impl Error for GSVError {}

impl Display for GSVError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "GSVError: ")?;
        match self {
            Self::Io(e) => Display::fmt(e, f),
            Self::Ort(e) => Display::fmt(e, f),
            Self::Shape(e) => Display::fmt(e, f),
            Self::SystemTime(e) => Display::fmt(e, f),
            Self::Hound(e) => Display::fmt(e, f),
            Self::AnyHow(e) => Display::fmt(e, f),
            Self::Common(e) => Display::fmt(e, f),
        }
    }
}

impl From<OrtError> for GSVError {
    fn from(value: OrtError) -> Self {
        Self::Ort(value)
    }
}

impl From<IoError> for GSVError {
    fn from(value: IoError) -> Self {
        Self::Io(value)
    }
}

impl From<ShapeError> for GSVError {
    fn from(value: ShapeError) -> Self {
        Self::Shape(value)
    }
}

impl From<SystemTimeError> for GSVError {
    fn from(value: SystemTimeError) -> Self {
        Self::SystemTime(value)
    }
}

impl From<HoundError> for GSVError {
    fn from(value: HoundError) -> Self {
        Self::Hound(value)
    }
}

impl From<anyhow::Error> for GSVError {
    fn from(value: anyhow::Error) -> Self {
        Self::AnyHow(value)
    }
}
impl From<String> for GSVError {
    fn from(value: String) -> Self {
        Self::Common(value)
    }
}

impl From<&str> for GSVError {
    fn from(value: &str) -> Self {
        Self::Common(value.into())
    }
}
