use knf_rs::compute_fbank;
use log::debug;
use ndarray::{
    Array, Array1, Array2, ArrayBase, ArrayD
};
use ort::{
    inputs,
    session::Session,
    value::{TensorRef},
};

use crate::GSVError;

pub struct SvModel {
    sv_session: Session,
}

impl SvModel {
    pub fn new(sv_session: Session) -> Self {
        Self { sv_session }
    }

    pub fn infer(&mut self, audio_16k: &Array1<f32>) -> Result<ArrayD<f32>, GSVError> {
        let audio_vec: Vec<f32> = audio_16k.to_vec();
        let features = compute_fbank(&audio_vec).unwrap();
        debug!("SV features shape: {:?}", features.shape());
        let input_features = TensorRef::from_array_view(features.view());

        let outputs = self.sv_session.run(inputs![
            "audio_feature" => input_features?,
        ])?;

        let output_tensor = outputs["sv_emb"].try_extract_array::<f32>()?.into_owned();

        Ok(output_tensor)
    }
}
