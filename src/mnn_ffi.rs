// src/mnn_ffi.rs
use libc;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int};
use std::ptr;

#[repr(C)]
pub struct MNNModel {
    _private: [u8; 0], // Opaque type
}

unsafe extern "C" {
    fn create_mnn_model(
        sovits_path: *const c_char,
        ssl_path: *const c_char,
        t2s_encoder_path: *const c_char,
        t2s_fs_decoder_path: *const c_char,
        t2s_s_decoder_path: *const c_char,
        num_layers: usize,
    ) -> *mut MNNModel;

    fn destroy_mnn_model(model: *mut MNNModel);

    fn run_ssl(
        model: *mut MNNModel,
        ref_audio_data: *const f32,
        ref_audio_size: usize,
        output_data: *mut *mut f32,
        output_size: *mut usize,
    ) -> c_int;

    fn run_inference(
        model: *mut MNNModel,
        ref_seq_data: *const i64,
        ref_seq_size: usize,
        ref_bert_data: *const f32,
        ref_bert_size: usize,
        text_seq_data: *const i64,
        text_seq_size: usize,
        text_bert_data: *const f32,
        text_bert_size: usize,
        ssl_content_data: *const f32,
        ssl_content_size: usize,
        ref_audio_data: *const f32,
        ref_audio_size: usize,
        output_data: *mut *mut f32,
        output_size: *mut usize,
    ) -> c_int;
}

unsafe fn ptr_to_vec<T: Copy>(ptr: *mut T, len: usize) -> Vec<T> {
    if ptr.is_null() || len == 0 {
        return Vec::new();
    }
    let slice = std::slice::from_raw_parts(ptr, len);
    let vec = slice.to_vec();
    libc::free(ptr as *mut libc::c_void);
    vec
}

pub struct MNNModelWrapper {
    ptr: *mut MNNModel,
}

impl MNNModelWrapper {
    pub fn new(
        sovits_path: &str,
        ssl_path: &str,
        t2s_encoder_path: &str,
        t2s_fs_decoder_path: &str,
        t2s_s_decoder_path: &str,
        num_layers: usize,
    ) -> Result<Self, crate::error::GSVError> {
        let sovits_path = CString::new(sovits_path)?;
        let ssl_path = CString::new(ssl_path)?;
        let t2s_encoder_path = CString::new(t2s_encoder_path)?;
        let t2s_fs_decoder_path = CString::new(t2s_fs_decoder_path)?;
        let t2s_s_decoder_path = CString::new(t2s_s_decoder_path)?;

        unsafe {
            let ptr = create_mnn_model(
                sovits_path.as_ptr(),
                ssl_path.as_ptr(),
                t2s_encoder_path.as_ptr(),
                t2s_fs_decoder_path.as_ptr(),
                t2s_s_decoder_path.as_ptr(),
                num_layers,
            );
            if ptr.is_null() {
                return Err(crate::error::GSVError::from("Failed to create MNN model"));
            }
            Ok(MNNModelWrapper { ptr })
        }
    }

    pub fn run_ssl(&self, ref_audio_data: &[f32]) -> Result<Vec<f32>, crate::error::GSVError> {
        let mut output_data = ptr::null_mut();
        let mut output_size = 0;
        
        unsafe {
            let ret = run_ssl(
                self.ptr,
                ref_audio_data.as_ptr(),
                ref_audio_data.len(),
                &mut output_data,
                &mut output_size,
            );
            if ret != 0 {
                return Err(crate::error::GSVError::from("Failed to run SSL inference"));
            }
            Ok(ptr_to_vec(output_data, output_size))
        }
    }

    pub fn run_inference(
        &self,
        ref_seq: &[i64],
        ref_bert: &[f32],
        text_seq: &[i64],
        text_bert: &[f32],
        ssl_content: &[f32],
        ref_audio: &[f32],
    ) -> Result<Vec<f32>, crate::error::GSVError> {
        let mut output_data = ptr::null_mut();
        let mut output_size = 0;

        unsafe {
            let ret = run_inference(
                self.ptr,
                ref_seq.as_ptr(),
                ref_seq.len(),
                ref_bert.as_ptr(),
                ref_bert.len(),
                text_seq.as_ptr(),
                text_seq.len(),
                text_bert.as_ptr(),
                text_bert.len(),
                ssl_content.as_ptr(),
                ssl_content.len(),
                ref_audio.as_ptr(),
                ref_audio.len(),
                &mut output_data,
                &mut output_size,
            );
            if ret != 0 {
                return Err(crate::error::GSVError::from("Failed to run inference"));
            }
            Ok(ptr_to_vec(output_data, output_size))
        }
    }
}

impl Drop for MNNModelWrapper {
    fn drop(&mut self) {
        unsafe {
            destroy_mnn_model(self.ptr);
        }
    }
}
