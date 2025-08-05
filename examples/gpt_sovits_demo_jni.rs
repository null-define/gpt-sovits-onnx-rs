use futures::StreamExt;
use gpt_sovits_onnx_rs::*;
use hound::WavWriter;
use std::path::Path;
use tokio::runtime::Runtime;

use jni::objects::{JClass, JString};

use jni::sys::{jboolean, jfloatArray, jlong};

use jni::JNIEnv;
use log::LevelFilter;

const JNI_TRUE: jboolean = 1;

const JNI_FALSE: jboolean = 0;

use android_logger::Config;

fn init_logging() {
    {
        android_logger::init_once(
            Config::default()
                .with_max_level(LevelFilter::Debug) // Set desired log level
                .with_tag("rust.gpt_sovits_demo"), // Tag for logcat
        );
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_example_gpt_1sovits_1demo_MainActivity_initModel(
    mut env: JNIEnv,
    _class: JClass,
    g2p_w_path: JString,
    g2p_en_path: JString,
    vits_path: JString,
    ssl_path: JString,
    t2s_encoder_path: JString,
    t2s_fs_decoder_path: JString,
    t2s_s_decoder_path: JString,
    bert_path: JString,
) -> jlong {
    init_logging();
    // Convert JString to Rust String with error handling
    let g2p_w: String = match env.get_string(&g2p_w_path) {
        Ok(s) => s.into(),
        Err(e) => {
            env.throw_new(
                "java/lang/IllegalArgumentException",
                format!("Couldn't get g2pW path: {}", e),
            )
            .expect("Failed to throw exception");
            return 0;
        }
    };
    let g2p_en: String = match env.get_string(&g2p_en_path) {
        Ok(s) => s.into(),
        Err(e) => {
            env.throw_new(
                "java/lang/IllegalArgumentException",
                format!("Couldn't get g2p_en path: {}", e),
            )
            .expect("Failed to throw exception");
            return 0;
        }
    };
    let vits: String = match env.get_string(&vits_path) {
        Ok(s) => s.into(),
        Err(e) => {
            env.throw_new(
                "java/lang/IllegalArgumentException",
                format!("Couldn't get vits path: {}", e),
            )
            .expect("Failed to throw exception");
            return 0;
        }
    };
    let ssl: String = match env.get_string(&ssl_path) {
        Ok(s) => s.into(),
        Err(e) => {
            env.throw_new(
                "java/lang/IllegalArgumentException",
                format!("Couldn't get ssl path: {}", e),
            )
            .expect("Failed to throw exception");
            return 0;
        }
    };
    let t2s_encoder: String = match env.get_string(&t2s_encoder_path) {
        Ok(s) => s.into(),
        Err(e) => {
            env.throw_new(
                "java/lang/IllegalArgumentException",
                format!("Couldn't get t2s encoder path: {}", e),
            )
            .expect("Failed to throw exception");
            return 0;
        }
    };
    let t2s_fs_decoder: String = match env.get_string(&t2s_fs_decoder_path) {
        Ok(s) => s.into(),
        Err(e) => {
            env.throw_new(
                "java/lang/IllegalArgumentException",
                format!("Couldn't get t2s fs decoder path: {}", e),
            )
            .expect("Failed to throw exception");
            return 0;
        }
    };
    let t2s_s_decoder: String = match env.get_string(&t2s_s_decoder_path) {
        Ok(s) => s.into(),
        Err(e) => {
            env.throw_new(
                "java/lang/IllegalArgumentException",
                format!("Couldn't get t2s s decoder path: {}", e),
            )
            .expect("Failed to throw exception");
            return 0;
        }
    };

    let bert: String = match env.get_string(&bert_path) {
        Ok(s) => s.into(),
        Err(e) => {
            env.throw_new(
                "java/lang/IllegalArgumentException",
                format!("Couldn't get bert path: {}", e),
            )
            .expect("Failed to throw exception");
            return 0;
        }
    };

    match TTSModel::new(
        Path::new(&vits),
        Path::new(&ssl),
        Path::new(&t2s_encoder),
        Path::new(&t2s_fs_decoder),
        Path::new(&t2s_s_decoder),
        Some(Path::new(&bert)),
        Some(Path::new(&g2p_w)),
        Some(Path::new(&g2p_en)),
    ) {
        Ok(model) => Box::into_raw(Box::new(model)) as jlong,
        Err(e) => {
            env.throw_new(
                "java/lang/RuntimeException",
                format!("Failed to initialize model: {}", e),
            )
            .expect("Failed to throw exception");
            0
        }
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_example_gpt_1sovits_1demo_MainActivity_processReferenceSync(
    mut env: JNIEnv,
    _class: JClass,
    model_handle: jlong,
    ref_audio_path: JString,
    ref_text: JString,
    language: jlong,
) -> jboolean {
    let model: &mut TTSModel = unsafe { &mut *(model_handle as *mut TTSModel) };
    let ref_audio: String = match env.get_string(&ref_audio_path) {
        Ok(s) => s.into(),
        Err(e) => {
            env.throw_new(
                "java/lang/IllegalArgumentException",
                format!("Couldn't get ref audio path: {}", e),
            )
            .expect("Failed to throw exception");
            return JNI_FALSE;
        }
    };
    let ref_text: String = match env.get_string(&ref_text) {
        Ok(s) => s.into(),
        Err(e) => {
            env.throw_new(
                "java/lang/IllegalArgumentException",
                format!("Couldn't get ref text: {}", e),
            )
            .expect("Failed to throw exception");
            return JNI_FALSE;
        }
    };

    let lang_id = if language == 0 {
        LangId::Auto
    } else {
        LangId::AutoYue
    };

    match model.process_reference_sync(Path::new(&ref_audio), &ref_text, lang_id) {
        Ok(_) => JNI_TRUE,
        Err(e) => {
            env.throw_new(
                "java/lang/RuntimeException",
                format!("Failed to process reference: {}", e),
            )
            .expect("Failed to throw exception");
            JNI_FALSE
        }
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_example_gpt_1sovits_1demo_MainActivity_runInferenceSync(
    mut env: JNIEnv,
    _class: JClass,
    model_handle: jlong,
    text: JString,
    language: jlong,
) -> jfloatArray {
    let model: &mut TTSModel = unsafe { &mut *(model_handle as *mut TTSModel) };
    let text: String = match env.get_string(&text) {
        Ok(s) => s.into(),
        Err(e) => {
            env.throw_new(
                "java/lang/IllegalArgumentException",
                format!("Couldn't get text: {}", e),
            )
            .expect("Failed to throw exception");
            return std::ptr::null_mut();
        }
    };

    
    let lang_id = if language == 0 {
        LangId::Auto
    } else {
        LangId::AutoYue
    };

    match model.synthesize_sync(
        &text,
        SamplingParamsBuilder::new().top_k(4).top_p(0.9).temperature(1.0).repetition_penalty(1.35).build(),
        lang_id,
    ) {
        Ok((_, samples_vec)) => {
            // Fix deprecated into_raw_vec
            let float_array = match env.new_float_array(samples_vec.len() as i32) {
                Ok(arr) => arr,
                Err(e) => {
                    env.throw_new(
                        "java/lang/RuntimeException",
                        format!("Couldn't create float array: {}", e),
                    )
                    .expect("Failed to throw exception");
                    return std::ptr::null_mut();
                }
            };
            match env.set_float_array_region(&float_array, 0, &samples_vec) {
                Ok(_) => float_array.into_raw(),
                Err(e) => {
                    env.throw_new(
                        "java/lang/RuntimeException",
                        format!("Couldn't set float array: {}", e),
                    )
                    .expect("Failed to throw exception");
                    std::ptr::null_mut()
                }
            }
        }
        Err(e) => {
            env.throw_new(
                "java/lang/RuntimeException",
                format!("Failed to run inference: {}", e),
            )
            .expect("Failed to throw exception");
            std::ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_example_gpt_1sovits_1demo_MainActivity_freeModel(
    mut env: JNIEnv,
    _class: JClass,
    model_handle: jlong,
) {
    if model_handle != 0 {
        unsafe { drop(Box::from_raw(model_handle as *mut TTSModel)) };
    }
}
