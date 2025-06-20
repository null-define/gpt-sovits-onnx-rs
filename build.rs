use cmake::Config;
use std::path::{Path, PathBuf};
use std::{env, fs, io};

fn add_link_search_path(dir: &Path) -> io::Result<()> {
    if dir.is_dir() {
        println!("cargo:rustc-link-search={}", dir.display());
        // Only recurse into subdirectories that are likely to contain libraries
        for entry in fs::read_dir(dir)? {
            let path = entry?.path();
            if path.is_dir() && path.join("libMNN.so").exists() {
                add_link_search_path(&path)?;
            }
        }
    }
    Ok(())
}

fn main() -> io::Result<()> {
    // Determine build profile and set CMake build type
    // let profile = env::var("PROFILE").unwrap_or("release".to_string());
    // Configure CMake build
    let mut config = Config::new("gpt-sovits-mnn-cc");
    config
        // .define("CMAKE_BUILD_TYPE", "Debug")
        .static_crt(true)
        .build_target("gpt-sovits-mnn-cc");

    let dst = config.build();

    // Set up paths
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Add link search paths
    add_link_search_path(&out.join("build"))?;
    println!("cargo:rustc-link-search=native={}", dst.display());
    println!("cargo:rustc-link-search=native={}", "gpt-sovits-mnn-cc/MNN-3.2.0/build");

    // Link libraries
    println!("cargo:rustc-link-lib=gpt-sovits-mnn-cc");
    println!("cargo:rustc-link-lib=dylib=MNN");
    println!("cargo:rustc-link-lib=stdc++");

    // Copy MNN library if it exists
    // let built_lib_path = mnn_dir.join("libMNN.so");
    // if built_lib_path.exists() {
    //     let dest_lib_path = final_target_dir.join("libMNN.so");
    //     fs::create_dir_all(&final_target_dir)?;
    //     fs::copy(&built_lib_path, &dest_lib_path).map(|_| {
    //         println!(
    //             "Copied {} to {}",
    //             built_lib_path.display(),
    //             dest_lib_path.display()
    //         );
    //     })?;
    // } else {
    //     eprintln!("Built library not found at: {}", built_lib_path.display());
    // }

    // Re-run build if source files change
    println!("cargo:rerun-if-changed=cc/gpt-sovits-mnn.cc");
    println!("cargo:rerun-if-changed=gpt-sovits-mnn-cc/CMakeLists.txt");

    Ok(())
}