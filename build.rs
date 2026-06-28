use std::path::PathBuf;

use khal_builder::KhalBuilder;

fn main() {
    let output_dir = PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR not set by cargo"))
        .join("shaders-spirv");

    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_PUSH_CONSTANTS");

    #[allow(unused_mut)]
    let mut builder = KhalBuilder::from_dependency("vortx-shaders", true);
    #[cfg(feature = "unsafe_remove_boundchecks")]
    {
        builder = builder.feature("unsafe_remove_boundchecks");
    }
    #[cfg(feature = "push_constants")]
    {
        builder = builder.feature("push_constants");
    }
    builder.build(&output_dir);
}
