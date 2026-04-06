use khal_builder::KhalBuilder;

fn main() {
    let shader_crate = "./vortx-shaders";
    let output_dir = "shaders-spirv";

    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_PUSH_CONSTANTS");

    #[allow(unused_mut)]
    let mut builder = KhalBuilder::new(shader_crate, true);
    #[cfg(feature = "push_constants")]
    {
        builder = builder.feature("push_constants");
    }
    builder.build(output_dir);
}
