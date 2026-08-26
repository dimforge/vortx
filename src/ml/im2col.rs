use khal::backend::{Backend, DispatchGrid, GpuBackend, GpuBackendError, GpuPass};
use khal::Shader;
use crate::tensor::{AsTensorMut, AsTensorRef, Tensor};


pub type Im2ColConfig = vortx_shaders::ml::im2col::Im2ColParams;

#[derive(Shader)]
pub struct Im2Col {
    pub im2col: vortx_shaders::ml::im2col::Im2col,
}

impl Im2Col {
    // im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
    // kernel: [OC, IC, KH, KW]
    // input: [N, IC, IH, IW]
    // result: [N, OH, OW, IC*KH*KW]
    pub fn launch(
        &self,
        backend: &GpuBackend,
        pass: &mut GpuPass,
        params: &mut Tensor<Im2ColConfig>,
        mut result: impl AsTensorMut<f32>,
        kernel: impl AsTensorRef<f32>,
        input: impl AsTensorRef<f32>,
        s0: u32,
        s1: u32,
        p0: u32,
        p1: u32,
        d0: u32,
        d1: u32,
        is_2d: bool,
    ) -> Result<(), GpuBackendError> {
        let mut result = result.as_tensor_mut();
        let kernel = kernel.as_tensor_ref();
        let input = input.as_tensor_ref();

        let ilayout = input.layout();
        let klayout = kernel.layout();
        let rlayout = result.layout();

        let ic = ilayout.size[if is_2d { 2 } else { 0 }];
        let ih = if is_2d { ilayout.size[0] } else { 1 };
        let iw = ilayout.size[1];

        let kh = if is_2d { klayout.size[0] } else { 1 };
        let kw = klayout.size[1];

        let oh = if is_2d { rlayout.size[2] } else { 1 };
        let ow = rlayout.size[0];

        let offset_delta = ilayout.stride[if is_2d { 2 } else { 0 }];
        let batch_offset = ilayout.stride[if is_2d { 3 } else { 2 }];

        let pelements = ow * kw * kh;
        let chw = ic * kh * kw;

        let config = Im2ColConfig {
            batch_offset,
            offset_delta,
            IC: ic,
            IW: iw,
            IH: ih,
            OW: ow,
            OH: oh,
            KW: kw,
            KH: kh,
            pelements,
            CHW: chw,
            s0: s0 as i32,
            s1: s1 as i32,
            p0: p0 as i32,
            p1: p1 as i32,
            d0: d0 as i32,
            d1: d1 as i32,
        };

        backend.write_buffer(params.buffer_mut(), 0, &[config])?;

        let batch = ilayout.size[if is_2d { 3 } else { 2 }];
        let grid = [(ow * kw * kh).div_ceil(32), oh, batch * ic];
        let mut buf_result = result.buffer_mut();

        self.im2col.call(
            pass,
            DispatchGrid::Grid(grid),
            &params.buffer().as_slice(),
            &input.buffer(),
            &mut buf_result,
        )?;

        Ok(())
    }
}
