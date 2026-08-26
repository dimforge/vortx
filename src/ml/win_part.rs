use khal::backend::{GpuBackend, GpuBackendError, GpuPass};
use khal::Shader;
use crate::shapes::TensorLayoutBuffers;
use crate::tensor::{AsTensorMut, AsTensorRef, Tensor};

#[derive(Shader)]
pub struct WinPart {
    pub win_part: vortx_shaders::ml::win_part::WinPart,
    pub win_unpart: vortx_shaders::ml::win_part::WinUnpart,
}

impl WinPart {
    pub fn launch(
        &self,
        backend: &GpuBackend,
        shapes: &mut TensorLayoutBuffers,
        pass: &mut GpuPass,
        mut result: impl AsTensorMut<f32>,
        source: impl AsTensorRef<f32>,
    ) -> Result<(), GpuBackendError> {
        let mut result = result.as_tensor_mut();
        let source = source.as_tensor_ref();
        shapes.insert(backend, result.layout())?;
        shapes.insert(backend, source.layout())?;
        let shape_result = shapes.get(result.layout()).unwrap();
        let shape_source = shapes.get(source.layout()).unwrap();

        let result_len = result.len() as u32;
        let mut buf_result = result.buffer_mut();

        self.win_part.call(
            pass,
            [result_len, 1, 1],
            &shape_result.as_slice(),
            &shape_source.as_slice(),
            &mut buf_result,
            &source.buffer(),
        )
    }

    pub fn launch_unpart(
        &self,
        backend: &GpuBackend,
        shapes: &mut TensorLayoutBuffers,
        pass: &mut GpuPass,
        window_size: &Tensor<u32>,
        mut result: impl AsTensorMut<f32>,
        source: impl AsTensorRef<f32>,
    ) -> Result<(), GpuBackendError> {
        let mut result = result.as_tensor_mut();
        let source = source.as_tensor_ref();
        shapes.insert(backend, result.layout())?;
        shapes.insert(backend, source.layout())?;
        let shape_result = shapes.get(result.layout()).unwrap();
        let shape_source = shapes.get(source.layout()).unwrap();

        let result_len = result.len() as u32;
        let mut buf_result = result.buffer_mut();

        self.win_unpart.call(
            pass,
            [result_len, 1, 1],
            &shape_result.as_slice(),
            &shape_source.as_slice(),
            &window_size.buffer().as_slice(),
            &mut buf_result,
            &source.buffer(),
        )
    }
}
