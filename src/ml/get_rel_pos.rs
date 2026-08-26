use khal::backend::{GpuBackend, GpuBackendError, GpuPass};
use khal::Shader;
use crate::shapes::TensorLayoutBuffers;
use crate::tensor::{AsTensorMut, AsTensorRef};

#[derive(Shader)]
pub struct GetRelPos {
    pub get_rel_pos: vortx_shaders::ml::get_rel_pos::GetRelPos,
    pub add_rel_pos_phase_a: vortx_shaders::ml::get_rel_pos::AddRelPosPhaseA,
    pub add_rel_pos_phase_b: vortx_shaders::ml::get_rel_pos::AddRelPosPhaseB,
}

impl GetRelPos {
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

        self.get_rel_pos.call(
            pass,
            [result_len, 1, 1],
            &shape_result.as_slice(),
            &shape_source.as_slice(),
            &mut buf_result,
            &source.buffer(),
        )
    }

    pub fn launch_add_rel_pos(
        &self,
        backend: &GpuBackend,
        shapes: &mut TensorLayoutBuffers,
        pass: &mut GpuPass,
        mut dst: impl AsTensorMut<f32>,
        src1: impl AsTensorRef<f32>,
        src2: impl AsTensorRef<f32>,
    ) -> Result<(), GpuBackendError> {
        let mut dst = dst.as_tensor_mut();
        let src1 = src1.as_tensor_ref();
        let src2 = src2.as_tensor_ref();

        assert_eq!(src1.layout().size, src2.layout().size);
        assert_eq!(src1.size(3), dst.size(2));
        assert_eq!(src1.size(1) * src1.size(1), dst.size(1));
        assert_eq!(src1.size(0) * src1.size(1), dst.size(0));

        shapes.insert(backend, src1.layout())?;
        let shape_src1 = shapes.get(src1.layout()).unwrap();

        // Phase A: add_rel_pos_phase_a(shape_src1, dst, src1)
        {
            let mut buf_dst = dst.buffer_mut();
            self.add_rel_pos_phase_a.call(
                pass,
                [src1.len() as u32, 1, 1],
                &shape_src1.as_slice(),
                &mut buf_dst,
                &src1.buffer(),
            )?;
        }

        // Phase B: add_rel_pos_phase_b(shape_src1, dst, src2)
        {
            let mut buf_dst = dst.buffer_mut();
            self.add_rel_pos_phase_b.call(
                pass,
                [src1.len() as u32, 1, 1],
                &shape_src1.as_slice(),
                &mut buf_dst,
                &src2.buffer(),
            )
        }
    }
}
