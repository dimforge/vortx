use crate::shapes::TensorLayoutBuffers;
use crate::tensor::{AsTensorMut, AsTensorRef, Tensor};
use khal::Shader;
use khal::backend::{GpuBackend, GpuBackendError, GpuPass};

#[derive(Shader)]
pub struct ConvTranspose2d {
    pub init_dest: vortx_shaders::ml::conv_transpose_2d::InitDest,
    pub init_wdata: vortx_shaders::ml::conv_transpose_2d::InitWdata,
    pub init_src_a: vortx_shaders::ml::conv_transpose_2d::InitSrcA,
    pub init_src_b: vortx_shaders::ml::conv_transpose_2d::InitSrcB,
    pub conv_transpose_2d_ref: vortx_shaders::ml::conv_transpose_2d::ConvTranspose2dRef,
    pub conv_transpose_2d: vortx_shaders::ml::conv_transpose_2d::ConvTranspose2d,
}

impl ConvTranspose2d {
    pub fn launch_ref(
        &self,
        backend: &GpuBackend,
        pass: &mut GpuPass,
        shapes: &mut TensorLayoutBuffers,
        stride: &Tensor<u32>,
        mut dest: impl AsTensorMut<f32>,
        src0: impl AsTensorRef<f32>,
        src1: impl AsTensorRef<f32>,
        mut wdata: impl AsTensorMut<f32>,
    ) -> Result<(), GpuBackendError> {
        let mut dest = dest.as_tensor_mut();
        let src0 = src0.as_tensor_ref();
        let src1 = src1.as_tensor_ref();
        let mut wdata = wdata.as_tensor_mut();

        assert_eq!(wdata.len(), src0.len() + src1.len());

        shapes.insert(backend, dest.layout())?;
        shapes.insert(backend, src0.layout())?;
        shapes.insert(backend, src1.layout())?;
        shapes.insert(backend, wdata.layout())?;
        let shape_dest = shapes.get(dest.layout()).unwrap();
        let shape_src0 = shapes.get(src0.layout()).unwrap();
        let shape_src1 = shapes.get(src1.layout()).unwrap();
        let shape_wdata = shapes.get(wdata.layout()).unwrap();

        // init_dest: shape_dest, dest
        {
            let dest_len = dest.len() as u32;
            let mut buf_dest = dest.buffer_mut();
            self.init_dest.call(
                pass,
                [dest_len, 1, 1],
                &shape_dest.as_slice(),
                &mut buf_dest,
            )?;
        }

        // init_wdata: shape_wdata, wdata
        {
            let wdata_len = wdata.len() as u32;
            let mut buf_wdata = wdata.buffer_mut();
            self.init_wdata.call(
                pass,
                [wdata_len, 1, 1],
                &shape_wdata.as_slice(),
                &mut buf_wdata,
            )?;
        }

        // init_src_a: shape_src0, src0, wdata
        {
            let mut buf_wdata = wdata.buffer_mut();
            self.init_src_a.call(
                pass,
                [src0.len() as u32, 1, 1],
                &shape_src0.as_slice(),
                &src0.buffer(),
                &mut buf_wdata,
            )?;
        }

        // init_src_b: shape_src0, shape_src1, src1, wdata
        {
            let mut buf_wdata = wdata.buffer_mut();
            self.init_src_b.call(
                pass,
                [src1.len() as u32, 1, 1],
                &shape_src0.as_slice(),
                &shape_src1.as_slice(),
                &src1.buffer(),
                &mut buf_wdata,
            )?;
        }

        // conv_transpose_2d_ref: shape_src0, shape_src1, shape_dest, stride, wdata, dest
        {
            let dest_size2 = dest.size(2);
            let mut buf_dest = dest.buffer_mut();
            self.conv_transpose_2d_ref.call(
                pass,
                [dest_size2, 1, 1],
                &shape_src0.as_slice(),
                &shape_src1.as_slice(),
                &shape_dest.as_slice(),
                &stride.buffer().as_slice(),
                &wdata.buffer(),
                &mut buf_dest,
            )?;
        }

        Ok(())
    }

    pub fn launch(
        &self,
        backend: &GpuBackend,
        pass: &mut GpuPass,
        shapes: &mut TensorLayoutBuffers,
        stride: &mut Tensor<u32>,
        mut dest: impl AsTensorMut<f32>,
        src0: impl AsTensorRef<f32>,
        src1: impl AsTensorRef<f32>,
    ) -> Result<(), GpuBackendError> {
        let mut dest = dest.as_tensor_mut();
        let src0 = src0.as_tensor_ref();
        let src1 = src1.as_tensor_ref();

        let src0 = src0.permute([3, 0, 1, 2]);
        let src1 = src1.permute([2, 0, 1, 3]);

        shapes.insert(backend, dest.layout())?;
        shapes.insert(backend, src0.layout())?;
        shapes.insert(backend, src1.layout())?;
        let shape_dest = shapes.get(dest.layout()).unwrap();
        let shape_src0 = shapes.get(src0.layout()).unwrap();
        let shape_src1 = shapes.get(src1.layout()).unwrap();

        // conv_transpose_2d: shape_src1, shape_src0, shape_dest, stride, src1, src0, dest
        let dest_size2 = dest.size(2);
        let mut buf_dest = dest.buffer_mut();

        self.conv_transpose_2d.call(
            pass,
            [dest_size2, 1, 1],
            &shape_src1.as_slice(),
            &shape_src0.as_slice(),
            &shape_dest.as_slice(),
            &stride.buffer().as_slice(),
            &src1.buffer(),
            &src0.buffer(),
            &mut buf_dest,
        )?;

        Ok(())
    }
}
