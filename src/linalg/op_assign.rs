use crate::shaders::linalg::{GpuAdd, GpuCopy, GpuCopyWithOffsets, GpuDiv, GpuMul, GpuSub};
use crate::shapes::TensorLayoutBuffers;
use crate::tensor::{AsTensorMut, AsTensorRef};
use khal::Shader;
use khal::backend::{GpuBackend, GpuBackendError, GpuPass};

// Re-export BinOpOffsets from shader crate.
pub use vortx_shaders::linalg::op_assign::BinOpOffsets;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[non_exhaustive]
/// The desired operation for the [`OpAssign`] kernel.
pub enum OpAssignVariant {
    /// Sum: `a += b`
    Add,
    /// Subtraction: `a -= b`
    Sub,
    /// Product: `a *= b`
    Mul,
    /// Division: `a /= b`
    Div,
    /// Copy: `a = b`
    Copy,
}

/// Modules for various in-place binary operations.
#[derive(Shader)]
pub struct OpAssign {
    /// Kernel for computing in-place the sum of two tensors.
    pub add: GpuAdd,
    /// Kernel for computing in-place the subtraction of two tensors.
    pub sub: GpuSub,
    /// Kernel for computing in-place the product of two tensors.
    pub mul: GpuMul,
    /// Kernel for computing in-place the division of two tensors.
    pub div: GpuDiv,
    /// Kernel for copying a tensor into another.
    pub copy: GpuCopy,
    /// Kernel for copying a tensor into another, using a custom offset where to start reading
    /// the source tensor.
    pub copy_with_offsets: GpuCopyWithOffsets,
}

impl OpAssign {
    /// Launches the kernel for a binary operation `variant` where the first operand
    /// `a` being read & written to, and `b` is only being read from (e.g. `a += b`).
    pub fn launch(
        &self,
        backend: &GpuBackend,
        #[cfg_attr(feature = "push_constants", allow(unused_variables))]
        shapes: &mut TensorLayoutBuffers,
        pass: &mut GpuPass,
        variant: OpAssignVariant,
        mut a: impl AsTensorMut<f32>,
        b: impl AsTensorRef<f32>,
    ) -> Result<(), GpuBackendError> {
        let mut a = a.as_tensor_mut();
        let b = b.as_tensor_ref();

        let Some((mut shape_a, mut shape_b)) = a.layout().broadcast_assign(b.layout()) else {
            // TODO: return an error instead of panic.
            panic!(
                "shape_a: {:?} is incompatible with shape_b: {:?}",
                a.layout(),
                b.layout()
            )
        };

        shape_a = shape_a.canonicalize();
        shape_b = shape_b.canonicalize();

        let num_threads = a.len() as u32;

        #[cfg(not(feature = "push_constants"))]
        {
            shapes.insert(backend, shape_a)?;
            shapes.insert(backend, shape_b)?;
            let shape_a_buf = shapes.get(shape_a).unwrap();
            let shape_b_buf = shapes.get(shape_b).unwrap();
            let mut buf_a = a.buffer_mut();

            macro_rules! call(
                ($kernel: expr) => {
                    $kernel.call(
                        pass,
                        num_threads,
                        &shape_a_buf.as_slice(),
                        &shape_b_buf.as_slice(),
                        &mut buf_a,
                        &b.buffer(),
                    )?
                }
            );

            match variant {
                OpAssignVariant::Add => call!(self.add),
                OpAssignVariant::Copy => call!(self.copy),
                OpAssignVariant::Div => call!(self.div),
                OpAssignVariant::Mul => call!(self.mul),
                OpAssignVariant::Sub => call!(self.sub),
            }
        }

        #[cfg(feature = "push_constants")]
        {
            let mut buf_a = a.buffer_mut();

            macro_rules! call(
                ($kernel: expr) => {
                    pipeline.call(
                        pass,
                        num_threads,
                        &mut buf_a,
                        &b.buffer(),
                        crate::shaders::linalg::Shapes2 {
                            shape_a: shape_a.into(),
                            shape_b: shape_b.into(),
                        },
                    )?
                }
            );

            match variant {
                OpAssignVariant::Add => call!(self.add),
                OpAssignVariant::Copy => call!(self.copy),
                OpAssignVariant::Div => call!(self.div),
                OpAssignVariant::Mul => call!(self.mul),
                OpAssignVariant::Sub => call!(self.sub),
            }
        }

        Ok(())
    }

    // FIXME: this only exists because we needed a quick fix to work around the limitation on
    //        buffer offset alignment when targeting WebGpu (in our case the buffer offset was 32
    //        but the hardware needed an alignment of 256).
    //        We should figure out a more general way of handling this.
    /// Launches the GPU kernel for copying the content of `b[offsets.b..]` into
    /// `a[offsets.a..]`.
    ///
    /// While this is similar to calling `launch` with an already offset tensor view,
    /// this is useful for cases where the desired offset is smaller than what's supported
    /// by the backend (for example WebGpu).
    pub fn launch_copy_with_offsets(
        &self,
        backend: &GpuBackend,
        // Note: copy_with_offsets shader doesn't use push_constants, so shapes buffer is always needed
        shapes: &mut TensorLayoutBuffers,
        pass: &mut GpuPass,
        offsets: impl AsTensorRef<BinOpOffsets>,
        mut a: impl AsTensorMut<f32>,
        b: impl AsTensorRef<f32>,
    ) -> Result<(), GpuBackendError> {
        let offsets = offsets.as_tensor_ref();
        let mut a = a.as_tensor_mut();
        let b = b.as_tensor_ref();
        let pipeline = &self.copy_with_offsets;

        let Some((mut shape_a, mut shape_b)) = a.layout().broadcast_assign(b.layout()) else {
            // TODO: return an error instead of panic.
            panic!(
                "shape_a: {:?} is incompatible with shape_b: {:?}",
                a.layout().size,
                b.layout().size
            )
        };

        shape_a = shape_a.canonicalize();
        shape_b = shape_b.canonicalize();

        let num_threads = a.len() as u32;

        // copy_with_offsets doesn't use push_constants for shapes
        shapes.insert(backend, shape_a)?;
        shapes.insert(backend, shape_b)?;
        let shape_a_buf = shapes.get(shape_a).unwrap();
        let shape_b_buf = shapes.get(shape_b).unwrap();
        let mut buf_a = a.buffer_mut();

        pipeline.call(
            pass,
            num_threads,
            &offsets.buffer(),
            &shape_a_buf.as_slice(),
            &shape_b_buf.as_slice(),
            &mut buf_a,
            &b.buffer(),
        )
    }
}

#[cfg(test)]
mod test {
    use super::OpAssignVariant;
    use crate::shapes::TensorLayoutBuffers;
    use crate::tensor::Tensor;
    use khal::BufferUsages;
    use khal::backend::{Backend, Encoder, GpuBackend, WebGpu};
    use khal::shader::Shader;
    use nalgebra::DVector;

    #[futures_test::test]
    #[serial_test::serial]
    async fn gpu_op_assign_webgpu() {
        let webgpu = WebGpu::default().await.unwrap();
        let backend = GpuBackend::WebGpu(webgpu);
        gpu_op_assign_with_backend(&backend).await;
    }

    #[cfg(feature = "cpu")]
    #[futures_test::test]
    async fn gpu_op_assign_cpu() {
        gpu_op_assign_with_backend(&GpuBackend::Cpu).await;
    }

    #[cfg(feature = "cuda")]
    #[futures_test::test]
    async fn gpu_op_assign_cuda() {
        let cuda = GpuBackend::Cuda(khal::backend::cuda::Cuda::new(0).unwrap());
        gpu_op_assign_with_backend(&cuda).await;
    }

    #[cfg(feature = "metal")]
    #[futures_test::test]
    #[serial_test::serial]
    async fn gpu_op_assign_metal() {
        let metal = GpuBackend::Metal(khal::backend::metal::Metal::new().unwrap());
        gpu_op_assign_with_backend(&metal).await;
    }

    async fn gpu_op_assign_with_backend(backend: &GpuBackend) {
        let ops = [
            OpAssignVariant::Add,
            OpAssignVariant::Sub,
            OpAssignVariant::Mul,
            OpAssignVariant::Div,
            OpAssignVariant::Copy,
        ];
        let op_assign = super::OpAssign::from_backend(backend).unwrap();

        for op in ops {
            println!("Testing: {:?}", op);

            let mut shapes = TensorLayoutBuffers::new(backend);
            let mut encoder = backend.begin_encoding();

            const LEN: u32 = 1757;

            let v0 = DVector::from_fn(LEN as usize, |i, _| i as f32 + 0.1);
            let v1 = DVector::from_fn(LEN as usize, |i, _| i as f32 * 10.0 + 0.1);
            let mut gpu_result = DVector::zeros(LEN as usize);
            let mut gpu_v0 =
                Tensor::vector(backend, &v0, BufferUsages::STORAGE | BufferUsages::COPY_SRC)
                    .unwrap();
            let gpu_v1 = Tensor::vector(backend, &v1, BufferUsages::STORAGE).unwrap();

            let mut pass = encoder.begin_pass("op_assign", None);
            op_assign
                .launch(backend, &mut shapes, &mut pass, op, &mut gpu_v0, &gpu_v1)
                .unwrap();
            drop(pass); // Ensure the pass is ended before the encoder is borrowed again.

            backend.submit(encoder).unwrap();
            backend
                .slow_read_buffer(gpu_v0.buffer(), gpu_result.as_mut_slice())
                .await
                .unwrap();

            let cpu_result = match op {
                OpAssignVariant::Add => v0 + v1,
                OpAssignVariant::Sub => v0 - v1,
                OpAssignVariant::Mul => v0.component_mul(&v1),
                OpAssignVariant::Div => v0.component_div(&v1),
                OpAssignVariant::Copy => v1.clone(),
            };

            approx::assert_relative_eq!(gpu_result, cpu_result, epsilon = 1.0e-7);
        }
    }
}
