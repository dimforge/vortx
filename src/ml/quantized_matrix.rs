use crate::ml::{
    GpuBlockQ4_0x2, GpuBlockQ4_1x2, GpuBlockQ4K, GpuBlockQ5_0x2, GpuBlockQ5_1x2, GpuBlockQ5K,
    GpuBlockQ6Kx2, GpuBlockQ8_0x2, GpuBlockQ8K,
};
use crate::shapes::TensorLayout;
use crate::tensor::Tensor;
use khal::ShaderArgs;
use khal::backend::{GpuDispatch, ShaderBinding};
use khal::shader::ShaderArgsError;

pub enum GpuQuantTensor {
    F32(Tensor<f32>),
    Q8_0(Tensor<GpuBlockQ8_0x2>),
    Q5_0(Tensor<GpuBlockQ5_0x2>),
    Q5_1(Tensor<GpuBlockQ5_1x2>),
    Q4_0(Tensor<GpuBlockQ4_0x2>),
    Q4_1(Tensor<GpuBlockQ4_1x2>),
    Q8K(Tensor<GpuBlockQ8K>),
    Q6K(Tensor<GpuBlockQ6Kx2>),
    Q5K(Tensor<GpuBlockQ5K>),
    Q4K(Tensor<GpuBlockQ4K>),
}

impl GpuQuantTensor {
    pub fn rank(&self) -> u32 {
        match self {
            GpuQuantTensor::F32(x) => x.rank(),
            GpuQuantTensor::Q8_0(x) => x.rank(),
            GpuQuantTensor::Q5_0(x) => x.rank(),
            GpuQuantTensor::Q5_1(x) => x.rank(),
            GpuQuantTensor::Q4_0(x) => x.rank(),
            GpuQuantTensor::Q4_1(x) => x.rank(),
            GpuQuantTensor::Q8K(x) => x.rank(),
            GpuQuantTensor::Q6K(x) => x.rank(),
            GpuQuantTensor::Q5K(x) => x.rank(),
            GpuQuantTensor::Q4K(x) => x.rank(),
        }
    }

    pub fn layout(&self) -> TensorLayout {
        match self {
            GpuQuantTensor::F32(x) => x.layout(),
            GpuQuantTensor::Q8_0(x) => x.layout(),
            GpuQuantTensor::Q5_0(x) => x.layout(),
            GpuQuantTensor::Q5_1(x) => x.layout(),
            GpuQuantTensor::Q4_0(x) => x.layout(),
            GpuQuantTensor::Q4_1(x) => x.layout(),
            GpuQuantTensor::Q8K(x) => x.layout(),
            GpuQuantTensor::Q6K(x) => x.layout(),
            GpuQuantTensor::Q5K(x) => x.layout(),
            GpuQuantTensor::Q4K(x) => x.layout(),
        }
    }
}

impl<'b> ShaderArgs<'b> for GpuQuantTensor {
    fn write_arg<'a>(
        &'b self,
        binding: ShaderBinding,
        dispatch: &mut GpuDispatch<'a>,
    ) -> Result<(), ShaderArgsError>
    where
        'b: 'a,
    {
        match self {
            GpuQuantTensor::F32(matrix) => matrix.buffer().write_arg(binding, dispatch),
            GpuQuantTensor::Q8_0(matrix) => matrix.buffer().write_arg(binding, dispatch),
            GpuQuantTensor::Q5_0(matrix) => matrix.buffer().write_arg(binding, dispatch),
            GpuQuantTensor::Q5_1(matrix) => matrix.buffer().write_arg(binding, dispatch),
            GpuQuantTensor::Q4_0(matrix) => matrix.buffer().write_arg(binding, dispatch),
            GpuQuantTensor::Q4_1(matrix) => matrix.buffer().write_arg(binding, dispatch),
            GpuQuantTensor::Q8K(matrix) => matrix.buffer().write_arg(binding, dispatch),
            GpuQuantTensor::Q6K(matrix) => matrix.buffer().write_arg(binding, dispatch),
            GpuQuantTensor::Q5K(matrix) => matrix.buffer().write_arg(binding, dispatch),
            GpuQuantTensor::Q4K(matrix) => matrix.buffer().write_arg(binding, dispatch),
        }
    }
}

macro_rules! impl_from(
    ($($variant: ident, $scalar: ident);*) => {$(
        impl From<Tensor<$scalar>> for GpuQuantTensor {
            fn from(value: Tensor<$scalar>) -> Self {
                Self::$variant(value)
            }
        }
    )*}
);

impl_from!(
    F32, f32;
    Q8_0, GpuBlockQ8_0x2;
    Q5_0, GpuBlockQ5_0x2;
    Q5_1, GpuBlockQ5_1x2;
    Q4_0, GpuBlockQ4_0x2;
    Q4_1, GpuBlockQ4_1x2;
    Q8K, GpuBlockQ8K;
    Q6K, GpuBlockQ6Kx2;
    Q5K, GpuBlockQ5K;
    Q4K, GpuBlockQ4K
);

impl GpuQuantTensor {
    pub fn shape(&self) -> TensorLayout {
        match self {
            Self::F32(m) => m.as_view().layout(),
            Self::Q8_0(m) => m.as_view().layout(),
            Self::Q5_0(m) => m.as_view().layout(),
            Self::Q5_1(m) => m.as_view().layout(),
            Self::Q4_0(m) => m.as_view().layout(),
            Self::Q4_1(m) => m.as_view().layout(),
            Self::Q8K(m) => m.as_view().layout(),
            Self::Q6K(m) => m.as_view().layout(),
            Self::Q5K(m) => m.as_view().layout(),
            Self::Q4K(m) => m.as_view().layout(),
        }
    }
}
