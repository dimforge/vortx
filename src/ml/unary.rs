use khal_std::glamx::Vec4;
use khal::backend::{GpuBackend, GpuBackendError, GpuPass};
use khal::Shader;
use nalgebra::{Dyn, StorageMut, Vector};
use crate::shapes::TensorLayoutBuffers;
use crate::tensor::{AsTensorMut, AsTensorRef, Tensor};

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[non_exhaustive]
/// Listing of all unary operations that can be applied by the [`Unary`] kernel.
pub enum UnaryOp {
    Abs,
    Sgn,
    Neg,
    Step,
    Elu,
    Gelu,
    GeluQuick,
    Silu,
    Tanh,
    Sin,
    Cos,
    Relu,
    Sigmoid,
    HardSigmoid,
    // HardSwish,
    Sqr,
    Sqrt,
    Log,
    Exp,
    Reciprocal,
    Erf,
    // Unary ops with extra args.
    LeakyRelu,
    Clamp,
    Scale,
    AddScalar, // Named GGML_OP_ADD1 in ggml.
    Pow,
}

impl UnaryOp {
    const fn has_args(self) -> bool {
        match self {
            Self::Abs
            | Self::Sgn
            | Self::Neg
            | Self::Step
            | Self::Elu
            | Self::Gelu
            | Self::GeluQuick
            | Self::Silu
            | Self::Tanh
            | Self::Relu
            | Self::Sigmoid
            | Self::HardSigmoid
            // | Self::HardSwish
            | Self::Sqr
            | Self::Sqrt
            | Self::Log
            | Self::Exp
            | Self::Reciprocal
            | Self::Erf
            | Self::Sin
            | Self::Cos => false,
            Self::LeakyRelu | Self::Clamp | Self::Scale | Self::AddScalar | Self::Pow => true,
        }
    }

    pub fn eval(self, x: f32, args: Vec4) -> f32 {
        match self {
            Self::Abs => x.abs(),
            Self::Sgn => x.signum(),
            Self::Neg => -x,
            Self::Step => {
                if x > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Elu => {
                if x > 0.0 {
                    x
                } else {
                    x.exp() - 1.0
                }
            }
            Self::Gelu => {
                const GELU_COEF_A: f32 = 0.044715;
                const SQRT_2_OVER_PI: f32 = 0.7978846;
                0.5 * x * (1.0 + (SQRT_2_OVER_PI * x * (1.0 + GELU_COEF_A * x * x)).tanh())
            }
            Self::GeluQuick => {
                const GELU_QUICK_COEF: f32 = -1.702;
                x * (1.0 / (1.0 + (GELU_QUICK_COEF * x).exp()))
            }
            Self::Silu => x / (1.0 + (-x).exp()),
            Self::Tanh => x.tanh(),
            Self::Relu => x.max(0.0),
            Self::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Self::HardSigmoid => 1.0f32.min(0.0f32.max((x + 3.0) / 6.0)),
            // Self::HardSwish => x * 1.0f32.min(0.0f32.max((x + 3.0) / 6.0)),
            Self::Sqr => x * x,
            Self::Sqrt => x.sqrt(),
            Self::Sin => x.sin(),
            Self::Cos => x.cos(),
            Self::Log => x.ln(),
            Self::Exp => x.exp(),
            Self::Reciprocal => 1.0 / x,
            Self::Erf => {
                // Abramowitz and Stegun approximation
                #[allow(clippy::excessive_precision)]
                let a1: f32 = 0.254829592;
                #[allow(clippy::excessive_precision)]
                let a2: f32 = -0.284496736;
                #[allow(clippy::excessive_precision)]
                let a3: f32 = 1.421413741;
                #[allow(clippy::excessive_precision)]
                let a4: f32 = -1.453152027;
                #[allow(clippy::excessive_precision)]
                let a5: f32 = 1.061405429;
                #[allow(clippy::excessive_precision)]
                let p: f32 = 0.3275911;
                let sign = if x >= 0.0 { 1.0 } else { -1.0 };
                let x = x.abs();
                let t = 1.0 / (1.0 + p * x);
                let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
                sign * y
            }
            Self::LeakyRelu => x.max(0.0) + x.min(0.0) * args.x,
            Self::Clamp => x.clamp(args.x, args.y),
            Self::Scale => x * args.x,
            Self::AddScalar => x + args.x,
            Self::Pow => x.powf(args.x),
        }
    }
}

/// Shader implementing various unary operations selected with [`UnaryOp`].
#[derive(Shader)]
pub struct Unary {
    pub abs_op: vortx_shaders::ml::unary::AbsOp,
    pub abs_inplace: vortx_shaders::ml::unary::AbsInplace,
    pub sgn_op: vortx_shaders::ml::unary::SgnOp,
    pub sgn_inplace: vortx_shaders::ml::unary::SgnInplace,
    pub neg_op: vortx_shaders::ml::unary::NegOp,
    pub neg_inplace: vortx_shaders::ml::unary::NegInplace,
    pub step_op: vortx_shaders::ml::unary::StepOp,
    pub step_inplace: vortx_shaders::ml::unary::StepInplace,
    pub elu_op: vortx_shaders::ml::unary::EluOp,
    pub elu_inplace: vortx_shaders::ml::unary::EluInplace,
    pub gelu_op: vortx_shaders::ml::unary::GeluOp,
    pub gelu_inplace: vortx_shaders::ml::unary::GeluInplace,
    pub gelu_quick_op: vortx_shaders::ml::unary::GeluQuickOp,
    pub gelu_quick_inplace: vortx_shaders::ml::unary::GeluQuickInplace,
    pub silu_op: vortx_shaders::ml::unary::SiluOp,
    pub silu_inplace: vortx_shaders::ml::unary::SiluInplace,
    pub tanh_op: vortx_shaders::ml::unary::TanhOp,
    pub tanh_inplace: vortx_shaders::ml::unary::TanhInplace,
    pub relu_op: vortx_shaders::ml::unary::ReluOp,
    pub relu_inplace: vortx_shaders::ml::unary::ReluInplace,
    pub sigmoid_op: vortx_shaders::ml::unary::SigmoidOp,
    pub sigmoid_inplace: vortx_shaders::ml::unary::SigmoidInplace,
    pub hard_sigmoid_op: vortx_shaders::ml::unary::HardSigmoidOp,
    pub hard_sigmoid_inplace: vortx_shaders::ml::unary::HardSigmoidInplace,
    // pub hard_swish_op: vortx_shaders::ml::unary::HardSwishOp,
    // pub hard_swish_inplace: vortx_shaders::ml::unary::HardSwishInplace,
    pub sqr_op: vortx_shaders::ml::unary::SqrOp,
    pub sqr_inplace: vortx_shaders::ml::unary::SqrInplace,
    pub sqrt_op: vortx_shaders::ml::unary::SqrtOp,
    pub sqrt_inplace: vortx_shaders::ml::unary::SqrtInplace,
    pub log_op: vortx_shaders::ml::unary::LogOp,
    pub log_inplace: vortx_shaders::ml::unary::LogInplace,
    pub leaky_relu_op: vortx_shaders::ml::unary::LeakyReluOp,
    pub leaky_relu_inplace: vortx_shaders::ml::unary::LeakyReluInplace,
    pub clamp_op: vortx_shaders::ml::unary::ClampOp,
    pub clamp_inplace: vortx_shaders::ml::unary::ClampInplace,
    pub scale_op: vortx_shaders::ml::unary::ScaleOp,
    pub scale_inplace: vortx_shaders::ml::unary::ScaleInplace,
    pub add_scalar_op: vortx_shaders::ml::unary::AddScalarOp,
    pub add_scalar_inplace: vortx_shaders::ml::unary::AddScalarInplace,
    pub sin_op: vortx_shaders::ml::unary::SinOp,
    pub sin_inplace: vortx_shaders::ml::unary::SinInplace,
    pub cos_op: vortx_shaders::ml::unary::CosOp,
    pub cos_inplace: vortx_shaders::ml::unary::CosInplace,
    pub exp_op: vortx_shaders::ml::unary::ExpOp,
    pub exp_inplace: vortx_shaders::ml::unary::ExpInplace,
    pub reciprocal_op: vortx_shaders::ml::unary::ReciprocalOp,
    pub reciprocal_inplace: vortx_shaders::ml::unary::ReciprocalInplace,
    pub erf_op: vortx_shaders::ml::unary::ErfOp,
    pub erf_inplace: vortx_shaders::ml::unary::ErfInplace,
    pub pow_op: vortx_shaders::ml::unary::PowOp,
    pub pow_inplace: vortx_shaders::ml::unary::PowInplace,
}

/// Helper macro for dispatching inplace unary ops without args.
/// Each kernel has its own generated args type with the same fields.
macro_rules! dispatch_inplace_no_args {
    ($self:expr, $shapes:expr, $backend:expr, $pass:expr, $src:expr, $len:expr, $kernel:ident, $($args_ty:ident)::+) => {{
        #[cfg(not(feature = "push_constants"))]
        {
            $shapes.insert($backend, $src.layout())?;
            let shape_src = $shapes.get($src.layout()).unwrap();
            let mut src_buf = $src.buffer_mut();

            $self
                .$kernel
                .call($pass, [$len, 1, 1], &shape_src.as_slice(), &mut src_buf)
        }

        #[cfg(feature = "push_constants")]
        {
            let shapes_val = $src.layout().into();
            let mut src_buf = $src.buffer_mut();

            $self
                .$kernel
                .call($pass, [$len, 1, 1], &mut src_buf, shapes_val)
        }
    }};
}

/// Helper macro for dispatching inplace unary ops with args.
macro_rules! dispatch_inplace_with_args {
    ($self:expr, $shapes:expr, $backend:expr, $pass:expr, $src:expr, $args:expr, $len:expr, $kernel:ident, $($args_ty:ident)::+) => {{
        #[cfg(not(feature = "push_constants"))]
        {
            $shapes.insert($backend, $src.layout())?;
            let shape_src = $shapes.get($src.layout()).unwrap();
            let mut src_buf = $src.buffer_mut();

            $self.$kernel.call(
                $pass,
                [$len, 1, 1],
                &shape_src.as_slice(),
                &mut src_buf,
                $args.unwrap().buffer(),
            )
        }

        #[cfg(feature = "push_constants")]
        {
            let shapes_val = $src.layout().into();
            let mut src_buf = $src.buffer_mut();

            $self.$kernel.call(
                $pass,
                [$len, 1, 1],
                &mut src_buf,
                $args.unwrap().buffer(),
                shapes_val,
            )
        }
    }};
}

/// Helper macro for dispatching non-inplace unary ops without args.
macro_rules! dispatch_op_no_args {
    ($self:expr, $shapes:expr, $backend:expr, $pass:expr, $src:expr, $dest:expr, $len:expr, $kernel:ident, $($args_ty:ident)::+) => {{
        #[cfg(not(feature = "push_constants"))]
        {
            $shapes.insert($backend, $dest.layout())?;
            $shapes.insert($backend, $src.layout())?;
            let shape_dst = $shapes.get($dest.layout()).unwrap();
            let shape_src = $shapes.get($src.layout()).unwrap();
            let mut dst_buf = $dest.buffer_mut();

            $self.$kernel.call(
                $pass,
                [$len, 1, 1],
                &shape_src.as_slice(),
                &$src.buffer(),
                &shape_dst.as_slice(),
                &mut dst_buf,
            )
        }

        #[cfg(feature = "push_constants")]
        {
            let shapes_val = $src.layout().into();
            let mut dst_buf = $dest.buffer_mut();

            $self.$kernel.call(
                $pass,
                [$len, 1, 1],
                &$src.buffer(),
                &mut dst_buf,
                shapes_val,
            )
        }
    }};
}

/// Helper macro for dispatching non-inplace unary ops with args.
macro_rules! dispatch_op_with_args {
    ($self:expr, $shapes:expr, $backend:expr, $pass:expr, $src:expr, $dest:expr, $args:expr, $len:expr, $kernel:ident, $($args_ty:ident)::+) => {{
        #[cfg(not(feature = "push_constants"))]
        {
            $shapes.insert($backend, $dest.layout())?;
            $shapes.insert($backend, $src.layout())?;
            let shape_dst = $shapes.get($dest.layout()).unwrap();
            let shape_src = $shapes.get($src.layout()).unwrap();
            let mut dst_buf = $dest.buffer_mut();

            $self.$kernel.call(
                $pass,
                [$len, 1, 1],
                &shape_src.as_slice(),
                &$src.buffer(),
                &shape_dst.as_slice(),
                &mut dst_buf,
                &$args.unwrap().buffer().as_slice(),
            )
        }

        #[cfg(feature = "push_constants")]
        {
            let shapes_val = $src.layout().into();
            let mut dst_buf = $dest.buffer_mut();

            $self.$kernel.call(
                $pass,
                [$len, 1, 1],
                &$src.buffer(),
                &mut dst_buf,
                &$args.unwrap().buffer().as_slice(),
                shapes_val,
            )
        }
    }};
}

impl Unary {
    pub fn launch_inplace(
        &self,
        backend: &GpuBackend,
        #[cfg_attr(feature = "push_constants", allow(unused_variables))]
        shapes: &mut TensorLayoutBuffers,
        pass: &mut GpuPass,
        op: UnaryOp,
        mut src: impl AsTensorMut<f32>,
        args: Option<&Tensor<Vec4>>,
    ) -> Result<(), GpuBackendError> {
        let mut src = src.as_tensor_mut();
        let len = src.len() as u32;

        assert_eq!(
            op.has_args(),
            args.is_some(),
            "Unary ops argument mismatch."
        );

        match op {
            UnaryOp::Abs => dispatch_inplace_no_args!(
                self,
                shapes,
                backend,
                pass,
                src,
                len,
                abs_inplace,
                vortx_shaders::ml::unary::AbsInplaceArgs
            ),
            UnaryOp::Sgn => dispatch_inplace_no_args!(
                self,
                shapes,
                backend,
                pass,
                src,
                len,
                sgn_inplace,
                vortx_shaders::ml::unary::SgnInplaceArgs
            ),
            UnaryOp::Neg => dispatch_inplace_no_args!(
                self,
                shapes,
                backend,
                pass,
                src,
                len,
                neg_inplace,
                vortx_shaders::ml::unary::NegInplaceArgs
            ),
            UnaryOp::Step => dispatch_inplace_no_args!(
                self,
                shapes,
                backend,
                pass,
                src,
                len,
                step_inplace,
                vortx_shaders::ml::unary::StepInplaceArgs
            ),
            UnaryOp::Elu => dispatch_inplace_no_args!(
                self,
                shapes,
                backend,
                pass,
                src,
                len,
                elu_inplace,
                vortx_shaders::ml::unary::EluInplaceArgs
            ),
            UnaryOp::Gelu => dispatch_inplace_no_args!(
                self,
                shapes,
                backend,
                pass,
                src,
                len,
                gelu_inplace,
                vortx_shaders::ml::unary::GeluInplaceArgs
            ),
            UnaryOp::GeluQuick => dispatch_inplace_no_args!(
                self,
                shapes,
                backend,
                pass,
                src,
                len,
                gelu_quick_inplace,
                vortx_shaders::ml::unary::GeluQuickInplaceArgs
            ),
            UnaryOp::Silu => dispatch_inplace_no_args!(
                self,
                shapes,
                backend,
                pass,
                src,
                len,
                silu_inplace,
                vortx_shaders::ml::unary::SiluInplaceArgs
            ),
            UnaryOp::Tanh => dispatch_inplace_no_args!(
                self,
                shapes,
                backend,
                pass,
                src,
                len,
                tanh_inplace,
                vortx_shaders::ml::unary::TanhInplaceArgs
            ),
            UnaryOp::Relu => dispatch_inplace_no_args!(
                self,
                shapes,
                backend,
                pass,
                src,
                len,
                relu_inplace,
                vortx_shaders::ml::unary::ReluInplaceArgs
            ),
            UnaryOp::Sigmoid => dispatch_inplace_no_args!(
                self,
                shapes,
                backend,
                pass,
                src,
                len,
                sigmoid_inplace,
                vortx_shaders::ml::unary::SigmoidInplaceArgs
            ),
            UnaryOp::HardSigmoid => dispatch_inplace_no_args!(
                self,
                shapes,
                backend,
                pass,
                src,
                len,
                hard_sigmoid_inplace,
                vortx_shaders::ml::unary::HardSigmoidInplaceArgs
            ),
            // UnaryOp::HardSwish => dispatch_inplace_no_args!(self, shapes, backend, pass, src, len, hard_swish_inplace, vortx_shaders::ml::unary::HardSwishInplaceArgs),
            UnaryOp::Sqr => dispatch_inplace_no_args!(
                self,
                shapes,
                backend,
                pass,
                src,
                len,
                sqr_inplace,
                vortx_shaders::ml::unary::SqrInplaceArgs
            ),
            UnaryOp::Sqrt => dispatch_inplace_no_args!(
                self,
                shapes,
                backend,
                pass,
                src,
                len,
                sqrt_inplace,
                vortx_shaders::ml::unary::SqrtInplaceArgs
            ),
            UnaryOp::Sin => dispatch_inplace_no_args!(
                self,
                shapes,
                backend,
                pass,
                src,
                len,
                sin_inplace,
                vortx_shaders::ml::unary::SinInplaceArgs
            ),
            UnaryOp::Cos => dispatch_inplace_no_args!(
                self,
                shapes,
                backend,
                pass,
                src,
                len,
                cos_inplace,
                vortx_shaders::ml::unary::CosInplaceArgs
            ),
            UnaryOp::Log => dispatch_inplace_no_args!(
                self,
                shapes,
                backend,
                pass,
                src,
                len,
                log_inplace,
                vortx_shaders::ml::unary::LogInplaceArgs
            ),
            UnaryOp::Exp => dispatch_inplace_no_args!(
                self,
                shapes,
                backend,
                pass,
                src,
                len,
                exp_inplace,
                vortx_shaders::ml::unary::ExpInplaceArgs
            ),
            UnaryOp::Reciprocal => dispatch_inplace_no_args!(
                self,
                shapes,
                backend,
                pass,
                src,
                len,
                reciprocal_inplace,
                vortx_shaders::ml::unary::ReciprocalInplaceArgs
            ),
            UnaryOp::Erf => dispatch_inplace_no_args!(
                self,
                shapes,
                backend,
                pass,
                src,
                len,
                erf_inplace,
                vortx_shaders::ml::unary::ErfInplaceArgs
            ),
            UnaryOp::LeakyRelu => dispatch_inplace_with_args!(
                self,
                shapes,
                backend,
                pass,
                src,
                args,
                len,
                leaky_relu_inplace,
                vortx_shaders::ml::unary::LeakyReluInplaceArgs
            ),
            UnaryOp::Clamp => dispatch_inplace_with_args!(
                self,
                shapes,
                backend,
                pass,
                src,
                args,
                len,
                clamp_inplace,
                vortx_shaders::ml::unary::ClampInplaceArgs
            ),
            UnaryOp::Scale => dispatch_inplace_with_args!(
                self,
                shapes,
                backend,
                pass,
                src,
                args,
                len,
                scale_inplace,
                vortx_shaders::ml::unary::ScaleInplaceArgs
            ),
            UnaryOp::AddScalar => dispatch_inplace_with_args!(
                self,
                shapes,
                backend,
                pass,
                src,
                args,
                len,
                add_scalar_inplace,
                vortx_shaders::ml::unary::AddScalarInplaceArgs
            ),
            UnaryOp::Pow => dispatch_inplace_with_args!(
                self,
                shapes,
                backend,
                pass,
                src,
                args,
                len,
                pow_inplace,
                vortx_shaders::ml::unary::PowInplaceArgs
            ),
        }
    }

    pub fn launch(
        &self,
        backend: &GpuBackend,
        #[cfg_attr(feature = "push_constants", allow(unused_variables))]
        shapes: &mut TensorLayoutBuffers,
        pass: &mut GpuPass,
        op: UnaryOp,
        mut dest: impl AsTensorMut<f32>,
        src: impl AsTensorRef<f32>,
        args: Option<&Tensor<Vec4>>,
    ) -> Result<(), GpuBackendError> {
        let mut dest = dest.as_tensor_mut();
        let src = src.as_tensor_ref();
        let len = dest.len() as u32;

        assert_eq!(
            op.has_args(),
            args.is_some(),
            "Unary ops argument mismatch."
        );

        match op {
            UnaryOp::Abs => dispatch_op_no_args!(
                self,
                shapes,
                backend,
                pass,
                src,
                dest,
                len,
                abs_op,
                vortx_shaders::ml::unary::AbsOpArgs
            ),
            UnaryOp::Sgn => dispatch_op_no_args!(
                self,
                shapes,
                backend,
                pass,
                src,
                dest,
                len,
                sgn_op,
                vortx_shaders::ml::unary::SgnOpArgs
            ),
            UnaryOp::Neg => dispatch_op_no_args!(
                self,
                shapes,
                backend,
                pass,
                src,
                dest,
                len,
                neg_op,
                vortx_shaders::ml::unary::NegOpArgs
            ),
            UnaryOp::Step => dispatch_op_no_args!(
                self,
                shapes,
                backend,
                pass,
                src,
                dest,
                len,
                step_op,
                vortx_shaders::ml::unary::StepOpArgs
            ),
            UnaryOp::Elu => dispatch_op_no_args!(
                self,
                shapes,
                backend,
                pass,
                src,
                dest,
                len,
                elu_op,
                vortx_shaders::ml::unary::EluOpArgs
            ),
            UnaryOp::Gelu => dispatch_op_no_args!(
                self,
                shapes,
                backend,
                pass,
                src,
                dest,
                len,
                gelu_op,
                vortx_shaders::ml::unary::GeluOpArgs
            ),
            UnaryOp::GeluQuick => dispatch_op_no_args!(
                self,
                shapes,
                backend,
                pass,
                src,
                dest,
                len,
                gelu_quick_op,
                vortx_shaders::ml::unary::GeluQuickOpArgs
            ),
            UnaryOp::Silu => dispatch_op_no_args!(
                self,
                shapes,
                backend,
                pass,
                src,
                dest,
                len,
                silu_op,
                vortx_shaders::ml::unary::SiluOpArgs
            ),
            UnaryOp::Tanh => dispatch_op_no_args!(
                self,
                shapes,
                backend,
                pass,
                src,
                dest,
                len,
                tanh_op,
                vortx_shaders::ml::unary::TanhOpArgs
            ),
            UnaryOp::Relu => dispatch_op_no_args!(
                self,
                shapes,
                backend,
                pass,
                src,
                dest,
                len,
                relu_op,
                vortx_shaders::ml::unary::ReluOpArgs
            ),
            UnaryOp::Sigmoid => dispatch_op_no_args!(
                self,
                shapes,
                backend,
                pass,
                src,
                dest,
                len,
                sigmoid_op,
                vortx_shaders::ml::unary::SigmoidOpArgs
            ),
            UnaryOp::HardSigmoid => dispatch_op_no_args!(
                self,
                shapes,
                backend,
                pass,
                src,
                dest,
                len,
                hard_sigmoid_op,
                vortx_shaders::ml::unary::HardSigmoidOpArgs
            ),
            // UnaryOp::HardSwish => dispatch_op_no_args!(self, shapes, backend, pass, src, dest, len, hard_swish_op, vortx_shaders::ml::unary::HardSwishOpArgs),
            UnaryOp::Sqr => dispatch_op_no_args!(
                self,
                shapes,
                backend,
                pass,
                src,
                dest,
                len,
                sqr_op,
                vortx_shaders::ml::unary::SqrOpArgs
            ),
            UnaryOp::Sqrt => dispatch_op_no_args!(
                self,
                shapes,
                backend,
                pass,
                src,
                dest,
                len,
                sqrt_op,
                vortx_shaders::ml::unary::SqrtOpArgs
            ),
            UnaryOp::Sin => dispatch_op_no_args!(
                self,
                shapes,
                backend,
                pass,
                src,
                dest,
                len,
                sin_op,
                vortx_shaders::ml::unary::SinOpArgs
            ),
            UnaryOp::Cos => dispatch_op_no_args!(
                self,
                shapes,
                backend,
                pass,
                src,
                dest,
                len,
                cos_op,
                vortx_shaders::ml::unary::CosOpArgs
            ),
            UnaryOp::Log => dispatch_op_no_args!(
                self,
                shapes,
                backend,
                pass,
                src,
                dest,
                len,
                log_op,
                vortx_shaders::ml::unary::LogOpArgs
            ),
            UnaryOp::Exp => dispatch_op_no_args!(
                self,
                shapes,
                backend,
                pass,
                src,
                dest,
                len,
                exp_op,
                vortx_shaders::ml::unary::ExpOpArgs
            ),
            UnaryOp::Reciprocal => dispatch_op_no_args!(
                self,
                shapes,
                backend,
                pass,
                src,
                dest,
                len,
                reciprocal_op,
                vortx_shaders::ml::unary::ReciprocalOpArgs
            ),
            UnaryOp::Erf => dispatch_op_no_args!(
                self,
                shapes,
                backend,
                pass,
                src,
                dest,
                len,
                erf_op,
                vortx_shaders::ml::unary::ErfOpArgs
            ),
            UnaryOp::LeakyRelu => dispatch_op_with_args!(
                self,
                shapes,
                backend,
                pass,
                src,
                dest,
                args,
                len,
                leaky_relu_op,
                vortx_shaders::ml::unary::LeakyReluOpArgs
            ),
            UnaryOp::Clamp => dispatch_op_with_args!(
                self,
                shapes,
                backend,
                pass,
                src,
                dest,
                args,
                len,
                clamp_op,
                vortx_shaders::ml::unary::ClampOpArgs
            ),
            UnaryOp::Scale => dispatch_op_with_args!(
                self,
                shapes,
                backend,
                pass,
                src,
                dest,
                args,
                len,
                scale_op,
                vortx_shaders::ml::unary::ScaleOpArgs
            ),
            UnaryOp::AddScalar => dispatch_op_with_args!(
                self,
                shapes,
                backend,
                pass,
                src,
                dest,
                args,
                len,
                add_scalar_op,
                vortx_shaders::ml::unary::AddScalarOpArgs
            ),
            UnaryOp::Pow => dispatch_op_with_args!(
                self,
                shapes,
                backend,
                pass,
                src,
                dest,
                args,
                len,
                pow_op,
                vortx_shaders::ml::unary::PowOpArgs
            ),
        }
    }

    pub fn run_cpu<S: StorageMut<f32, Dyn>>(
        &self,
        op: UnaryOp,
        vals: &mut Vector<f32, Dyn, S>,
        args: Vec4,
    ) {
        vals.apply(|x| *x = op.eval(*x, args));
    }
}

#[cfg(test)]
#[cfg(feature = "rand")]
mod test {
    use crate::ml::UnaryOp;
    use khal_std::glamx::Vec4;
    use khal::backend::WebGpu;
    use khal::backend::{Backend, Encoder, GpuBackend};
    use khal::{BufferUsages, Shader};
    use nalgebra::DVector;
    use crate::shapes::TensorLayoutBuffers;
    use crate::tensor::Tensor;
    use wgpu::{Features, Limits};

    #[futures_test::test]
    #[serial_test::serial]
    async fn gpu_unary_ops_webgpu() {
        let webgpu = WebGpu::new(Features::default(), Limits::default())
            .await
            .unwrap();
        let backend = GpuBackend::WebGpu(webgpu);
        gpu_unary_ops_generic(&backend).await;
    }

    async fn gpu_unary_ops_generic(backend: &GpuBackend) {
        let unop = super::Unary::from_backend(backend).unwrap();

        let ops = [
            UnaryOp::Abs,
            UnaryOp::Sgn,
            UnaryOp::Neg,
            UnaryOp::Step,
            UnaryOp::Elu,
            UnaryOp::Gelu,
            UnaryOp::GeluQuick,
            UnaryOp::Silu,
            UnaryOp::Tanh,
            UnaryOp::Relu,
            UnaryOp::Sigmoid,
            UnaryOp::HardSigmoid,
            // UnaryOp::HardSwish,
            UnaryOp::Sqr,
            UnaryOp::Sqrt,
            UnaryOp::Sin,
            UnaryOp::Cos,
            UnaryOp::Log,
            UnaryOp::Exp,
            UnaryOp::Reciprocal,
            UnaryOp::Erf,
            UnaryOp::LeakyRelu,
            UnaryOp::Clamp,
            UnaryOp::Scale,
            UnaryOp::AddScalar,
            UnaryOp::Pow,
        ];

        for op in ops {
            let mut shapes = TensorLayoutBuffers::new(backend);

            println!("Checking {:?}", op);

            const LEN: u32 = 1757;

            let src = DVector::new_random(LEN as usize);
            let dst = DVector::zeros(LEN as usize);
            let mut dst_read = DVector::zeros(LEN as usize);
            let mut args = Vec4::new(
                rand::random(),
                rand::random(),
                rand::random(),
                rand::random(),
            );
            if args.y < args.x {
                let (x, y) = (args.x, args.y);
                args.x = y;
                args.y = x; // Ensure min <= max for clamp.
            }
            let gpu_args = op
                .has_args()
                .then(|| Tensor::scalar(backend, args, BufferUsages::STORAGE).unwrap());
            let gpu_src = Tensor::vector(backend, &src, BufferUsages::STORAGE).unwrap();
            let mut gpu_dst = Tensor::vector(
                backend,
                &dst,
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            )
            .unwrap();

            let mut encoder = backend.begin_encoding();
            let mut pass = encoder.begin_pass("test", None);
            unop.launch(
                backend,
                &mut shapes,
                &mut pass,
                op,
                &mut gpu_dst,
                gpu_src.as_view(),
                gpu_args.as_ref(),
            )
            .unwrap();
            drop(pass);

            backend.submit(encoder).unwrap();
            backend.synchronize().unwrap();

            backend
                .slow_read_buffer(gpu_dst.buffer(), dst_read.as_mut_slice())
                .await
                .unwrap();

            let mut cpu_result = src;
            unop.run_cpu(op, &mut cpu_result, args);

            approx::assert_relative_eq!(dst_read, cpu_result, epsilon = 1.0e-5);
        }
    }

    #[cfg(feature = "cpu")]
    #[futures_test::test]
    async fn gpu_unary_ops_cpu() {
        let backend = GpuBackend::Cpu;
        gpu_unary_ops_generic(&backend).await;
    }

    #[cfg(feature = "cuda")]
    #[futures_test::test]
    #[serial_test::serial]
    async fn gpu_unary_ops_cuda() {
        let cuda = khal::backend::Cuda::new(0).unwrap();
        let backend = GpuBackend::Cuda(cuda);
        gpu_unary_ops_generic(&backend).await;
    }
}
