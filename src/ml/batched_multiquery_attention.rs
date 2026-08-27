use crate::ml::SoftMax;
use crate::tensor::{AsTensorMut, AsTensorRef};
use khal::Shader;
use khal::backend::{GpuBackend, GpuBackendError, GpuPass};
use nalgebra::{DMatrix, DVector};
use vortx_shaders::ml::AttentionParams;

#[derive(Shader)]
/// Fused attention shader: Q*K^T, scale, mask, softmax and *V in one kernel.
pub struct FusedAttention {
    pub fused_attention: vortx_shaders::ml::FusedAttention,
    pub fused_attention_online: vortx_shaders::ml::FusedAttentionOnline,
    pub flash_attention: vortx_shaders::ml::FlashAttention,
}

impl FusedAttention {
    /// Launch the fused attention kernel.
    pub fn launch(
        &self,
        _backend: &GpuBackend,
        pass: &mut GpuPass,
        params: &AttentionParams,
        params_gpu: impl AsTensorRef<AttentionParams>,
        q: impl AsTensorRef<f32>,
        key_cache: impl AsTensorRef<f32>,
        value_cache: impl AsTensorRef<f32>,
        mut xb: impl AsTensorMut<f32>,
    ) -> Result<(), GpuBackendError> {
        const WORKGROUP_SIZE: u32 = 128;
        const MAX_SEQ_LEN: u32 = 2048;

        let params_gpu = params_gpu.as_tensor_ref();
        let q = q.as_tensor_ref();
        let key_cache = key_cache.as_tensor_ref();
        let value_cache = value_cache.as_tensor_ref();
        let mut xb = xb.as_tensor_mut();

        let n_heads = params.n_heads;
        let seq_len = params.pos + 1;

        // Choose kernel based on sequence length:
        // - fused_attention: stores all scores in shared memory (fast, limited to 2048 tokens)
        // - flash_attention: block-wise processing with online softmax (efficient for long sequences)
        if seq_len <= MAX_SEQ_LEN {
            let mut buf_out = xb.buffer_mut();
            self.fused_attention.call(
                pass,
                [n_heads * WORKGROUP_SIZE, 1, 1],
                &params_gpu.buffer(),
                &q.buffer(),
                &key_cache.buffer(),
                &value_cache.buffer(),
                &mut buf_out,
            )
        } else {
            let mut buf_out = xb.buffer_mut();
            self.flash_attention.call(
                pass,
                [n_heads * WORKGROUP_SIZE, 1, 1],
                &params_gpu.buffer(),
                &q.buffer(),
                &key_cache.buffer(),
                &value_cache.buffer(),
                &mut buf_out,
            )
        }
    }

    pub fn run_cpu(
        params: &AttentionParams,
        q: &DVector<f32>,
        key_cache: &DMatrix<f32>,
        value_cache: &DMatrix<f32>,
        attn: &mut DMatrix<f32>,
        xb: &mut DVector<f32>,
    ) {
        // The number of embedding vector elements associated to each query head.
        let head_size = params.head_size as usize;
        // The number of query head associated to one key/value head.
        let kv_mul = params.kv_mul as usize;

        // Multihead attention. Iterate over all head.
        // TODO: in llama2.c, each head is iterated on in parallel.
        for h in 0..params.n_heads as usize {
            // Get the query vector for this head.
            let q = q.rows(h * head_size, head_size);
            // Attention scores for this head.
            let mut att = attn.column_mut(h);

            // Iterate over all timesteps (tokens in the sequence), including the current one, but
            // not past the current one due to causality.
            // See the KV cache explanation there: https://youtu.be/Mn_9W1nCFLo?si=3n4GH9f2OzMb5Np0&t=2940
            // -> This is iterating through all the green columns (from K^t) that are the rotated
            //    (by RoPE). The values set in this loop into the `att` variable here (attention
            //    scores) are the elements in the pink row (at the bottom of the QK^t matrix) divide
            //    by sqrt(params.head_size) (in other words, this is what's given to softmax afterward.
            for t in 0..=params.pos as usize {
                // Get the key vector for this head and at this timestep.
                let k = key_cache.column(t); // TODO: does key_cache have the right dim?
                let k_head = k.rows((h / kv_mul) * head_size, head_size);

                // Calculate the attention score as the dot product of q and k.
                let mut score = q.dot(&k_head);
                score /= (head_size as f32).sqrt();
                // Save the score to the attention buffer.
                att[t] = score;
            }

            // Softmax the scores to get attention weights from 0..=pos inclusively.
            SoftMax::run_cpu(&mut att.rows_mut(0, params.pos as usize + 1));

            // Weighted sum of the values, store back into xb.
            // /!\ xb is now changing semantic, storing the weighted sums for all the heads.
            //       Now xb contains the "Attention 4" row from https://youtu.be/Mn_9W1nCFLo?si=550ar5aUg1I1k60l&t=2940.
            let mut xb = xb.rows_mut(h * head_size, head_size);
            xb.fill(0.0);
            for t in 0..=params.pos as usize {
                let v = value_cache.column(t);
                let v_head = v.rows((h / kv_mul) * head_size, head_size);
                xb.axpy(att[t], &v_head, 1.0);
            }
        }
    }
}

/*
#[cfg(test)]
mod test {
    use crate::ml::ops::{AttentionParams, SoftMax};
    use nalgebra::{DMatrix, DVector};
    use khal::gpu::GpuInstance;
    use khal::kernel::CommandEncoderExt;
    use crate::shapes::TensorLayoutBuffers;
    use crate::tensor::{Tensor, Tensor, Tensor};
    use khal::Shader;
    use crate::Gemv;
    use wgpu::BufferUsages;

    #[futures_test::test]
    #[serial_test::serial]
    async fn gpu_attention() {
        let gpu = GpuInstance::new().await.unwrap();
        let batched_multihead_attention =
            super::BatchedMultiqueryAttention::from_backend(gpu.backend()).unwrap();
        let mut encoder = gpu.backend().create_command_encoder(&Default::default());

        // let mut params = AttentionParams { seq_len: 131072, kv_dim: 256, kv_mul: 6, n_heads: 12, head_size: 128, pos: 9 };
        let params = AttentionParams {
            seq_len: 1024,
            kv_dim: 768,
            kv_mul: 1,
            n_heads: 12,
            head_size: 64,
            pos: 6,
        };

        let q = DVector::new_random((params.n_heads * params.head_size) as usize);
        let key_cache = DMatrix::new_random(params.kv_dim as usize, params.seq_len as usize);
        let value_cache = DMatrix::new_random(params.kv_dim as usize, params.seq_len as usize);
        let mut attn = DMatrix::zeros(params.seq_len as usize, params.n_heads as usize);
        let mut xb = DVector::zeros((params.n_heads * params.head_size) as usize);

        let gpu_params = Tensor::scalar(gpu.backend(), params, BufferUsages::UNIFORM);
        let gpu_q = Tensor::vector(gpu.backend(), q.as_slice(), BufferUsages::STORAGE);
        let gpu_key_cache = Tensor::matrix(gpu.backend(), &key_cache, BufferUsages::STORAGE);
        let gpu_value_cache = Tensor::matrix(gpu.backend(), &value_cache, BufferUsages::STORAGE);
        let gpu_attn = Tensor::matrix(
            gpu.backend(),
            &attn,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );
        let gpu_xb = Tensor::vector(
            gpu.backend(),
            xb.as_slice(),
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );

        let gpu_staging_xb = Tensor::vector_uninit(
            gpu.backend(),
            xb.len() as u32,
            BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        );
        let gpu_staging_attn = Tensor::matrix_uninit(
            gpu.backend(),
            attn.nrows() as u32,
            attn.ncols() as u32,
            BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        );

        let mut pass = encoder.compute_pass("test", None);
        batched_multihead_attention.launch(
            gpu.backend(),
            &mut pass,
            params.n_heads,
            &gpu_params,
            &gpu_q,
            &gpu_key_cache,
            &gpu_value_cache,
            &gpu_attn,
            &gpu_xb,
        );
        drop(pass);

        gpu_staging_xb.copy_from(&mut encoder, &gpu_xb);
        gpu_staging_attn.copy_from(&mut encoder, &gpu_attn);

        gpu.queue().submit(Some(encoder.finish()));

        super::FudedAttention::run_cpu(
            &params,
            &q,
            &key_cache,
            &value_cache,
            &mut attn,
            &mut xb,
        );

        approx::assert_relative_eq!(
            DVector::from(gpu_staging_xb.read(gpu.backend()).await.unwrap()),
            xb,
            epsilon = 1.0e-5
        );

        approx::assert_relative_eq!(
            DMatrix::from_vec(
                attn.nrows(),
                attn.ncols(),
                gpu_staging_attn.read(gpu.backend()).await.unwrap()
            ),
            attn,
            epsilon = 1.0e-5
        );
    }

    #[futures_test::test]
    #[serial_test::serial]
    async fn gpu_attention_multi() {
        let gpu = GpuInstance::new().await.unwrap();
        let batched_multihead_attention =
            super::BatchedMultiqueryAttention::from_backend(gpu.backend()).unwrap();
        let shapes = TensorLayoutBuffers::new();
        let matmul = Gemv::from_backend(gpu.backend()).unwrap();
        let softmax = SoftMax::from_backend(gpu.backend()).unwrap();

        // let mut params = AttentionParams { seq_len: 131072, kv_dim: 256, kv_mul: 6, n_heads: 12, head_size: 128, pos: 0 };
        let mut params = AttentionParams {
            seq_len: 1024,
            kv_dim: 768,
            kv_mul: 1,
            n_heads: 12,
            head_size: 64,
            pos: 0,
        };

        let q = DVector::new_random((params.n_heads * params.head_size) as usize);
        let key_cache = DMatrix::new_random(params.kv_dim as usize, params.seq_len as usize);
        let value_cache = DMatrix::new_random(params.kv_dim as usize, params.seq_len as usize);
        let mut attn = DMatrix::zeros(params.seq_len as usize, params.n_heads as usize);
        let mut xb = DVector::zeros((params.n_heads * params.head_size) as usize);

        let gpu_q = Tensor::vector(gpu.backend(), q.as_slice(), BufferUsages::STORAGE);
        let gpu_key_cache = Tensor::matrix(gpu.backend(), &key_cache, BufferUsages::STORAGE);
        let gpu_value_cache = Tensor::matrix(gpu.backend(), &value_cache, BufferUsages::STORAGE);
        let gpu_attn = Tensor::matrix(
            gpu.backend(),
            &attn,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );
        let gpu_xb = Tensor::vector(
            gpu.backend(),
            xb.as_slice(),
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );

        let gpu_staging_xb = Tensor::vector_uninit(
            gpu.backend(),
            xb.len() as u32,
            BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        );
        let gpu_staging_attn = Tensor::matrix_uninit(
            gpu.backend(),
            attn.nrows() as u32,
            attn.ncols() as u32,
            BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        );

        for pos in 0..9 {
            let mut encoder = gpu.backend().create_command_encoder(&Default::default());
            params.pos = pos;

            let gpu_params = Tensor::scalar(gpu.backend(), params, BufferUsages::UNIFORM);

            let mut pass = encoder.compute_pass("test", None);
            batched_multihead_attention.launch(
                gpu.backend(),
                &shapes,
                gpu.queue(),
                &mut pass,
                &matmul,
                &softmax,
                &params,
                &gpu_params,
                &gpu_q,
                &gpu_key_cache,
                &gpu_value_cache,
                &gpu_attn,
                &gpu_xb,
            );
            drop(pass);

            gpu_staging_xb.copy_from(&mut encoder, &gpu_xb);
            gpu_staging_attn.copy_from(&mut encoder, &gpu_attn);

            gpu.queue().submit(Some(encoder.finish()));

            super::BatchedMultiqueryAttention::run_cpu(
                &params,
                &q,
                &key_cache,
                &value_cache,
                &mut attn,
                &mut xb,
            );

            // NOTE: we can't compare attn since they don't have the same layout.
            // approx::assert_relative_eq!(
            //     DMatrix::from_vec(
            //         attn.nrows(),
            //         attn.ncols(),
            //         gpu_staging_attn.read(gpu.backend()).await.unwrap()
            //     ),
            //     attn,
            //     epsilon = 1.0e-5
            // );

            approx::assert_relative_eq!(
                DVector::from(gpu_staging_xb.read(gpu.backend()).await.unwrap()),
                xb,
                epsilon = 1.0e-5
            );
        }
    }
}
*/
