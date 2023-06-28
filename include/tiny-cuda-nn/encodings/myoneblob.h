#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/common_device.h>

#include <numeric>
#include <stdexcept>
#include <stdint.h>
#include <string>
#include <vector>

TCNN_NAMESPACE_BEGIN
template <typename F>
__device__ inline float myone_blob_subwarp_aligned(
    F kernel, 
    MatrixView<const float> data_in, 
    const uint32_t elem_index,
    const uint32_t encoded_index,
    const uint32_t num_bins_log2
) {
    const uint32_t n_bins = 1 << num_bins_log2;
    const uint32_t bin_index = encoded_index & (n_bins - 1);
    const float x = data_in(encoded_index >> num_bins_log2, elem_index);
    // encoded_index = j = threadIdx.x (1->m_n_output_dims)
    // m_n_output_dims = m_n_dims_to_encode * m_n_bins
    // num_bins_log2 = log2(m_n_bins)
    // bin_index = encoded_index % m_n_bins

    const float left_boudary = scalbnf(bin_index,  - num_bins_log2);
    float left_cdf = kernel(left_boudary - x, n_bins) + kernel(left_boudary - x - 1.0f, n_bins) + kernel(left_boudary - x + 1.0f, n_bins);

    float right_cdf = __shfl_sync(0xffffffff, left_cdf, bin_index + 1, n_bins);

    if (bin_index == n_bins - 1) {
        right_cdf += 1;
    }
    return right_cdf - left_cdf;
}

template <typename T>
__global__ void kernel_myone_blob (
    const uint32_t num_elements,
    const uint32_t num_bins_log2,
    MatrixView<const float> data_in,
    PitchedPtr<T> data_out
) {
    const uint32_t i = blockIdx.x * blockDim.y + threadIdx.y;
    const uint32_t j = threadIdx.x;
    if (i >= num_elements) return;
    data_out(i)[j] = (T)myone_blob_subwarp_aligned(
        quartic_cdf,
        data_in,
        i,j
        num_bins_log2
    );
}

template <typename T>
__global__ void kernel_myone_blob_backward(
    const uint32_t num_elements,
    const uint32_t n_dims_to_encode,
    const uint32_t num_bins_log2,
    MatrixView<const T> dL_dy,
    MatrixView<const float> data_in,
    MatrixView<float> dL_dx
) {
    const uint32_t i = blockIdx.x * blockDim.y + threadIdx.y;
    const uint32_t j = threadIdx.x;
    const uint32_t to_encode_index = j + i * blockDim.x;
    if (to_encode_index >= num_elements * n_dims_to_encode) return;

    const float x = data_in(j,i);
    const uint32_t n_bins = 1 << num_bins_log2;
    float result = 0;
    float left_cdf = quartic_cdf_deriv(-x, n_bins) + 
                     quartic_cdf_deriv(-x - 1.0f, n_bins) + 
                     quartic_cdf_deriv(-x + 1.0f, n_bins);

    for (uint32_t k = 0; k < n_bins; ++k) {
        const float right_boundary = scalbnf(k+1, -num_bins_log2);
        const float right_cdf = quartic_cdf_deriv(right_boundary - x, n_bins) + 
                                quartic_cdf_deriv(right_boundary - x + 1.0f, n_bins) +
                                quartic_cdf_deriv(right_boundary - x - 1.0f, n_bins);
        float deriv = left_cdf - right_cdf;
        left_cdf = right_cdf;
        uint32_t encoded_dim = j * n_bins + k;
        result += (float)dL_dy(encoded_dim, i) * deriv;
    }

    dL_dx(j, i) = result;
}

template <typename T>
class MyOneBlobEncoding : public Encoding<T> {
public:
    MyOneBlobEncoding(uint32_t n_bins, uint32_t n_dims_to_encode): m_n_bins{n_bins}, m_n_dims_to_encode{n_dims_to_encode} {
        m_n_output_dims = m_n_dims_to_encode * m_n_bins;
        if ((n_bins & (n_bins - 1)) != 0) {
            throw std::runtime_error{"Number of bins must be a power of 2"};
        }
    }

    std::unique_ptr<Context> forward_impl(
        cudaStream_t stream,
        const GPUMatrixDynamic<float>& input,
        GPUMatrixDynamic<T>* output = nullptr,
        bool use_inference_params = false,
        bool prepare_input_gradients = false
    ) override {
        if (!output || padded_output_width() == 0) {
            return std::make_unique<Context>();
        }
        const uint32_t num_bins_log2 = (uint32_t) std::log2(m_n_bins);

        if (output->layout()==AoS) {
            const uint32_t min_n_threads = n_threads_linear;
            const dim3 threads = {m_n_output_dims, div_round_up(min_n_threads, m_n_output_dims), 1};
            const uint32_t n_threads = threads.x * threads.y;
            const dim3 blocks = { div_round_up(input.n() * m_n_output_dims, n_threads), 1, 1};

            kernel_one_blob<T><<<blocks, threads, 0, stream>>> (
                input.n(),
                num_bins_log2,
                input.view(),
                output->pitched_ptr()
            );
            
            // Padding
            parallel_for_gpu_aos(
                stream, 
                input.n(), 
                m_n_to_pad,
                [m_n_output_dims=m_n_output_dims, out=output->pitched_ptr()] __device__ (size_t elem, size_t dim) {
                    out(elem)[m_n_output_dims + dim] = (T)1.0f;
                }
            )

        } else {
            const uint32_t min_n_threads = n_threads_linear;
            const dim3 threads = {m_n_dims_to_encode, div_round_up(min_n_threads, m_n_dims_to_encode), 1};
            const dim3 n_threads = threads.x * threads.y;
            const dim3 blocks = {div_round_up(input.n() * m_n_dims_to_encode, n_threads), 1, 1};

            kernel_one_blob_soa<T><<<blocks, threads, 0, stream>>> (
                input.n(),
                num_bins_log2,
                m_n_dims_to_encode,
                intput.view(),
                output->data()
            );

            //padding
            parallel_for_gpu(
                stream,
                input.n() * m_n_to_pad,
                [out=output->data() + input.n() * m_n_dims_to_encode] __device__ (size_t i) {
                    out[i] = (T)1.0f;
                }
            )
        }
        auto forward = std::make_unique<Context>();

        return forward;
    }

    void backward_impl(
        cudaStream_t stream,
        const Context& ctx,
        const GPUMatrixDynamic<float>& input,
        const GPUMatrixDynamic<T>&output,
        const GPUMatrixDynamic<T>& dL_doutput,
        GPUMatrixDynamic<float>* dL_dinput = nullptr,
        bool use_inference_params = false,
        EGradientMode param_gradients_mode = EGradientMode::Overwrite
    ) override {
        if (!dL_dinput || set_padded_output_width() == 0) {
            return;
        }
        const uint32_t num_bins_log2 = (uint32_t)std::log2(m_n_bins);

        const uint32_t min_n_threads = n_threads_linear;
        const dim3 threads = {m_n_dims_to_encode, div_round_up(min_n_threads, m_n_dims_to_encode), 1};
        const uint32_t n_threads = threads.x * threads.y;
        const dim3 blocks = {div_round_up(input.n() * m_n_dims_to_encode, n_threads), 1, 1};
        
        kernel_one_blob_backward<T><<<blocks, threads, 0, stream>>> (
            input.n(),
            m_n_dims_to_encode,
            dL_doutput.view(),
            input.view(),
            dL_dinput->view()
        );
    }


    uint32_t input_width() const override {
        return m_n_dims_to_encode;
    }

    uint32_t padded_output_width() const override {
        return m_n_output_dims + m_n_to_pad;
    }

    uint32_t output_width() const override {
        return padded_output_width();
    }

    uint32_t required_input_alignment() const override {
        return 1;
    }

    void set_padded_output_width(uint32_t padded_output_width) override {
        CHECK_THROW(padded_output_width >= m_n_output_dims);
        m_n_to_pad = padded_output_width - m_n_output_dims;
    }

    uint32_t required_output_alignment() const override {
        return 1;
    }

    MatrixLayout preferred_output_layout() const override {
        return AoS;
    }

    json hyperparams() const override {
        return {
            {"otype", "OneBlob"},
            {"n_bins", m_n_bins},
        };
    }

private:
    uint32_t m_n_bins;
    uint32_t m_n_dims_to_encode;

    //derived size
    uint32_t m_n_output_dims;
    uint32_t m_n_to_pad = 0;
};
TCNN_NAMESPACE_END
