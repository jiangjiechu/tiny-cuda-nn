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

template <typename T>
__global__ void myfrequency_encoding(
    const uint32_t num_elements,
    const uint32_t n_frequencies,
    const uint32_t num_to_encode,
    const uint32_t num_to_pad,
    MatrixView<const float> data_in,
    MatrixView<T> data_out,
    float* __restrict__ dy_dx)
{
    const uint32_t encode_index = threadIdx.x + blockIdx.x*blockDim.x;
    if (encode_index >= num_elements) return;

    const uint32_t fan_out_encoded = num_to_encode * n_frequencies * 2;
    const uint32_t fan_out = fan_out_encoded + num_to_pad;

    const uint32_t i = encode_index / fan_out;
    const uint32_t j = encode_index - i * fan_out;

    /* Layout of outputs (for each input record):
     * frequency-encoded input dimension 0
     * frequency-encoded input dimension 1
     * frequency-encoded input dimension ...
     *  padding (value 1.f)
    */
   if (j>= fan_out_encoded) {
    data_out(j, i) = 1;
   } else {
     /* Layout of encoded features (e.g. when inputs abcd.. are XYZ positions):
      *     --> j
      *  |  sin(pi a.x), cos(pi a.x), sin(2pi a.x), cos(2pi a.x), sin(4pi a.x) ...
      *  |  sin(pi a.y), cos(pi a.y), sin(2pi a.y), cos(2pi a.y), sin(4pi a.y) ...
      *  v  sin(pi a.z), cos(pi a.z), sin(2pi a.z), cos(2pi a.z), sin(4pi a.z) ...
      *  i  (padding)
     */
    const uint32_t encoded_input_freature_i = j / (n_frequencies * 2);
    const uint32_t log2_freqency = (j/2) % n_frequencies;

    const float phase_shift = (j % 2) * (PI/2);

    const float x = scalbnf(data_in(encoded_input_freature_i, i), log2_freqency);
    const float input = x * PI + phase_shift;
    data_out(j,i) = (T)__sinf(input);
    if(dy_dx != nullptr) {
        dy_dx[i*fan_out_encoded + j] = scalbnf(1.0f, log2_freqency) * PI * __cosf(input);
    }
   }
}

template <typename T>
__global__ void myfrequency_encoding_backward(
    const uint32_t num_elements,
    const uint32_t n_dims_to_encode,
    const uint32_t n_frequencies,
    MatrixView<const T> dL_dy,
    const float* dy_dx,
    MatrixView<float> dL_dx
) {
    const uint32_t encode_index = threadIdx.x + blockIdx.x*blockDim.x;
    if (encoded_index >= num_elements) return;

    const uint32_t i = encode_index / n_dims_to_encode;
    const uint32_t j = encode_index - i * n_dims_to_encode;

    const uint32_t output_per_input = n_frequencies * 2;

    float result = 0;
    for (int k = 0; k < output_per_input; ++k) {
        result += (float)dL_dy(j * output_per_input + k, i) * dy_dx[i * n_dims_to_encode]
    }
}

template <typename T>
class MyFrequencyEncoding : public Encoding<T> {
public:
    MyFrequencyEncoding(uint32_t n_frequencies, uint32_t m_n_dims_to_encode)
    : m_n_frequencies{n_frequencies}, m_n_dims_to_encode{m_n_dims_to_encode} {
        m_n_output_dims = m_n_dims_to_encode * m_n_frequencies * 2;
    }

    std::unique_ptr<Contex> forward_impl(
        cudaStream_t stream,
        const GPUMatrixDynamic<float>& input,
        GPUMatrixDynamic<T>* output = nullptr,
        bool use_inference_params = false,
        bool prepare_input_gradients = false
    ) override {
        auto forward = std::make_unique<ForwardContext>();

        if (!output || set_padded_output_width() == 0) {
            return forward;
        }

        if (prepare_input_gradients) {
            forward->dy_dx = GPUMatrix<float>{m_n_dims_to_encode * m_n_frequencies * 2, input.n(), stream};
        }

        linear_kernel(myfrequency_encoding<T>, 0, stream,
            input.n() * padded_output_width(),
            m_n_frequencies,
            m_n_dims_to_encode,
            m_n_to_pad,
            input.view(),
            output->view(),
        );
        return forward;
    }

    void backward_impl(
        cudaStream_t stream,
        const Context& ctx,
        const GPUMatrixDynamic<float>& input,
        const GPUMatrixDynamic<T>& output,
        const GPUMatrixDynamic<T>& DL_doutput,
        GPUMatrixDynamic<float>* dL_dinput = nullptr,
        bool use_inference_parameters = false,
        EGradientMode param_gradients_mode = EGradientMode::Overwrite
    ) override {
        if (!dL_dinput || padded_output_width() == 0) {
            return;
        }

        const auto& forward = dynamic_cast<const ForwardContext&> (ctx);
        linear_kernel(myfrequency_encoding_backward<T>, 0, stream,
            input.n(), m_n_dims_to_encode,
            m_n_dims_to_encode,
            dL_doutput.view(),
            forward.dy_dx.data(),
            dL_dinput->view()
        )
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

    uint32_t required_output_alignment() const override {
        return 1;
    }

    MatrixLayout preferred_output_layout() const override {
        return AoS;
    }

    json hyperparameters() const override {
        return {
            {"otype", "MyFrequency"},
            {"n_frequencies", m_n_frequencies},

        }
    }


private:
    struct ForwardContext : public Context {
        GPUMatrix<float> dy_dx;
    };
    
    uint32_t m_n_frequencies;
    uint32_t m_n_dims_to_encode;

    //derived sizes
    uint32_t m_n_output_dims;
    uint32_t m_n_to_pad = 0;
};
TCNN_NAMESPACE_END