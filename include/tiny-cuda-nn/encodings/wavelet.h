#pragma once
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/multi_stream.h>
#include <tiny-cuda-nn/random.h>

#include <stdexcept>
#include <stdint.h>
#include <string>
#include <vector>

// #include <tiny-cuda-nn/encodings/grid.h>

TCNN_NAMESPACE_BEGIN
struct WaveletOffsetTable {
    uint32_t data[MAX_N_LEVELS+1] = {};
    uint32_t size = 0;
};
enum class WaveFamily {
    haar,
    db2,
    db3,
    db4,
};

//Upsampling followed by convolution
__host__ void upconv(const float* coef_in, 
                       uint32_t coef_len, 
                       const float* filter_lo, 
                       const float* filter_hi, 
                       const uint32_t filter_len, 
                       float*& result,
                       uint32_t& result_len,
                       const uint32_t level,
                       bool rec_a) 
{
    uint32_t temp_len = 0;
    const float* coef = coef_in;
    for (uint32_t l=0;l<level;++l) {
        temp_len = 2 * coef_len + filter_len - 2;
        float* temp = new float[temp_len];
        for (uint32_t i = 0; i < temp_len; ++i) {
            temp[i] = 0.f;
        }
        uint32_t o = 0;
        const float* filter = (rec_a || l>0) ? filter_lo : filter_hi;
        for (uint32_t i = 0; i < min(coef_len, (filter_len / 2)); ++i, o+=2) {
            for (uint32_t j = 0; j <= i; ++j) {
                temp[o] += filter[j*2] * coef[i-j];
                temp[o+1] += filter[j*2+1] * coef[i-j];
            }
        }
        for (uint32_t i = min(coef_len, (filter_len / 2)); i<coef_len; ++i, o+=2) {
            for (uint32_t j = 0; j < (filter_len / 2); ++j) {
                temp[o] += filter[j*2] * coef[i-j];
                temp[o+1] += filter[j*2+1] * coef[i-j];
            }
        }
        for (uint32_t i = coef_len; i < (filter_len / 2); ++i, o+=2) {
            for (uint32_t j = i - (coef_len-1); j < i+1; ++j) {
                temp[o] += filter[j*2] * coef[i-j];
                temp[o+1] += filter[j*2+1] * coef[i-j];
            }
        }
        for (uint32_t i = max(coef_len, filter_len); i < coef_len+(filter_len/2); ++i, o+=2) {
            for (uint32_t j = i - (coef_len-1); j < (filter_len/2); ++j) {
                temp[o] += filter[j*2] * coef[i-j];
                temp[o+1] += filter[j*2+1] * coef[i-j];
            }
        }
        coef = temp;
        coef_len = temp_len;
    }
    result_len = (1 << level) * (filter_len - 1);
    result_len = result_len > coef_len + 2 ? result_len : coef_len + 2;
    result = new float[result_len];
    for (uint32_t i =0;i < result_len;i++) {
        result[i] = (i >0 || i < coef_len) ? coef[i] : 0;
    }
}


template <WaveFamily WAVE_TYPE>
inline __device__ uint32_t get_wave_length() {
    switch (WAVE_TYPE) {
        case WaveFamily::haar: return 1;
        case WaveFamily::db2: return 3;
        case WaveFamily::db4: return 7;
    }
}

inline WaveFamily string_to_wavelet(const std::string& wave_type) {
    if (equals_case_insensitive(wave_type, "haar")) {
        return WaveFamily::haar;
    } else if (equals_case_insensitive(wave_type, "db2")) {
        return WaveFamily::db2;
    } else if (equals_case_insensitive(wave_type, "db4")) {
        return WaveFamily::db4;
    }
    throw std::runtime_error{fmt::format("Wavelet type {} not implemented", wave_type)};
}

inline std::string to_string(WaveFamily wave_type) {
    switch (wave_type) {
        case WaveFamily::haar : return "haar";
        case WaveFamily::db2 : return "db2";
        case WaveFamily::db4 : return "db4";
        default: throw std::runtime_error{"Invalid wavelet family"};
    }
}

template <uint32_t N_DIMS, HashType HASH_TYPE>
__device__ uint32_t wave_hash(const uint32_t pos_grid[N_DIMS]) {
	switch (HASH_TYPE) {
		case HashType::Prime: return prime_hash<N_DIMS>(pos_grid);
		case HashType::CoherentPrime: return coherent_prime_hash<N_DIMS>(pos_grid);
		case HashType::ReversedPrime: return reversed_prime_hash<N_DIMS>(pos_grid);
		case HashType::Rng: return rng_hash<N_DIMS>(pos_grid);
	}
	return 0;
}

__host__ void calc_haar (const uint32_t level, bool recon_a, float*& result, uint32_t& result_len) {
    constexpr float rec_lo[2] = { 0.7071067811865476, 0.7071067811865476};
    constexpr float rec_hi[2] = { 0.7071067811865476,-0.7071067811865476};
    constexpr uint32_t filter_len = 2;

    float coef[1] = {powf(2,level / 2.0f)};
    uint32_t coef_len = 1;

    upconv(coef, coef_len, rec_lo, rec_hi, filter_len, result, result_len, level,recon_a);
}
__host__ void calc_db2 (const uint32_t level, bool recon_a, float*& result, uint32_t& result_len) {
    constexpr float rec_lo[4] = {0.48296291314453416, 0.8365163037378079, 0.2241438680420134, -0.12940952255126037};
    constexpr float rec_hi[4] = {-0.12940952255126037, -0.2241438680420134, 0.8365163037378079, -0.48296291314453416};

    constexpr uint32_t filter_len = 4;

    float coef[1] = {powf(2,level / 2.0f)};
    uint32_t coef_len = 1;

    upconv(coef, coef_len, rec_lo, rec_hi, filter_len, result, result_len, level,recon_a);
}
__host__ void calc_db4 (const uint32_t level, bool recon_a, float*& result, uint32_t& result_len) {
    constexpr float rec_lo[8] = {0.33267055295008263, 0.8068915093110925, 0.45987750211849154, -0.13501102001025458, -0.08544127388202666, 0.03522629188570953};
    constexpr float rec_hi[8] = {0.03522629188570953, 0.08544127388202666, -0.13501102001025458, -0.45987750211849154, 0.8068915093110925, -0.33267055295008263};

    constexpr uint32_t filter_len = 8;

    float coef[1] = {powf(2,level / 2.0f)};
    uint32_t coef_len = 1;

    upconv(coef, coef_len, rec_lo, rec_hi, filter_len, result, result_len, level,recon_a);
}

template <WaveFamily WAVE_TYPE>
__host__ void calc_wavefun (const uint32_t level, bool recon_a, float*& result, uint32_t& result_len) {
    if (WAVE_TYPE==WaveFamily::haar) {
        calc_haar(level, recon_a, result, result_len);
    }else if (WAVE_TYPE==WaveFamily::db2) {
        calc_db2(level, recon_a, result, result_len);
    }else if (WAVE_TYPE==WaveFamily::db4) {
        calc_db4(level, recon_a, result, result_len);
    }else {
        std::runtime_error{"Wavelet not implemented"};
    }
}

template <uint32_t N_DIMS, HashType HASH_TYPE>
__device__ uint32_t wave_index(const uint32_t hashmap_size, const uint32_t grid_resolution, const uint32_t pos_grid[N_DIMS], const uint32_t idx_hl) {
	uint32_t index = 0;

	// The second part of the loop condition is needed to avoid integer overflows in finer levels.
    index = grid_hash<N_DIMS, HASH_TYPE>(pos_grid);

	return index % hashmap_size + (idx_hl-1) * hashmap_size;
}

template <typename T, uint32_t N_POS_DIMS, uint32_t N_FEATURES_PER_LEVEL,WaveFamily WAVE_TYPE, HashType HASH_TYPE> 
__global__ void kernel_wavelet(
    const uint32_t num_elements,
    const uint32_t num_grid_features,
    const WaveletOffsetTable offset_table,
    const uint32_t base_resolution,
    float max_level,
    const float* __restrict__ max_level_gpu,
    const InterpolationType interpolation_type,
    const T* __restrict__ grid,
    MatrixView<const float> positions_in,
    T* __restrict__ encoded_positions,
    float* __restrict__ dy_dx,
    const float* phi_data,
    const float* psi_data,
    const uint32_t wave_data_len
) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= num_elements) return;

    const uint32_t level = blockIdx.y;
    if (max_level_gpu) {
        max_level = (max_level_gpu[i] * num_grid_features) / N_FEATURES_PER_LEVEL;
    } else { 
        max_level  = (max_level * num_grid_features) / N_FEATURES_PER_LEVEL;
    }

    if (level>= max_level + 1e-3f) {
        if (encoded_positions) {
            for (uint32_t f = 0; f < N_FEATURES_PER_LEVEL; ++f) { 
                encoded_positions[i + (level*N_FEATURES_PER_LEVEL + f) * num_elements] = (T) 0.0f;
            }
        }
        if (dy_dx) { 
            //wavelet layer will only be the first layer, thus no dy_dx
            for (uint32_t f = 0; f < N_FEATURES_PER_LEVEL; ++f) {
                ((vector_fullp_t<N_POS_DIMS>*) dy_dx)[i + (level * N_FEATURES_PER_LEVEL + f) * num_elements] = {0.0f};
            }
        }
        return;
    }

    grid += offset_table.data[level] * N_FEATURES_PER_LEVEL;
    const uint32_t hashmap_size = (offset_table.data[level + 1] - offset_table.data[level]) / ((1<<N_POS_DIMS) - 1);
    const float scale = grid_scale(level, 1, base_resolution);
    const uint32_t resolution = grid_resolution(scale);
    const uint32_t wave_length = get_wave_length<WAVE_TYPE>();

    float pos[N_POS_DIMS];
    float pos_derivative[N_POS_DIMS];
    uint32_t pos_grid[N_POS_DIMS];

    TCNN_PRAGMA_UNROLL
    for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
        pos_fract(positions_in(dim, i), &pos[dim], &pos_derivative[dim], &pos_grid[dim], scale, identity_fun, identity_derivative);
    }

    auto wave_grid_val = [&] (const uint32_t local_pos[N_POS_DIMS], const uint32_t idx_hl) {
        const uint32_t index = wave_index<N_POS_DIMS, HASH_TYPE> (hashmap_size, resolution, local_pos, idx_hl) * N_FEATURES_PER_LEVEL;
        return *(vector_t<T, N_FEATURES_PER_LEVEL>*)&grid[index];
    };

    auto phi = [&] (const float x) {
        float x_in = (float) x / wave_length * wave_data_len;
        int x_idx = int(x_in);
        if (interpolation_type==InterpolationType::Nearest) {
            return phi_data[x_idx];
        }
        float _x = x_in - x_idx;
        if (x_idx < 0 || x_idx >= wave_data_len-1) {
            return 0.f;
        }
        return (float)(phi_data[x_idx] * (1.0-_x) + phi_data[x_idx+1] * _x);
    };
    auto psi = [&] (const float x) {
        float x_in = (float) x / wave_length * wave_data_len;
        int x_idx = int(x_in);
        if (interpolation_type==InterpolationType::Nearest) {
            return phi_data[x_idx];
        }
        float _x = x_in - x_idx;
        if (x_idx < 0 || x_idx >= wave_data_len-1) {
            return 0.f;
        }
        return (float)(psi_data[x_idx] * (1.0-_x) + psi_data[x_idx+1] * _x);
    };

	if (encoded_positions) {
		// N-linear interpolation
		vector_t<T, N_FEATURES_PER_LEVEL> result = {};

        if (level==0) {
            for (uint32_t step=0; step < wave_length; ++step) {
                float weight = 1.0f;
                float temp_x;
                uint32_t pos_grid_local[N_POS_DIMS];
                for (uint32_t dim=0; dim < N_POS_DIMS; ++dim) {
                    temp_x = pos[dim] + step;
                    weight *= phi(temp_x);
                    pos_grid_local[dim] = pos_grid[dim] - step;
                }
                result = fma((T)weight, wave_grid_val(pos_grid_local, 1), result);
                TCNN_PRAGMA_UNROLL
                for (uint32_t f = 0; f < N_FEATURES_PER_LEVEL; ++f) {
                    encoded_positions[i + (level * N_FEATURES_PER_LEVEL + f) * num_elements] = result[f];
                }
            }
        } else {
            for (uint32_t idx_hl = 1; idx_hl < (1 << N_POS_DIMS); ++idx_hl) {
                // idx_hl binarilly encodes L or H in different dimensions.
                // 000 -> LLL
                // 001 -> LLH
                // ...
                // 111 -> HHH
                for (uint32_t step=0; step < wave_length; ++step) {
                    float weight = 1.0f;
                    float scaling;
                    float temp_x;
                    uint32_t pos_grid_local[N_POS_DIMS];
                    for (uint32_t dim=0; dim < N_POS_DIMS; ++dim) {
                        if ((idx_hl & (1<<dim)) ==0) {
                            scaling = 0.5;
                            temp_x = pos[dim] * scaling + step;
                            weight *= phi(temp_x);
                        } else {
                            scaling = 1.0f;
                            temp_x = pos[dim] * scaling + step;
                            weight *= psi(temp_x);
                        }
                        pos_grid_local[dim] = pos_grid[dim] - step / scaling;
                    }
                    result = fma((T)weight, wave_grid_val(pos_grid_local, idx_hl), result);
                    TCNN_PRAGMA_UNROLL
                    for (uint32_t f = 0; f < N_FEATURES_PER_LEVEL; ++f) {
                        encoded_positions[i + (level * N_FEATURES_PER_LEVEL + f) * num_elements] = result[f];
                    }
                }
            }

        }
	}

    //Gradient
    if (dy_dx) {
        //wavelet layer is not intended to backward to any other layers
    }
}

template <typename T, typename GRAD_T, uint32_t N_POS_DIMS, uint32_t N_FEATURES_PER_LEVEL, uint32_t N_FEATURES_PER_THREAD, WaveFamily WAVE_TYPE, HashType HASH_TYPE>
__global__ void kernel_wavelet_backward(
    const uint32_t num_elements,
    const uint32_t num_grid_features,
    const WaveletOffsetTable offset_table,
    const uint32_t base_resolution,
    float max_level,
    const float* __restrict__ max_level_gpu,
    const InterpolationType interpolation_type,
    GRAD_T* __restrict__ grid_gradient,
    MatrixView<const float> positions_in,
    const T* __restrict__ dL_dy,
    const float* phi_data,
    const float* psi_data,
    const uint32_t wave_data_len
) {
	const uint32_t i = ((blockIdx.x * blockDim.x + threadIdx.x) * N_FEATURES_PER_THREAD) / N_FEATURES_PER_LEVEL;
	if (i >= num_elements) return;

	const uint32_t level = blockIdx.y ; // <- the level is the same for all threads.
	const uint32_t feature = (blockIdx.x * blockDim.x + threadIdx.x) * N_FEATURES_PER_THREAD - i * N_FEATURES_PER_LEVEL;

	if (max_level_gpu) {
		max_level = (max_level_gpu[i] * num_grid_features) / N_FEATURES_PER_LEVEL;
	} else {
		max_level = (max_level * num_grid_features) / N_FEATURES_PER_LEVEL;
	}

	if (level > max_level + 1e-3f) {
		return;
	}
    grid_gradient += offset_table.data[level] * N_FEATURES_PER_LEVEL;
    const uint32_t hashmap_size = (offset_table.data[level + 1] - offset_table.data[level]) / ((1<<N_POS_DIMS)-1);

    const float scale = grid_scale(level, 1.0f, base_resolution);
    const uint32_t resolution = grid_resolution(scale);
    const uint32_t wave_length = get_wave_length<WAVE_TYPE>();

	auto add_grid_gradient = [&](const uint32_t local_pos[N_POS_DIMS], const vector_t<T, N_FEATURES_PER_THREAD>& grad, const float weight, uint32_t idx_hl) {
		uint32_t index = wave_index<N_POS_DIMS, HASH_TYPE>(hashmap_size, resolution, local_pos, idx_hl) * N_FEATURES_PER_LEVEL + feature;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600 // atomicAdd(__half2) is only supported with compute capability 60 and above
		if (N_FEATURES_PER_THREAD > 1 && std::is_same<GRAD_T, __half>::value) {
			for (uint32_t f = 0; f < N_FEATURES_PER_THREAD; f += 2) {
				__half2 v = {(__half)((float)grad[f] * weight), (__half)((float)grad[f+1] * weight)};
				atomicAdd((__half2*)&grid_gradient[index + f], v);
			}
		} else
#endif
		{
			if (std::is_same<GRAD_T, __half>::value) {
				// Should never happen
				//printf("Attempted to use atomicAdd(__half)\n")
			} else {
				for (uint32_t f = 0; f < N_FEATURES_PER_THREAD; ++f) {
					atomicAdd((float*)&grid_gradient[index + f], (float)grad[f] * weight);
				}
			}
		}
	}; 

    auto phi = [&] (const float x) {
        float x_in = (float) x / wave_length * wave_data_len;
        int x_idx = int(x_in);
        if (interpolation_type==InterpolationType::Nearest) {
            return phi_data[x_idx];
        }
        float _x = x_in - x_idx;
        if (x_idx < 0 || x_idx >= wave_data_len-1) {
            return 0.f;
        }
        return (float)(phi_data[x_idx] * (1.0-_x) + phi_data[x_idx+1] * _x);
    };
    auto psi = [&] (const float x) {
        float x_in = (float) x / wave_length * wave_data_len;
        int x_idx = int(x_in);
        if (interpolation_type==InterpolationType::Nearest) {
            return phi_data[x_idx];
        }
        float _x = x_in - x_idx;
        if (x_idx < 0 || x_idx >= wave_data_len-1) {
            return 0.f;
        }
        return (float)(psi_data[x_idx] * (1.0-_x) + psi_data[x_idx+1] * _x);
    };

    float pos[N_POS_DIMS];
    uint32_t pos_grid[N_POS_DIMS];

    TCNN_PRAGMA_UNROLL
    for (uint32_t dim=0; dim < N_POS_DIMS; ++dim) {
        pos_fract(positions_in(dim, i), &pos[dim], &pos_grid[dim], scale, identity_fun);
    }

    vector_t<T,N_FEATURES_PER_THREAD> grad;

    TCNN_PRAGMA_UNROLL
    for (uint32_t f = 0; f < N_FEATURES_PER_THREAD; ++f) {
        grad[f] = dL_dy[i + (level * N_FEATURES_PER_LEVEL + feature + f) * num_elements];
    }

    if (level==0) {
        uint32_t pos_grid_local[N_POS_DIMS];
        for (uint32_t step=0; step< wave_length; step++) {
            float weight = 1.0f;
            float temp_x;
            TCNN_PRAGMA_UNROLL
            for (uint32_t dim=0; dim<N_POS_DIMS; ++dim) {
                temp_x = pos[dim] + step;
                weight *= phi(temp_x);
                pos_grid_local[dim] = pos_grid[dim] - step;
            }
            add_grid_gradient(pos_grid_local, grad, weight, 1);
        }
    } else {
        TCNN_PRAGMA_UNROLL
        for (uint32_t idx_hl = 1; idx_hl < (1 <<N_POS_DIMS); ++idx_hl) {
            uint32_t pos_grid_local[N_POS_DIMS];
            TCNN_PRAGMA_UNROLL
            for (uint32_t step=0; step<wave_length; step++) {
                float weight = 1.0f;
                float scaling;
                float temp_x;
                TCNN_PRAGMA_UNROLL
                for (uint32_t dim=0; dim<N_POS_DIMS; ++dim) {
                    if((idx_hl&(1<<dim)==0)) {
                        scaling = 0.5;
                        temp_x = pos[dim]*scaling + step;
                        weight *= phi(temp_x);
                        pos_grid_local[dim] = pos_grid[dim] - step/scaling;
                    } else {
                        scaling = 1.0f;
                        temp_x = pos[dim]*scaling + step;
                        weight *= psi(temp_x);
                        pos_grid_local[dim] = pos_grid[dim] - step/scaling;
                    }
                }
                add_grid_gradient(pos_grid_local, grad, weight, idx_hl);
            }
        }

    }

}

template <typename T>
class WaveletEncoding : public Encoding<T> {
public:
	virtual uint32_t n_pos_dims() const = 0;
	virtual uint32_t n_features_per_level() const = 0;

	virtual size_t level_n_params(uint32_t level) const = 0;
	virtual size_t level_params_offset(uint32_t level) const = 0;

	virtual const WaveletOffsetTable& wavelet_offset_table() const = 0;

	float max_level() const {
		return m_max_level;
	}

	void set_max_level(float value) {
		m_max_level = value;
	}

	float* max_level_gpu() const {
		return m_max_level_gpu;
	}

	void set_max_level_gpu(float* value) {
		m_max_level_gpu = value;
	}

protected:
    float m_min_level;
    float m_max_level= 1000.f;
    float* m_max_level_gpu = nullptr;
};

template <typename T,
          uint32_t N_POS_DIMS=3,
          uint32_t N_FEATURES_PER_LEVEL=2,
          WaveFamily WAVE_TYPE=WaveFamily::haar,
          HashType HASH_TYPE=HashType::CoherentPrime>
class WaveletEncodingTemplated : public WaveletEncoding<T> {
public:
#if TCNN_MIN_GPU_ARCH >= 62 || TCNN_MIN_GPU_ARCH == 60
	// The GPUs that we tested this on do not have an efficient 1D fp16
	// atomicAdd feature. Thus, we accumulate gradients at fp32 if we're
	// forced to use 1D atomicAdds. As soon as 2D or higher is possible,
	// we can make use the efficient atomicAdd(half2) function.
	using grad_t = std::conditional_t<N_FEATURES_PER_LEVEL == 1, float, T>;
#else
	// atomicAdd(__half2) is only supported with compute capability 60 and above.
	// Since atomicAdd(__half) is relatively slow / doesn't exist for low compute
	// capabilities, accumulate in fp32 instead.
	using grad_t = float;
#endif
    WaveletEncodingTemplated(
        uint32_t n_features,
        uint32_t log2_hashmap_size,
        uint32_t base_resolution,
        InterpolationType interpolation_type
    ):        
    m_family{WAVE_TYPE},
    m_n_features{n_features},
    m_interpolation_type{interpolation_type},
    m_base_resolution{base_resolution}

    {
        m_n_levels = div_round_up(m_n_features, N_FEATURES_PER_LEVEL);
        uint32_t offset = 0;
        if (m_n_levels > MAX_N_LEVELS) {
            throw std::runtime_error{fmt::format("WaveletEncoding: m_n_levels={} must be at most MAX_N_LEVELS={}", m_n_levels, MAX_N_LEVELS)};
        }

        m_offset_table.data[0] = 0;
        for (uint32_t i = 0 ; i < m_n_levels; ++i) {
            //Compute number of dense params for the given level
            //wavelet has such layout, taking 2D for example:
            // +---------+---------+
            // |   LL    |   LH    |
            // +---------+---------+
            // |   HL    |   HH    |
            // +---------+---------+
            const uint32_t resolution = base_resolution * (1 << i);

            uint32_t max_params = std::numeric_limits<uint32_t>::max()/2;
			uint32_t params_in_level = std::pow((float)resolution, N_POS_DIMS) > (float)max_params ? max_params : powi(resolution, N_POS_DIMS);
            params_in_level = std::min(params_in_level, (1u << log2_hashmap_size));

            m_offset_table.data[i] = offset;
            offset += params_in_level;

        }
        m_offset_table.data[m_n_levels] = offset;
        m_offset_table.size = m_n_levels+1;

        m_wavefun_level = log2f(base_resolution) + m_n_levels + 1;
        m_wavefun_level = min(m_wavefun_level, 10u);
        m_n_params = m_offset_table.data[m_n_levels] * N_FEATURES_PER_LEVEL;
        m_n_output_dims = m_n_features;
        if (n_features % N_FEATURES_PER_LEVEL != 0) {
            throw std::runtime_error{fmt::format("WaveletEncoding: n_features={} must be a multiple of N_FEATURES_PER_LEVEL", n_features, N_FEATURES_PER_LEVEL)};
        }
        float* phi_data;
        float* psi_data;
        calc_wavefun<WAVE_TYPE>(m_wavefun_level, true, phi_data, m_wave_data_len);
        calc_wavefun<WAVE_TYPE>(m_wavefun_level, false, psi_data, m_wave_data_len);

        m_phi_data = GPUMemory<float>(m_wave_data_len);
        m_phi_data.copy_from_host(phi_data);
        m_psi_data = GPUMemory<float>(m_wave_data_len);
        m_psi_data.copy_from_host(psi_data);
    }
    std::unique_ptr<Context> forward_impl(
        cudaStream_t stream,
        const GPUMatrixDynamic<float>& input,
        GPUMatrixDynamic<T>* output = nullptr,
        bool use_inference_params = false,
        bool prepare_input_gradients = false
    ) override {
        auto forward = std::make_unique<ForwardContext>();

        const uint32_t num_elements = input.n();
        if ((!output && !prepare_input_gradients) || padded_output_width() == 0 || num_elements == 0) {
            return forward;
        }

        SyncedMultiStream synced_streams {stream, m_n_to_pad > 0 ? 2u : 1u};
        if (output && m_n_to_pad > 0) {
            if(output-> layout() == AoS) {
                parallel_for_gpu_aos(synced_streams.get(1), num_elements, m_n_to_pad, [n_output_dims = m_n_output_dims, out=output->pitched_ptr()] __device__ (size_t elem, size_t dim) {
                    out(elem)[n_output_dims + dim] = 0;
                });
			} else {
				parallel_for_gpu(synced_streams.get(1), num_elements * m_n_to_pad, [out=output->data() + num_elements * m_n_output_dims] __device__ (size_t i) {
					out[i] = 0;
				});
            }
        }

		// Idea: each block only takes care of _one_ wavelet level (but may iterate over multiple input elements).
		// This way, only one level of the hashmap needs to fit into caches at a time (and it reused for consecutive
		// elements) until it is time to process the next level.

        static constexpr uint32_t N_THREADS_WAVELET = 512;
        const dim3 blocks_hashgrid = { div_round_up(num_elements, N_THREADS_WAVELET), m_n_levels, 1};
        T* encoded_positions_soa = output ? output->data() : nullptr;
        GPUMemoryArena::Allocation workspace;
        if (output && output->layout() == AoS) {
            workspace = allocate_workspace(synced_streams.get(0), num_elements * m_n_features * sizeof(T));
            encoded_positions_soa = (T*) workspace.data();
        }

        if (prepare_input_gradients) {
            forward->dy_dx = GPUMatrix<float, RM>{N_POS_DIMS * m_n_features, input.n(), synced_streams.get(0)};
        }

        kernel_wavelet<T, N_POS_DIMS, N_FEATURES_PER_LEVEL, WAVE_TYPE, HASH_TYPE><<<blocks_hashgrid, N_THREADS_WAVELET, 0, synced_streams.get(0)>>>(
            num_elements,
            m_n_features,
            m_offset_table,
            m_base_resolution,
            this->m_max_level,
            this->m_max_level_gpu,
            m_interpolation_type,
            this->inference_params(),
            forward->positions.data() ? forward->positions.view() : input.view(),
            encoded_positions_soa,
            forward->dy_dx.data(),
            this->phi_data(),
            this->psi_data(),
            m_wave_data_len
		);

        if (output && output->layout() == AoS) {
            // Transpose result (was stored row major due to coalescing)
            const dim3 threads_transpose = {m_n_levels * N_FEATURES_PER_LEVEL, 8, 1};
            const uint32_t blocks_transpose = div_round_up(num_elements, threads_transpose.y);
            transpose_encoded_position<T><<<blocks_transpose, threads_transpose, 0, synced_streams.get(0)>>> (
                num_elements,
                encoded_positions_soa,
                output->pitched_ptr()
            );
        }

        return forward;
    }

    void backward_impl (
        cudaStream_t stream,
        const Context& ctx,
        const GPUMatrixDynamic<float>& input,
        const GPUMatrixDynamic<T>& output,
        const GPUMatrixDynamic<T>& dL_doutput,
        GPUMatrixDynamic<float>* dL_dinput=nullptr,
        bool use_infernce_params = false,
        EGradientMode param_gradients_mode = EGradientMode::Overwrite

    ) override {
        const uint32_t num_elements = input.n();
        if((!dL_dinput && param_gradients_mode==EGradientMode::Ignore) || padded_output_width()==0 || num_elements==0) {
            return;
        }
        const auto& forward = dynamic_cast<const ForwardContext&>(ctx);
        const T* dL_dy_rm = dL_doutput.data();
        GPUMemoryArena::Allocation workspace;
        if (dL_doutput.layout() == CM) {
			workspace = allocate_workspace(stream, num_elements * m_n_features * sizeof(T));

			// Transpose dL_dy. Use the buffer previously occupied by the encoded positions
			const dim3 threads_transpose = { m_n_levels * N_FEATURES_PER_LEVEL, 8, 1 };
			const uint32_t blocks_transpose = div_round_up(num_elements, threads_transpose.y);
			transpose_gradients<T><<<blocks_transpose, threads_transpose, 0, stream>>>(
				num_elements,
				(T*)workspace.data(),
				dL_doutput.pitched_ptr()
			);

			dL_dy_rm = (const T*)workspace.data();
        }

        if (param_gradients_mode != EGradientMode::Ignore) {
            grad_t* grid_gradient;
            GPUMemoryArena::Allocation grid_gradient_tmp;
            if(!std::is_same<grad_t, T>::value) {
                grid_gradient_tmp = allocate_workspace(stream, m_n_params*sizeof(grad_t));
                grid_gradient = (grad_t*) grid_gradient_tmp.data();
            } else {
                grid_gradient = (grad_t*) this->gradients();
            }
			if (param_gradients_mode == EGradientMode::Overwrite) {
				CUDA_CHECK_THROW(cudaMemsetAsync(grid_gradient, 0, n_params() * sizeof(grad_t), stream));
			}
			static constexpr uint32_t N_THREADS_WAVELET = 256;
			static constexpr uint32_t N_FEATURES_PER_THREAD = std::min(2u, N_FEATURES_PER_LEVEL);

			const dim3 blocks_hashgrid = { div_round_up(num_elements * N_FEATURES_PER_LEVEL / N_FEATURES_PER_THREAD, N_THREADS_WAVELET), m_n_levels, 1 };
			kernel_wavelet_backward<T, grad_t, N_POS_DIMS, N_FEATURES_PER_LEVEL, N_FEATURES_PER_THREAD, WAVE_TYPE, HASH_TYPE><<<blocks_hashgrid, N_THREADS_WAVELET, 0, stream>>>(
				num_elements,
				m_n_features,
				m_offset_table,
				m_base_resolution,
				this->m_max_level,
				this->m_max_level_gpu,
				m_interpolation_type,
				grid_gradient,
				forward.positions.data() ? forward.positions.view() : input.view(), // positions SoA
				dL_dy_rm, // gradients SoA
                this->phi_data(),
                this->psi_data(),
                m_wave_data_len
			);
            if (!std::is_same<grad_t, T>::value) {
				parallel_for_gpu(stream, n_params(), [grad=this->gradients(), grad_tmp=grid_gradient] __device__ (size_t i) {
					grad[i] = (T)grad_tmp[i];
				});
            }
        }
    }
    float* phi_data() {
        return m_phi_data.data();
    }
    float* psi_data() {
        return m_psi_data.data();
    }
    uint32_t input_width() const override {
        return N_POS_DIMS;
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
		return N_FEATURES_PER_LEVEL;
	}

	MatrixLayout preferred_output_layout() const override {
		return SoA;
	}
    void set_params_impl(T* params, T* inference_params, T* gradients) override { }
    void initialize_params(pcg32& rnd, float* params_full_precision, float scale = 1) override {
		generate_random_uniform<float>(rnd, n_params(), params_full_precision, -1e-4f * scale, 1e-4f * scale);
    }
    size_t n_params() const override {
        return m_n_params;
    }
	size_t level_n_params(uint32_t level) const override {
		return level_params_offset(level + 1) - level_params_offset(level);
	}
    size_t level_params_offset(uint32_t level) const override {
        if (level >= m_offset_table.size) {
            throw std::runtime_error{"Out of bounds params offset request."};
        }
        return m_offset_table.data[level];
    }
    const WaveletOffsetTable& wavelet_offset_table() const override {
        return m_offset_table;
    }
    std::vector<std::pair<uint32_t, uint32_t>> layer_sizes() const override {
        return {};
    }
    uint32_t n_pos_dims() const override {
        return N_POS_DIMS;
    }
    uint32_t n_features_per_level() const override {
        return N_FEATURES_PER_LEVEL;
    }
    json hyperparams() const override {
        json result = {
            {"otype", "Wavelet"},
            {"family", to_string(m_family)},
            {"n_levels", m_n_levels},
            {"n_features_per_level", N_FEATURES_PER_LEVEL},
            {"base_resolution", m_base_resolution}, 
            {"interpolation", to_string(m_interpolation_type)},
            {"hash", to_string(HASH_TYPE)},
        };
        return result;
    }


private:
    WaveFamily m_family;
    uint32_t m_wavefun_level;
    GPUMemory<float> m_phi_data; 
    GPUMemory<float> m_psi_data;
    uint32_t m_wave_data_len;
    struct ForwardContext : public Context {
        GPUMatrix<float, RM> positions;
        GPUMatrix<float, RM> dy_dx;
    };

    uint32_t m_n_features;
    uint32_t m_n_levels;
	uint32_t m_n_params;
	WaveletOffsetTable m_offset_table;
	uint32_t m_log2_hashmap_size;
	uint32_t m_base_resolution;


	// derived sizes
	uint32_t m_n_output_dims;
	uint32_t m_n_to_pad = 0;


	InterpolationType m_interpolation_type;
};

template <typename T, uint32_t N_FEATURES_PER_LEVEL, WaveFamily WAVE_TYPE, HashType HASH_TYPE> 
WaveletEncoding<T>* create_wavelet_encoding_templated_2(uint32_t n_dims_to_encode, const json& encoding) {
    const uint32_t log2_hashmap_size = encoding.value("log2_hashmap_size", 19u);
    const std::string encoding_type = encoding.value("otype", "Wavelet");
    uint32_t n_features;
    n_features = N_FEATURES_PER_LEVEL * encoding.value("n_levels", 16u);

#define TCNN_WAVELET_PARAMS \
    n_features, \
    log2_hashmap_size, \
    encoding.value("base_resolution", 16u), \
    string_to_interpolation_type(encoding.value("interpolation", "Linear")), \

    switch (n_dims_to_encode) {
        case 2: return new WaveletEncodingTemplated<T, 2, N_FEATURES_PER_LEVEL, WAVE_TYPE, HASH_TYPE>{TCNN_WAVELET_PARAMS};
        case 3: return new WaveletEncodingTemplated<T, 3, N_FEATURES_PER_LEVEL, WAVE_TYPE, HASH_TYPE>{TCNN_WAVELET_PARAMS};
        case 4: return new WaveletEncodingTemplated<T, 4, N_FEATURES_PER_LEVEL, WAVE_TYPE, HASH_TYPE>{TCNN_WAVELET_PARAMS};
		default: throw std::runtime_error{"WaveletEncoding: number of input dims must be 2 ~ 4."};
    }
#undef TCNN_GRID_PARAMS
}

template <typename T, WaveFamily WAVE_TYPE, HashType HASH_TYPE>
WaveletEncoding<T>* create_wavelet_encoding_templated_1(uint32_t n_dims_to_encode, const json& encoding) {
	const uint32_t n_features_per_level = encoding.value("n_features_per_level", 2u);
	switch (n_features_per_level) {
		case 1: return create_wavelet_encoding_templated_2<T, 1, WAVE_TYPE, HASH_TYPE>(n_dims_to_encode, encoding);
		case 2: return create_wavelet_encoding_templated_2<T, 2, WAVE_TYPE, HASH_TYPE>(n_dims_to_encode, encoding);
		case 4: return create_wavelet_encoding_templated_2<T, 4, WAVE_TYPE, HASH_TYPE>(n_dims_to_encode, encoding);
		case 8: return create_wavelet_encoding_templated_2<T, 8, WAVE_TYPE, HASH_TYPE>(n_dims_to_encode, encoding);
		default: throw std::runtime_error{"WaveletEncoding: n_features_per_level must be 1, 2, 4, or 8."};
	}
}

template <typename T, HashType HASH_TYPE>
WaveletEncoding<T>* create_wavelet_encoding_templated_0(uint32_t n_dims_to_encode, const json& encoding) {
    const WaveFamily wave_type = string_to_wavelet(encoding.value("type", "db2"));
	switch (wave_type) {
		case WaveFamily::haar: return create_wavelet_encoding_templated_1<T, WaveFamily::haar, HASH_TYPE>(n_dims_to_encode, encoding);
		case WaveFamily::db2: return create_wavelet_encoding_templated_1<T, WaveFamily::db2, HASH_TYPE>(n_dims_to_encode, encoding);
		case WaveFamily::db4: return create_wavelet_encoding_templated_1<T, WaveFamily::db4, HASH_TYPE>(n_dims_to_encode, encoding);
		default: throw std::runtime_error{"WaveletEncoding: WaveFamily must be haar, db2 or db4."};
	}
}

template <typename T>
WaveletEncoding<T>* create_wavelet_encoding(uint32_t n_dims_to_encode, const json& encoding) {
	const HashType hash_type = string_to_hash_type(encoding.value("hash", "CoherentPrime"));
	switch (hash_type) {
		case HashType::Prime: return create_wavelet_encoding_templated_0<T, HashType::Prime>(n_dims_to_encode, encoding);
		case HashType::CoherentPrime: return create_wavelet_encoding_templated_0<T, HashType::CoherentPrime>(n_dims_to_encode, encoding);
		case HashType::ReversedPrime: return create_wavelet_encoding_templated_0<T, HashType::ReversedPrime>(n_dims_to_encode, encoding);
		case HashType::Rng: return create_wavelet_encoding_templated_0<T, HashType::Rng>(n_dims_to_encode, encoding);
		default: throw std::runtime_error{"WaveletEncoding: invalid hash type."};
	}
}


TCNN_NAMESPACE_END