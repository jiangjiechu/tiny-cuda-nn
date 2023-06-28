#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/multi_stream.h>
#include <tiny-cuda-nn/random.h>

#include <stdexcept>
#include <vector>
#include <string>
#include <stdint.h>

TCNN_NAMESPACE_BEGIN

static constexpr uint32_t MY_MAX_N_LEVELS = 128;

struct GridOffsetTable {
    uint32_t data[MY_MAX_N_LEVELS+1] = {};
    uint32_t size = 0;
};

enum class GridType {
    Hash,
    Dense,
    Tiled,
};

inline GridType mystring_to_grid_type(const std::string& grid_type) {
    if (equals_case_insensitive(grid_type, "Hash")) {
        return GridType::Hash;
    }else if (equals_case_insensitive(grid_type, "Dense")) {
        return GridType::Dense;
    }else if (equals_case_insensitive(grid_type, "Tiled")) {
        return GridType::Tiled;
    }

    throw std::runtime_error{fmt::format("Invalid grid type: {}", grid_type)};
}

inline std::string myto_string(GridType grid_type) {
    switch (grid_type) {
        case GridType::Hash: return "Hash";
        case GridType::Dense: return "Dense";
        case GridType::Tiled: return "Tiled";
        default: throw std::runtime_error{"Invalid grid type."};
    }
}
template <uint32_t N_DIMS, uint32_t N_PRIMES>
__device__ uint32_t lcg_hash(const uint32_t pos_grid[N_DIMS], const uint32_t primes[N_PRIMES]) {
    static_assert(NDIMS <= N_PRIMES, "lcg_hash can only hash up to N_PRIMES dimensions.");
    uint32_t result = 0;
    TCNN_PRAGMA_UNROLL
    for (uint32_t i = 0; i < N_DIMS; i++) {
        result ^= pos_grid[i] * primes[i];
    }
    return result;
}

template <uint32_t N_DIMS>
__device__ uint32_t prime_hash(const uint32_t pos_grid[N_DIMS]) {
	constexpr uint32_t factors[7] = { 1958374283u, 2654435761u, 805459861u, 3674653429u, 2097192037u, 1434869437u, 2165219737u };
    return lcg_hash<N_DIMS, 7>(pos_grid, factors);
}

template <uint32_t N_DIMS>
__device__ uint32_t coherent_prime_hash(const uint32_t pos_grid[N_DIMS]) {
	constexpr uint32_t factors[7] = { 1u, 2654435761u, 805459861u, 3674653429u, 2097192037u, 1434869437u, 2165219737u };
    return lcg_hash<N_DIMS, 7> (pos_grid,factors);
}

template <uint32_t N_DIMS>
__device__ uint32_t reversed_prime_hash(const uint32_t pos_grid[N_DIMS]) {
	constexpr uint32_t factors[7] = { 2165219737u, 1434869437u, 2097192037u, 3674653429u, 805459861u, 2654435761u, 1958374283u };
	return lcg_hash<N_DIMS, 7>(pos_grid, factors);
}

template <uint32_t N_DIMS>
__device__ uint32_t rng_hash(const uint32_t pos_grid[N_DIMS], const uint32_t seed = 1337) {
    constexpr uint32_t N_BITS_PER_DIM = 64/ NDIMS;
    uint64_t step = 0;
    TCNN_PRAGMA_UNROLL
    for (uint32_t i = 0; i< N_DIMS; i++) { 
        step ^= (uint64_t) pos_grid[i] << (i * N_BITS_PER_DIM);
    }

    ::pcg32 rng{seed};
    rng.advance((int64_t)step);
    return rng.next_uint();
}

template <uint32_t N_DIMS, HashType HASH_TYPE>
__device__ uint32_t grid_hash(const uint32_t pos_grid[N_DIMS]) {
    switch (HASH_TYPE) {
        case HashType::Prime : return prime_hash<N_DIMS>(pos_grid);
        case HashType::CoherentPrime : return coherent_prime_hash<N_DIMS>(pos_grid);
        case HashType::ReversedPrime : return reversed_prime_hash<N_DIMS>(pos_grid);
        case HashType::Rng : return rng_hash<N_DIMS>(pos_grid);
    }

    return 0;
}

__device__ inline float random_val(uint32_t seed, uint32_t idx) {
    pcg32 rng{seed};
    rng.advance(idx);
    return rng.next_float();
}

TCNN_HOST_DEVICE inline float grid_scale (uint32_t level, float log2_per_level_scale, uint32_t base_resolution) {
    // The -1 means that `base_resolution` refers to the number of grid _vertices_ rather
    // than the number of cells. This is slightly different from the notation in the paper,
    // but results in nice, power-of-2-scaled parameter grids that fit better into cache lines.
    return exp2f(level * log2_per_level_scale) * base_resolution - 1.0f;
}

TCNN_HOST_DEVICE inline uint32_t grid_resolution(float scale) {
    return (uint32_t)ceilf(scale) + 1;
}

template <typename T>
__global__ void extract_position(
    const uint32_t num_elements, 
    PitchedPtr<const float> data_in,
    T* __restrict__ output
) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= num_elements) return;
    const uint32_t dim_idx = threadIdx.y;
    output[i+ dim_idx * num_elements] = (T)data_in(i)[dim_idx];
}

template <typename T, uint32_t N_POS_DIMS, uint32_t N_FEATURES_PER_LEVEL, HashType HASH_TYPE>
__global__ void kernel_grid(
    const uint32_t num_elements,
    const uint32_t num_grid_features,
    const GridOffsetTable offset_table,
    const uint32_t base_resolution,
    const float log2_perlevel_scale,
    float max_level,
    const float* __restrict__ max_level_gpu,
    const InterpolationType interpolation_type,
    const GridType grid_type,
    const T* __restrict__ grid, 
    MatrixView <const float> positions_in,
    T* __restrict__ encoded_postions,
    float* __restrict__ dy_dx
) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= num_elements) return;
    const uint32_t level = blockIdx.y;
    if (max_level_gpu) {
        max_level = (max_level_gpu[i] * num_grid_features) / N_FEATURES_PER_LEVEL;
    } else {
        max_level = (max_level * num_grid_features) / N_FEATURES_PER_LEVEL;
    }

    if (level >= max_level + 1e-3f) {
        if (encoded_postions) {
            #pragma unroll
            for (uint32_t f = 0; f < N_FEATURES_PER_LEVEL; ++f) {
                encoded_postions[i + (level * N_FEATURES_PER_LEVEL + f) * num_elements] = (T) 0.0f;
            }
        }

        // Gradient is zero for zeroed-out dimensions
        if (dy_dx) {
            #pragma unroll
            for (uint32_t f = 0; f < N_FEATURES_PER_LEVEL; ++f) { 
                ((vector_fullp_t<N_POS_DIMS>*)dy_dx)[i + (level * N_FEATURES_PER_LEVEL + f) * num_elements] = {0.0f};
            }
        }
        return;
    }

    grid += offset_table.data[level] * N_FEATURES_PER_LEVEL;
    const uint32_t hashmap_size = offset_table.data[level + 1] - offset_table.data[level];
    const float scale = grid_scale(level, log2_perlevel_scale, base_resolution);
    const uint32_t resolution = grid_resolution(scale);

    float pos[N_POS_DIMS];
    float pos_derivation[N_POS_DIMS];
    uint32_t pos_grid[N_POS_DIMS];

    if (interpolation_type == InterpolationType::Nearest || interpolation_type == InterpolationType::Linear) {
        #pragma unroll
        for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
            pos_fract(positions_in(dim,i), &pos[dim], &pos_derivation[dim], &pos_grid[dim], scale, identity_derivative);
        }
    } else {
        #pragma unroll
        for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
            pos_fract(positions_in(dim,i), &pos[dim], &pos_derivation[dim], &pos_grid[dim], sclae, smoothstep);
        }
    }

    auto grid_val = [&](const uint32_t local_pos[N_POS_DIMS]) {
        const uint32_t index = grid_index<N_POS_DIMS, HASH_TYPE> (grid_type, hashmap_size, resolution, local_pos) * N_FEATURES_PER_LEVEL;
        return *(vector_t<T, N_FEATURES_PER_LEVEL>*)&grid[index];
    };

    if (interpolation_type == InterpolationType::Nearest) {
        auto result = grid_val(pos_grid);
        if (encoded_postions) {
            #pragma unroll;
            for (uint32_t f = 0; f < N_FEATURES_PER_LEVEL; ++f) {
                encoded_postions[i + (level * N_FEATURES_PER_LEVEL + f) * num_elements] = result[f];
            }
        }

        // Gradient is zerof when there's no interpolation
        if (dy_dx) {
            #pragma unroll;
            for (uint32_t f = 0; f < N_FEATURES_PER_LEVEL; ++f) {
                ((vector_fullp_t<N_POS_DIMS>*)dy_dx)[i + (level * N_FEATURES_PER_LEVEL + f) * num_elements] = {0.0f}
            }
        }

        return;
    }

    if (encoded_postions) {
        // N-linear interpolation
        vector_t<T, N_FEATURES_PER_LEVEL> result = {};

        #pragma unroll
        for (uint32_t idx = 0; idx < (1 << N_POS_DIMS); ++idx) {
            float weight = 1;
            uint32_t pos_grid_local[N_POS_DIMS];


            #pragma unroll
            for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
                if ((idx & (1 << dim)) == 0) {
                    weight *= 1 - pos[dim];
                    pos_grid_local[dim] = pos_grid[dim];
                } else {
                    weight *= pos[dim];
                    pos_grid_local[dim] = pos_grid[dim] + 1;
                }
            }
            result = fma((T)weight, grid_val(pos_grid_local), result);
        }
        #pragma unroll
        for (uint32_t f = 0; f < N_FEATURES_PER_LEVEL; ++f) {
            encoded_postions[i + (level * N_FEATURES_PER_LEVEL + f) * num_elements] = result[f];
        }
    }

    // Gradient


}


TCNN_NAMESPACE_END

