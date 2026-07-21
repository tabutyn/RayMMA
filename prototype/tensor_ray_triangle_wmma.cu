// SPDX-License-Identifier: MIT
//
// Minimal WMMA prototype for separated ray/triangle intersection algebra.
//
// One warp evaluates four triangles against sixteen rays.  The A operand has
// four rows per triangle (Nt, Nu, Nv, Delta).  The B operand has one column per
// ray.  The inner dimension contains ten useful features and six zero pads:
//
//   ray feature = [ 1 | O.xyz | r.xyz | cross(O,r).xyz | zero padding ]
//
// See ALGEBRA.md for the complete derivation.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

using namespace nvcuda;

namespace {

constexpr int kM = 16;
constexpr int kN = 16;
constexpr int kK = 16;
constexpr int kTriangleCount = 4;
constexpr int kRayCount = 16;
constexpr int kChannelsPerTriangle = 4;

enum Feature : int {
    kOne = 0,
    kOx = 1,
    kOy = 2,
    kOz = 3,
    kRx = 4,
    kRy = 5,
    kRz = 6,
    kMx = 7,
    kMy = 8,
    kMz = 9,
};

enum Channel : int {
    kNt = 0,
    kNu = 1,
    kNv = 2,
    kDelta = 3,
};

struct Vec3 {
    float x;
    float y;
    float z;
};

struct Triangle {
    Vec3 A;
    Vec3 C;  // edge multiplied by barycentric u
    Vec3 B;  // edge multiplied by barycentric v
};

struct Ray {
    Vec3 O;
    Vec3 r;
};

Vec3 operator-(const Vec3& a, const Vec3& b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

Vec3 cross(const Vec3& a, const Vec3& b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    };
}

float dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

void set3(std::array<float, kM * kK>& matrix,
          int row,
          int first_column,
          const Vec3& value) {
    matrix[row * kK + first_column + 0] = value.x;
    matrix[row * kK + first_column + 1] = value.y;
    matrix[row * kK + first_column + 2] = value.z;
}

// A is stored row-major because the WMMA matrix_a fragment below is row-major.
std::array<float, kM * kK> packTriangleCoefficients(
    const std::array<Triangle, kTriangleCount>& triangles) {
    std::array<float, kM * kK> packed{};

    for (int triangle_index = 0; triangle_index < kTriangleCount;
         ++triangle_index) {
        const Triangle& triangle = triangles[triangle_index];

        // All of these quantities depend only on the triangle and can be
        // cached with the BVH leaf or triangle record.
        const Vec3 n = cross(triangle.C, triangle.B);
        const float an = dot(triangle.A, n);
        const Vec3 pu = cross(triangle.B, triangle.A);
        const Vec3 pv = cross(triangle.A, triangle.C);

        const int row_nt = triangle_index * kChannelsPerTriangle + kNt;
        const int row_nu = triangle_index * kChannelsPerTriangle + kNu;
        const int row_nv = triangle_index * kChannelsPerTriangle + kNv;
        const int row_delta = triangle_index * kChannelsPerTriangle + kDelta;

        // Nt = dot([an,-n,0,0], [1,O,r,m]).
        packed[row_nt * kK + kOne] = an;
        set3(packed, row_nt, kOx, {-n.x, -n.y, -n.z});

        // Nu = dot([0,0,pu,-B], [1,O,r,m]).
        set3(packed, row_nu, kRx, pu);
        set3(packed,
             row_nu,
             kMx,
             {-triangle.B.x, -triangle.B.y, -triangle.B.z});

        // Nv = dot([0,0,pv,C], [1,O,r,m]).
        set3(packed, row_nv, kRx, pv);
        set3(packed, row_nv, kMx, triangle.C);

        // Delta = dot([0,0,n,0], [1,O,r,m]).
        set3(packed, row_delta, kRx, n);
    }

    return packed;
}

// B is logically a K-by-N matrix and is stored column-major.  Consequently,
// every ray occupies one contiguous 16-element column.
std::array<float, kK * kN> packRayFeatures(
    const std::array<Ray, kRayCount>& rays) {
    std::array<float, kK * kN> packed{};

    for (int ray_index = 0; ray_index < kRayCount; ++ray_index) {
        const Ray& ray = rays[ray_index];
        const Vec3 moment = cross(ray.O, ray.r);
        float* column = packed.data() + ray_index * kK;

        column[kOne] = 1.0f;
        column[kOx] = ray.O.x;
        column[kOy] = ray.O.y;
        column[kOz] = ray.O.z;
        column[kRx] = ray.r.x;
        column[kRy] = ray.r.y;
        column[kRz] = ray.r.z;
        column[kMx] = moment.x;
        column[kMy] = moment.y;
        column[kMz] = moment.z;
        // Features 10 through 15 remain zero.
    }

    return packed;
}

std::vector<half> convertToHalf(const float* source, std::size_t count) {
    std::vector<half> result(count);
    for (std::size_t i = 0; i < count; ++i) {
        result[i] = __float2half_rn(source[i]);
    }
    return result;
}

// This is the complete Tensor Core computation.  It must be launched with one
// full warp.  cudaMalloc supplies the alignment required by load_matrix_sync.
__global__ void intersectTileWmma(const half* triangle_coefficients,
                                  const half* ray_features,
                                  float* output) {
    wmma::fragment<wmma::matrix_a,
                   kM,
                   kN,
                   kK,
                   half,
                   wmma::row_major>
        triangle_fragment;
    wmma::fragment<wmma::matrix_b,
                   kM,
                   kN,
                   kK,
                   half,
                   wmma::col_major>
        ray_fragment;
    wmma::fragment<wmma::accumulator, kM, kN, kK, float>
        output_fragment;

    wmma::fill_fragment(output_fragment, 0.0f);
    wmma::load_matrix_sync(triangle_fragment, triangle_coefficients, kK);
    wmma::load_matrix_sync(ray_fragment, ray_features, kK);
    wmma::mma_sync(output_fragment,
                   triangle_fragment,
                   ray_fragment,
                   output_fragment);
    wmma::store_matrix_sync(
        output, output_fragment, kN, wmma::mem_row_major);
}

#define CUDA_CHECK(expression)                                                \
    do {                                                                      \
        const cudaError_t status = (expression);                              \
        if (status != cudaSuccess) {                                          \
            std::cerr << "CUDA failure at " << __FILE__ << ':' << __LINE__   \
                      << ": " << cudaGetErrorString(status) << '\n';          \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (false)

// Independent Gaussian-elimination reference for
// [r,-C,-B] * [t,u,v]^T = A-O.  This intentionally does not reuse the
// separated formula being tested.
bool solveDirect(const Triangle& triangle,
                 const Ray& ray,
                 std::array<double, 3>* solution) {
    const Vec3 q = triangle.A - ray.O;
    double matrix[3][4] = {
        {ray.r.x, -triangle.C.x, -triangle.B.x, q.x},
        {ray.r.y, -triangle.C.y, -triangle.B.y, q.y},
        {ray.r.z, -triangle.C.z, -triangle.B.z, q.z},
    };

    for (int column = 0; column < 3; ++column) {
        int pivot = column;
        for (int row = column + 1; row < 3; ++row) {
            if (std::abs(matrix[row][column]) >
                std::abs(matrix[pivot][column])) {
                pivot = row;
            }
        }
        if (std::abs(matrix[pivot][column]) < 1.0e-12) {
            return false;
        }
        if (pivot != column) {
            for (int j = column; j < 4; ++j) {
                std::swap(matrix[pivot][j], matrix[column][j]);
            }
        }

        const double divisor = matrix[column][column];
        for (int j = column; j < 4; ++j) {
            matrix[column][j] /= divisor;
        }
        for (int row = 0; row < 3; ++row) {
            if (row == column) {
                continue;
            }
            const double factor = matrix[row][column];
            for (int j = column; j < 4; ++j) {
                matrix[row][j] -= factor * matrix[column][j];
            }
        }
    }

    *solution = {matrix[0][3], matrix[1][3], matrix[2][3]};
    return true;
}

bool nearlyEqual(float actual,
                 float expected,
                 float absolute_tolerance,
                 float relative_tolerance) {
    return std::abs(actual - expected) <=
           absolute_tolerance + relative_tolerance * std::abs(expected);
}

bool hitFromNumerators(float nt,
                       float nu,
                       float nv,
                       float delta,
                       float t_min,
                       float t_max) {
    if (std::abs(delta) <= 1.0e-5f) {
        return false;
    }
    const float sign = delta < 0.0f ? -1.0f : 1.0f;
    const float d = sign * delta;
    const float T = sign * nt;
    const float U = sign * nu;
    const float V = sign * nv;
    const float tolerance = 2.0e-3f * d;

    return U >= -tolerance && V >= -tolerance &&
           U + V <= d + tolerance && T >= t_min * d && T <= t_max * d;
}

std::array<Triangle, kTriangleCount> makeTriangles() {
    return {{
        {{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}},
        {{-0.4f, -0.2f, 0.8f}, {1.2f, 0.1f, 0.2f}, {0.1f, 1.1f, -0.1f}},
        {{0.3f, -0.5f, 1.6f}, {0.8f, 0.2f, 0.5f}, {-0.2f, 1.0f, 0.25f}},
        {{-0.7f, 0.3f, 2.3f}, {1.1f, -0.1f, -0.3f}, {0.2f, 0.9f, 0.4f}},
    }};
}

std::array<Ray, kRayCount> makeRays() {
    std::array<Ray, kRayCount> rays{};
    for (int i = 0; i < kRayCount; ++i) {
        const int column = i % 4;
        const int row = i / 4;
        rays[i].O = {
            -0.35f + 0.32f * static_cast<float>(column),
            -0.25f + 0.30f * static_cast<float>(row),
            -1.5f + 0.03f * static_cast<float>(i),
        };
        rays[i].r = {
            0.025f * static_cast<float>(column - 1),
            -0.02f * static_cast<float>(row - 1),
            1.0f + 0.015f * static_cast<float>((i % 3) - 1),
        };
    }
    return rays;
}

}  // namespace

int main() {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        std::cerr << "No CUDA device was found.\n";
        return EXIT_FAILURE;
    }

    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
    if (properties.major < 7) {
        std::cerr << "WMMA with FP16 inputs requires compute capability 7.0 "
                     "or newer; found "
                  << properties.major << '.' << properties.minor << ".\n";
        return EXIT_FAILURE;
    }

    const auto triangles = makeTriangles();
    const auto rays = makeRays();
    const auto triangle_coefficients_f = packTriangleCoefficients(triangles);
    const auto ray_features_f = packRayFeatures(rays);
    const auto triangle_coefficients_h =
        convertToHalf(triangle_coefficients_f.data(),
                      triangle_coefficients_f.size());
    const auto ray_features_h =
        convertToHalf(ray_features_f.data(), ray_features_f.size());

    half* device_triangle_coefficients = nullptr;
    half* device_ray_features = nullptr;
    float* device_output = nullptr;
    CUDA_CHECK(cudaMalloc(&device_triangle_coefficients,
                          triangle_coefficients_h.size() * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&device_ray_features,
                          ray_features_h.size() * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&device_output, kM * kN * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(device_triangle_coefficients,
                          triangle_coefficients_h.data(),
                          triangle_coefficients_h.size() * sizeof(half),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_ray_features,
                          ray_features_h.data(),
                          ray_features_h.size() * sizeof(half),
                          cudaMemcpyHostToDevice));

    // One block containing exactly one warp computes the whole demonstration
    // tile. A production kernel would assign one warp to every candidate tile.
    intersectTileWmma<<<1, 32>>>(device_triangle_coefficients,
                                 device_ray_features,
                                 device_output);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::array<float, kM * kN> gpu_output{};
    CUDA_CHECK(cudaMemcpy(gpu_output.data(),
                          device_output,
                          gpu_output.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(device_triangle_coefficients));
    CUDA_CHECK(cudaFree(device_ray_features));
    CUDA_CHECK(cudaFree(device_output));

    int failures = 0;
    float maximum_raw_error = 0.0f;
    double maximum_ratio_error = 0.0;
    int reported_hits = 0;

    for (int triangle_index = 0; triangle_index < kTriangleCount;
         ++triangle_index) {
        for (int ray_index = 0; ray_index < kRayCount; ++ray_index) {
            float values[kChannelsPerTriangle]{};

            for (int channel = 0; channel < kChannelsPerTriangle; ++channel) {
                const int output_row =
                    triangle_index * kChannelsPerTriangle + channel;
                float expected = 0.0f;
                for (int feature = 0; feature < kK; ++feature) {
                    expected +=
                        triangle_coefficients_f[output_row * kK + feature] *
                        ray_features_f[ray_index * kK + feature];
                }

                const float actual =
                    gpu_output[output_row * kN + ray_index];
                values[channel] = actual;
                maximum_raw_error =
                    std::max(maximum_raw_error, std::abs(actual - expected));

                // This tolerance covers conversion of both operands to FP16.
                if (!nearlyEqual(actual, expected, 2.0e-2f, 2.0e-2f)) {
                    ++failures;
                    std::cerr << "Raw mismatch: triangle=" << triangle_index
                              << " ray=" << ray_index
                              << " channel=" << channel
                              << " expected=" << expected
                              << " actual=" << actual << '\n';
                }
            }

            std::array<double, 3> direct{};
            if (solveDirect(triangles[triangle_index], rays[ray_index], &direct)
                && std::abs(values[kDelta]) > 0.25f) {
                const std::array<double, 3> tensor_solution = {
                    values[kNt] / values[kDelta],
                    values[kNu] / values[kDelta],
                    values[kNv] / values[kDelta],
                };
                for (int component = 0; component < 3; ++component) {
                    const double error =
                        std::abs(tensor_solution[component] - direct[component]);
                    maximum_ratio_error = std::max(maximum_ratio_error, error);
                    const double tolerance =
                        5.0e-2 + 3.0e-2 * std::abs(direct[component]);
                    if (error > tolerance) {
                        ++failures;
                        std::cerr << "Ratio mismatch: triangle="
                                  << triangle_index << " ray=" << ray_index
                                  << " component=" << component
                                  << " expected=" << direct[component]
                                  << " actual=" << tensor_solution[component]
                                  << '\n';
                    }
                }
            }

            if (hitFromNumerators(values[kNt],
                                  values[kNu],
                                  values[kNv],
                                  values[kDelta],
                                  0.0f,
                                  1000.0f) &&
                reported_hits < 8) {
                std::cout << "hit triangle=" << triangle_index
                          << " ray=" << ray_index
                          << " t=" << values[kNt] / values[kDelta]
                          << " u=" << values[kNu] / values[kDelta]
                          << " v=" << values[kNv] / values[kDelta] << '\n';
                ++reported_hits;
            }
        }
    }

    std::cout << std::setprecision(7)
              << "GPU: " << properties.name << '\n'
              << "maximum raw-output error: " << maximum_raw_error << '\n'
              << "maximum recovered-coordinate error: "
              << maximum_ratio_error << '\n';

    if (failures != 0) {
        std::cerr << failures << " verification checks failed.\n";
        return EXIT_FAILURE;
    }

    std::cout << "All WMMA checks passed.\n";
    return EXIT_SUCCESS;
}
