# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.24)

# Set this before CUDA compiler detection so the default build targets the GPU
# in the current machine. Override it for cross-compilation or CI.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(raymma VERSION 0.1.0 LANGUAGES CXX CUDA)

include(CTest)

option(RAYMMA_BUILD_BENCHMARK
    "Build the standalone wide-BVH research benchmark" ON)
option(RAYMMA_ENABLE_EXTERNAL_SCENES
    "Enable the CC0 open model and separately licensed benchmark scenes" OFF)
option(RAYMMA_ENABLE_TINYBVH
    "Enable the optional TinyBVH SAH/SBVH production builders" OFF)
option(RAYMMA_BUILD_VIEWER
    "Build the live CUDA/OpenGL viewer and benchmark image viewer" OFF)

find_package(CUDAToolkit REQUIRED)

if(RAYMMA_ENABLE_TINYBVH)
    set(RAYMMA_TINYBVH_ROOT "" CACHE PATH
        "Path to a TinyBVH checkout containing tiny_bvh.h")
    find_path(RAYMMA_TINYBVH_INCLUDE_DIR tiny_bvh.h
        HINTS "${RAYMMA_TINYBVH_ROOT}"
        REQUIRED)
endif()

set(RAYMMA_OPEN_MODEL_ASSET_DIR "" CACHE PATH
    "Directory containing the fetched CC0 Coastal Cliff 01 glTF asset")

foreach(scene SIBENIK SPONZA SANMIGUEL)
    string(TOLOWER "${scene}" scene_lower)
    set(RAYMMA_${scene}_PATH "" CACHE FILEPATH
        "Path to the separately licensed ${scene_lower}.obj benchmark scene")
endforeach()

set(OPEN_MODEL_CACHE_DIR "${CMAKE_CURRENT_BINARY_DIR}/open-model-assets")
set(OPEN_MODEL_CACHE_FILES)
set(RAYMMA_HAS_OPEN_MODEL_ASSETS OFF)

if(RAYMMA_ENABLE_EXTERNAL_SCENES AND RAYMMA_OPEN_MODEL_ASSET_DIR)
    set(_raymma_open_model_input
        "${RAYMMA_OPEN_MODEL_ASSET_DIR}/coastal_cliff_01_1k.gltf")
    if(EXISTS "${_raymma_open_model_input}")
        find_program(BLENDER_EXECUTABLE blender)
        if(BLENDER_EXECUTABLE)
            set(_raymma_open_model_lods Low Mid High)
            set(_raymma_open_model_targets 8517 71313 0)
            foreach(lod target IN ZIP_LISTS
                    _raymma_open_model_lods _raymma_open_model_targets)
                set(output "${OPEN_MODEL_CACHE_DIR}/CoastalCliff${lod}.brtri")
                set(texture_output
                    "${OPEN_MODEL_CACHE_DIR}/CoastalCliff${lod}-albedo.png")
                add_custom_command(
                    OUTPUT "${output}" "${texture_output}"
                    COMMAND "${CMAKE_COMMAND}" -E make_directory
                            "${OPEN_MODEL_CACHE_DIR}"
                    COMMAND "${BLENDER_EXECUTABLE}"
                            --background --factory-startup
                            --python
                            "${CMAKE_CURRENT_SOURCE_DIR}/tools/extract_scene_triangles.py"
                            --
                            --input "${_raymma_open_model_input}"
                            --output "${output}"
                            --texture-output "${texture_output}"
                            --target-triangles "${target}"
                    DEPENDS
                        "${_raymma_open_model_input}"
                        "${CMAKE_CURRENT_SOURCE_DIR}/tools/extract_scene_triangles.py"
                    COMMENT "Extracting CC0 Coastal Cliff ${lod} ray-tracing triangles"
                    VERBATIM)
                list(APPEND OPEN_MODEL_CACHE_FILES
                    "${output}" "${texture_output}")
            endforeach()
            add_custom_target(
                open-model-ray-assets DEPENDS ${OPEN_MODEL_CACHE_FILES})
            set(RAYMMA_HAS_OPEN_MODEL_ASSETS ON)
        else()
            message(WARNING
                "The Coastal Cliff glTF was found, but Blender was not. "
                "Coastal Cliff scenes will be unavailable.")
        endif()
    else()
        message(WARNING
            "RAYMMA_OPEN_MODEL_ASSET_DIR does not contain the fetched "
            "Coastal Cliff asset. Run tools/fetch_open_model.sh first.")
    endif()
endif()

if(RAYMMA_ENABLE_EXTERNAL_SCENES)
    set(_raymma_sibenik_path "${RAYMMA_SIBENIK_PATH}")
    set(_raymma_sponza_path "${RAYMMA_SPONZA_PATH}")
    set(_raymma_sanmiguel_path "${RAYMMA_SANMIGUEL_PATH}")
else()
    set(_raymma_sibenik_path "")
    set(_raymma_sponza_path "")
    set(_raymma_sanmiguel_path "")
endif()

set(_raymma_coastal_cliff_low_path "")
set(_raymma_coastal_cliff_mid_path "")
set(_raymma_coastal_cliff_high_path "")
if(RAYMMA_HAS_OPEN_MODEL_ASSETS)
    set(_raymma_coastal_cliff_low_path
        "${OPEN_MODEL_CACHE_DIR}/CoastalCliffLow.brtri")
    set(_raymma_coastal_cliff_mid_path
        "${OPEN_MODEL_CACHE_DIR}/CoastalCliffMid.brtri")
    set(_raymma_coastal_cliff_high_path
        "${OPEN_MODEL_CACHE_DIR}/CoastalCliffHigh.brtri")
endif()

if(RAYMMA_BUILD_BENCHMARK)
    add_executable(tensor-wide-bvh-bench src/research_benchmark.cu)
    if(TARGET open-model-ray-assets)
        add_dependencies(tensor-wide-bvh-bench open-model-ray-assets)
    endif()
    target_compile_features(tensor-wide-bvh-bench PRIVATE cxx_std_17)
    target_compile_definitions(tensor-wide-bvh-bench PRIVATE
        RAYMMA_VERSION="${PROJECT_VERSION}"
        RAYMMA_BUILD_CONFIG="$<CONFIG>"
        COASTAL_CLIFF_LOW_PATH="${_raymma_coastal_cliff_low_path}"
        COASTAL_CLIFF_MID_PATH="${_raymma_coastal_cliff_mid_path}"
        COASTAL_CLIFF_HIGH_PATH="${_raymma_coastal_cliff_high_path}"
        SIBENIK_PATH="${_raymma_sibenik_path}"
        SPONZA_PATH="${_raymma_sponza_path}"
        SANMIGUEL_PATH="${_raymma_sanmiguel_path}")
    target_link_libraries(tensor-wide-bvh-bench PRIVATE CUDA::cudart)
    if(RAYMMA_ENABLE_TINYBVH)
        target_sources(tensor-wide-bvh-bench PRIVATE
            src/tinybvh_builder.cpp)
        target_include_directories(tensor-wide-bvh-bench PRIVATE
            "${RAYMMA_TINYBVH_INCLUDE_DIR}"
            "${CMAKE_CURRENT_SOURCE_DIR}/src")
        target_compile_definitions(tensor-wide-bvh-bench PRIVATE
            RAYMMA_HAS_TINYBVH=1)
    else()
        target_compile_definitions(tensor-wide-bvh-bench PRIVATE
            RAYMMA_HAS_TINYBVH=0)
    endif()
endif()

if(RAYMMA_BUILD_VIEWER)
    if(NOT RAYMMA_ENABLE_TINYBVH)
        message(FATAL_ERROR
            "The live viewer requires RAYMMA_ENABLE_TINYBVH=ON")
    endif()
    find_package(OpenGL REQUIRED)
    find_package(glfw3 REQUIRED)
    find_package(JPEG REQUIRED)
    find_package(PNG REQUIRED)
    add_executable(raymma-image-viewer src/viewer.cpp)
    target_compile_features(raymma-image-viewer PRIVATE cxx_std_17)
    target_link_libraries(raymma-image-viewer PRIVATE OpenGL::GL glfw)

    add_executable(raymma-viewer
        src/live_viewer.cu
        src/tinybvh_builder.cpp)
    if(TARGET open-model-ray-assets)
        add_dependencies(raymma-viewer open-model-ray-assets)
    endif()
    target_compile_features(raymma-viewer PRIVATE cxx_std_17)
    target_include_directories(raymma-viewer PRIVATE
        "${RAYMMA_TINYBVH_INCLUDE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_compile_definitions(raymma-viewer PRIVATE
        RAYMMA_HAS_TINYBVH=1)
    target_link_libraries(raymma-viewer PRIVATE
        CUDA::cudart OpenGL::GL glfw JPEG::JPEG PNG::PNG)
endif()

add_executable(tensor-ray-correctness
    prototype/tensor_ray_triangle_wmma.cu)
target_compile_features(tensor-ray-correctness PRIVATE cxx_std_17)
target_link_libraries(tensor-ray-correctness PRIVATE CUDA::cudart)

if(BUILD_TESTING)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_test(NAME tensor-ray-algebra
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/prototype/verify_algebra.py")
    add_test(NAME lambda-cloud-offline
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_lambda_cloud.py")
    add_test(NAME tensor-ray-wmma COMMAND tensor-ray-correctness)
    if(TARGET tensor-wide-bvh-bench)
        add_test(NAME tensor-wide-bvh-smoke
            COMMAND tensor-wide-bvh-bench --quick --scene Grid)
        add_test(NAME tensor-wide-bvh-candidate-rich
            COMMAND tensor-wide-bvh-bench
                    --quick --candidate-rich --scene Grid)
        add_test(NAME tensor-wide-bvh-uvt-depthsorted
            COMMAND tensor-wide-bvh-bench
                    --quick --variant uvt-depthsorted --scene Grid)
        add_test(NAME tensor-wide-bvh-e0e1e2
            COMMAND tensor-wide-bvh-bench
                    --quick --variant e0e1e2 --scene Grid)
        add_test(NAME tensor-wide-bvh-uvt-depthsorted-depth-order
            COMMAND tensor-wide-bvh-bench
                    --quick --resolution 17x17 --leaf 4
                    --variant uvt-depthsorted --scene RegressionDepthSorted)
        add_test(NAME tensor-wide-bvh-e0e1e2-depth-order
            COMMAND tensor-wide-bvh-bench
                    --quick --resolution 17x17 --leaf 4
                    --variant e0e1e2 --scene RegressionDepthSorted)
        add_test(NAME tensor-wide-bvh-uvt-depthsorted-depth-clip
            COMMAND tensor-wide-bvh-bench
                    --quick --resolution 17x17 --leaf 4
                    --variant uvt-depthsorted --scene RegressionDepthClip)
        add_test(NAME tensor-wide-bvh-e0e1e2-depth-clip
            COMMAND tensor-wide-bvh-bench
                    --quick --resolution 17x17 --leaf 4
                    --variant e0e1e2 --scene RegressionDepthClip)
        foreach(count 5 6 7)
            add_test(NAME tensor-wide-bvh-odd-${count}
                COMMAND tensor-wide-bvh-bench
                        --quick --leaf 4 --scene RegressionOdd${count})
        endforeach()
        add_test(NAME tensor-wide-bvh-fp16-range
            COMMAND tensor-wide-bvh-bench
                    --quick --resolution 17x17 --leaf 4
                    --scene RegressionTinyFar)
        add_test(NAME tensor-wide-bvh-secondary
            COMMAND tensor-wide-bvh-bench
                    --quick --ray-mode secondary --scene Grid)
        if(RAYMMA_ENABLE_TINYBVH)
            add_test(NAME tensor-wide-bvh-tinybvh-sah
                COMMAND tensor-wide-bvh-bench
                        --quick --bvh tinybvh-sah --scene Grid)
            add_test(NAME tensor-wide-bvh-tinybvh-sbvh
                COMMAND tensor-wide-bvh-bench
                        --quick --bvh tinybvh-sbvh --scene Grid)
        endif()
    endif()
endif()

message(STATUS
    "RayMMA: benchmark=${RAYMMA_BUILD_BENCHMARK}, "
    "viewer=${RAYMMA_BUILD_VIEWER}, "
    "TinyBVH=${RAYMMA_ENABLE_TINYBVH}, "
    "external scenes=${RAYMMA_ENABLE_EXTERNAL_SCENES}")

# Override CMAKE_CUDA_ARCHITECTURES for cross-compilation, e.g. 86 for Ampere.
