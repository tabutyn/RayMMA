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
    "Enable user-supplied Checker and standard benchmark scenes" OFF)
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

set(RAYMMA_CHECKER_ASSET_DIR "" CACHE PATH
    "Directory containing user-supplied Low1.glb, Mid1.glb, and High1.glb")

foreach(scene SIBENIK SPONZA SANMIGUEL)
    string(TOLOWER "${scene}" scene_lower)
    set(RAYMMA_${scene}_PATH "" CACHE FILEPATH
        "Path to the separately licensed ${scene_lower}.obj benchmark scene")
endforeach()

set(CHECKER_CACHE_DIR "${CMAKE_CURRENT_BINARY_DIR}/checker-assets")
set(CHECKER_CACHE_FILES)
set(RAYMMA_HAS_CHECKER_ASSETS OFF)

if(RAYMMA_ENABLE_EXTERNAL_SCENES AND RAYMMA_CHECKER_ASSET_DIR)
    set(_raymma_checker_inputs_present ON)
    foreach(lod Low Mid High)
        if(NOT EXISTS "${RAYMMA_CHECKER_ASSET_DIR}/${lod}1.glb")
            set(_raymma_checker_inputs_present OFF)
        endif()
    endforeach()

    if(_raymma_checker_inputs_present)
        find_program(BLENDER_EXECUTABLE blender)
        if(BLENDER_EXECUTABLE)
            foreach(lod Low Mid High)
                set(output "${CHECKER_CACHE_DIR}/${lod}1.brtri")
                set(texture_output
                    "${CHECKER_CACHE_DIR}/${lod}1-albedo.png")
                add_custom_command(
                    OUTPUT "${output}" "${texture_output}"
                    COMMAND "${CMAKE_COMMAND}" -E make_directory
                            "${CHECKER_CACHE_DIR}"
                    COMMAND "${BLENDER_EXECUTABLE}"
                            --background --factory-startup
                            --python
                            "${CMAKE_CURRENT_SOURCE_DIR}/tools/extract_checker_triangles.py"
                            --
                            --input
                            "${RAYMMA_CHECKER_ASSET_DIR}/${lod}1.glb"
                            --output "${output}"
                            --texture-output "${texture_output}"
                    DEPENDS
                        "${RAYMMA_CHECKER_ASSET_DIR}/${lod}1.glb"
                        "${CMAKE_CURRENT_SOURCE_DIR}/tools/extract_checker_triangles.py"
                    COMMENT "Extracting Checker1 ${lod} ray-tracing triangles"
                    VERBATIM)
                list(APPEND CHECKER_CACHE_FILES
                    "${output}" "${texture_output}")
            endforeach()
            add_custom_target(
                checker-ray-assets DEPENDS ${CHECKER_CACHE_FILES})
            set(RAYMMA_HAS_CHECKER_ASSETS ON)
        else()
            message(WARNING
                "Checker GLBs were found, but Blender was not. "
                "Checker scenes will be unavailable.")
        endif()
    else()
        message(WARNING
            "RAYMMA_CHECKER_ASSET_DIR does not contain all three Checker GLBs. "
            "Checker scenes will be unavailable.")
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

set(_raymma_checker_low_path "")
set(_raymma_checker_mid_path "")
set(_raymma_checker_high_path "")
if(RAYMMA_HAS_CHECKER_ASSETS)
    set(_raymma_checker_low_path "${CHECKER_CACHE_DIR}/Low1.brtri")
    set(_raymma_checker_mid_path "${CHECKER_CACHE_DIR}/Mid1.brtri")
    set(_raymma_checker_high_path "${CHECKER_CACHE_DIR}/High1.brtri")
endif()

if(RAYMMA_BUILD_BENCHMARK)
    add_executable(tensor-wide-bvh-bench src/research_benchmark.cu)
    if(TARGET checker-ray-assets)
        add_dependencies(tensor-wide-bvh-bench checker-ray-assets)
    endif()
    target_compile_features(tensor-wide-bvh-bench PRIVATE cxx_std_17)
    target_compile_definitions(tensor-wide-bvh-bench PRIVATE
        RAYMMA_VERSION="${PROJECT_VERSION}"
        RAYMMA_BUILD_CONFIG="$<CONFIG>"
        CHECKER_LOW_PATH="${_raymma_checker_low_path}"
        CHECKER_MID_PATH="${_raymma_checker_mid_path}"
        CHECKER_HIGH_PATH="${_raymma_checker_high_path}"
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
    add_test(NAME tensor-ray-wmma COMMAND tensor-ray-correctness)
    if(TARGET tensor-wide-bvh-bench)
        add_test(NAME tensor-wide-bvh-smoke
            COMMAND tensor-wide-bvh-bench --quick --scene Grid)
        add_test(NAME tensor-wide-bvh-candidate-rich
            COMMAND tensor-wide-bvh-bench
                    --quick --candidate-rich --scene Grid)
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
