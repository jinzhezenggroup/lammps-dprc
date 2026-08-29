cmake_minimum_required(VERSION 3.24)

project(lammps_dprc VERSION 0.1.0 LANGUAGES C CXX)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(FATAL_ERROR "In-source builds are not supported")
endif()

# Installed plugins are tied to a matching LAMMPS/MPI/toolchain environment,
# but must not retain absolute paths to the developer's build dependencies.
# The host deployment supplies libxtbloom and MPI through its normal loader
# configuration.
set(CMAKE_SKIP_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_REMOVE_ENVIRONMENT_RPATH TRUE)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

get_filename_component(_dprc_default_lammps "${CMAKE_CURRENT_SOURCE_DIR}/../lammps/src" ABSOLUTE)
get_filename_component(_dprc_default_xtbloom "${CMAKE_CURRENT_SOURCE_DIR}/../xtbloom" ABSOLUTE)
get_filename_component(_dprc_default_deepmd "${CMAKE_CURRENT_SOURCE_DIR}/../deepmd-kit" ABSOLUTE)
get_filename_component(
  _dprc_default_xtbloom_generated
  "${CMAKE_CURRENT_SOURCE_DIR}/../xtbloom/build/lammps-dprc-cpu/generated/include"
  ABSOLUTE)

set(LAMMPS_SOURCE_DIR "${_dprc_default_lammps}" CACHE PATH "Path to the pinned LAMMPS src directory")
set(XTBLOOM_SOURCE_DIR "${_dprc_default_xtbloom}" CACHE PATH "Path to the pinned xTBloom source tree")
set(DEEPMD_SOURCE_DIR "${_dprc_default_deepmd}" CACHE PATH
  "Path to the pinned DeePMD-kit checkout that produced the C API library")
set(XTBLOOM_GENERATED_INCLUDE_DIR "${_dprc_default_xtbloom_generated}" CACHE PATH
  "Path containing the generated xtbloom/version.h for the pinned source tree")
set(DPRC_XTBLOOM_LIBRARY "" CACHE FILEPATH
  "Runtime-provided xTBloom shared library used by public-C-ABI integration tests")
set(DPRC_EXPECTED_XTBLOOM_LIBRARY_SHA256 "" CACHE STRING
  "Optional expected SHA-256 for the explicitly selected xTBloom shared library")
set(DPRC_LAMMPS_EXECUTABLE "" CACHE FILEPATH "LAMMPS executable used for plugin load tests")
set(DPRC_REFERENCE_LAMMPS_EXECUTABLE "" CACHE FILEPATH
  "Optional QMMM-XTB-enabled LAMMPS executable used only for libxTB oracle calls")
set(DPRC_DEEPMD_INCLUDE_DIR "" CACHE PATH
  "Directory containing deepmd/c_api.h with canonical batch API version 31+")
set(DPRC_DEEPMD_C_LIBRARY "" CACHE FILEPATH
  "Runtime-provided DeePMD C API shared library linked by dprcplugin")
set(DPRC_DEEPMD_MODEL "" CACHE FILEPATH
  "Optional frozen canonical DeePMD model used for DPRc runtime tests")
set(DPRC_EXPECTED_DEEPMD_C_LIBRARY_SHA256 "" CACHE STRING
  "Required SHA-256 of DPRC_DEEPMD_C_LIBRARY when the C API integration is enabled")
set(DPRC_EXPECTED_DEEPMD_MODEL_SHA256 "" CACHE STRING
  "Required SHA-256 of DPRC_DEEPMD_MODEL when runtime tests are enabled")
set(DPRC_LAMMPS_SIZES "smallbig" CACHE STRING "LAMMPS integer-size mode")
set_property(CACHE DPRC_LAMMPS_SIZES PROPERTY STRINGS smallbig bigbig)
set(DPRC_XTBLOOM_BACKEND "AUTO" CACHE STRING
  "xTBloom backend selected by qmmm/xtb/dprc (AUTO, CPU, or CUDA)")
set_property(CACHE DPRC_XTBLOOM_BACKEND PROPERTY STRINGS AUTO CPU CUDA)
set(DPRC_XTBLOOM_DEVICE_ID "-1" CACHE STRING
  "CUDA device ordinal for qmmm/xtb/dprc; -1 selects the current/default device")
set(DPRC_XTBLOOM_CPU_THREADS "0" CACHE STRING
  "CPU worker count for the one broker-owned xTBloom context; 0 uses the runtime default")

set(DPRC_EXPECTED_LAMMPS_REVISION
    "7bd373ebc61a8028be955e6be862d5a091fd59c5" CACHE STRING "Supported LAMMPS revision")
set(DPRC_EXPECTED_XTBLOOM_REVISION
    "3c474e1c1b639098f72ae7523472bd5f65ad3ab5" CACHE STRING "Supported xTBloom revision")
set(DPRC_EXPECTED_DEEPMD_REVISION
    "835a42f001b41e1bee0646b7e1403855b7fe6340" CACHE STRING
    "Reviewed DeePMD compact-evaluation revision")
option(DPRC_ALLOW_UNPINNED_DEPENDENCIES "Allow dependency revisions other than the reviewed pins" OFF)
option(DPRC_REQUIRE_CLEAN_DEPENDENCIES "Reject dirty required dependency worktrees" ON)
option(DPRC_BUILD_TESTING "Build and register scaffold tests" ON)
option(DPRC_ENABLE_CLASSICAL_CUDA
  "Build the single-context batched CUDA LJ/TIP4P/PPPM backend" OFF)
option(DPRC_ENABLE_KOKKOS_RUNTIME_TESTS
  "Run GPU tests that require a Kokkos-enabled LAMMPS executable" OFF)
option(DPRC_REQUIRE_XTBLOOM_LIBRARY
  "Fail configuration unless the xTBloom integration library is available" OFF)
option(DPRC_REQUIRE_DEEPMD_C_API
  "Fail configuration unless the DeePMD C API library is available" OFF)
set(DPRC_CUFFT_INCLUDE_DIR "" CACHE PATH
  "Directory containing cufft.h for the classical CUDA backend")
set(DPRC_CUFFT_LIBRARY "" CACHE FILEPATH
  "Exact shared cuFFT library for the classical CUDA backend")
set(DPRC_EXPECTED_CUFFT_HEADER_SHA256 "" CACHE STRING
  "Required SHA-256 of the explicitly selected cufft.h")
set(DPRC_EXPECTED_CUFFT_LIBRARY_SHA256 "" CACHE STRING
  "Required SHA-256 of the explicitly selected cuFFT shared library")

if(DPRC_ENABLE_CLASSICAL_CUDA)
  # Keep compiler and cuFFT selection explicit.  This machine can use a staged
  # CUDA compiler with cuFFT supplied by another reviewed toolkit payload, so
  # FindCUDAToolkit's single-root assumption would select the wrong artifacts.
  enable_language(CUDA)
  if(NOT DPRC_CUFFT_INCLUDE_DIR OR
     NOT EXISTS "${DPRC_CUFFT_INCLUDE_DIR}/cufft.h")
    message(FATAL_ERROR
      "DPRC_ENABLE_CLASSICAL_CUDA=ON requires DPRC_CUFFT_INCLUDE_DIR containing cufft.h")
  endif()
  if(NOT DPRC_CUFFT_LIBRARY OR NOT EXISTS "${DPRC_CUFFT_LIBRARY}")
    message(FATAL_ERROR
      "DPRC_ENABLE_CLASSICAL_CUDA=ON requires an exact DPRC_CUFFT_LIBRARY path")
  endif()
  if(NOT DPRC_EXPECTED_CUFFT_HEADER_SHA256 OR
     NOT DPRC_EXPECTED_CUFFT_LIBRARY_SHA256)
    message(FATAL_ERROR
      "The classical CUDA backend requires expected SHA-256 values for cufft.h and the cuFFT library")
  endif()
  file(SHA256 "${DPRC_CUFFT_INCLUDE_DIR}/cufft.h" _dprc_cufft_header_sha256)
  file(REAL_PATH "${DPRC_CUFFT_LIBRARY}" _dprc_cufft_library_realpath)
  file(SHA256 "${_dprc_cufft_library_realpath}" _dprc_cufft_library_sha256)
  if(NOT _dprc_cufft_header_sha256 STREQUAL DPRC_EXPECTED_CUFFT_HEADER_SHA256)
    message(FATAL_ERROR
      "cufft.h SHA-256 ${_dprc_cufft_header_sha256} differs from the expected value")
  endif()
  if(NOT _dprc_cufft_library_sha256 STREQUAL DPRC_EXPECTED_CUFFT_LIBRARY_SHA256)
    message(FATAL_ERROR
      "cuFFT library SHA-256 ${_dprc_cufft_library_sha256} differs from the expected value")
  endif()
  message(STATUS
    "cuFFT runtime: ${_dprc_cufft_library_realpath} (SHA-256 ${_dprc_cufft_library_sha256})")
endif()

function(dprc_verify_git_dependency name source_path expected_revision)
  get_filename_component(repository_path "${source_path}/.." ABSOLUTE)
  if(name STREQUAL "xTBloom")
    set(repository_path "${source_path}")
  endif()

  execute_process(
    COMMAND git -C "${repository_path}" rev-parse HEAD
    OUTPUT_VARIABLE actual_revision
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE revision_status)
  if(NOT revision_status EQUAL 0)
    message(FATAL_ERROR "${name} is not a readable Git checkout: ${repository_path}")
  endif()

  if(NOT actual_revision STREQUAL expected_revision)
    if(DPRC_ALLOW_UNPINNED_DEPENDENCIES)
      message(WARNING "${name} revision ${actual_revision} differs from reviewed pin ${expected_revision}")
    else()
      message(FATAL_ERROR
        "${name} revision ${actual_revision} differs from reviewed pin ${expected_revision}; "
        "set DPRC_ALLOW_UNPINNED_DEPENDENCIES=ON only for diagnostic development")
    endif()
  endif()

  if(DPRC_REQUIRE_CLEAN_DEPENDENCIES)
    execute_process(
      COMMAND git -C "${repository_path}" status --porcelain
      OUTPUT_VARIABLE dirty_output
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE dirty_status)
    if(NOT dirty_status EQUAL 0 OR NOT dirty_output STREQUAL "")
      message(FATAL_ERROR "${name} checkout must be clean for this configuration: ${repository_path}")
    endif()
  endif()
endfunction()

if(NOT EXISTS "${LAMMPS_SOURCE_DIR}/lammpsplugin.h" OR
   NOT EXISTS "${LAMMPS_SOURCE_DIR}/version.h")
  message(FATAL_ERROR "LAMMPS_SOURCE_DIR must point to a LAMMPS src directory")
endif()
if(NOT EXISTS "${XTBLOOM_SOURCE_DIR}/include/xtbloom/xtbloom.h")
  message(FATAL_ERROR "XTBLOOM_SOURCE_DIR must point to the xTBloom source tree")
endif()
if(NOT EXISTS "${XTBLOOM_GENERATED_INCLUDE_DIR}/xtbloom/version.h")
  message(FATAL_ERROR
    "XTBLOOM_GENERATED_INCLUDE_DIR must contain xtbloom/version.h; configure the pinned "
    "xTBloom source tree before configuring this project")
endif()

dprc_verify_git_dependency("LAMMPS" "${LAMMPS_SOURCE_DIR}" "${DPRC_EXPECTED_LAMMPS_REVISION}")
dprc_verify_git_dependency("xTBloom" "${XTBLOOM_SOURCE_DIR}" "${DPRC_EXPECTED_XTBLOOM_REVISION}")

# The legacy LAMMPS command exposes xTB's MM-hardness policy rather than
# xTBloom's explicit per-site gamma array. Generate only the two canonical
# element tables needed for that translation into the build tree, with both
# the source revision and byte hashes pinned.
find_package(Python3 3.11 REQUIRED COMPONENTS Interpreter)
set(DPRC_EXPECTED_GFN1_PARAMETER_SHA256
  "0ecdc3f5f12990c5a7e0f0bd7e6fe931ecf72d7630e6a6a3cc396c51766a40a0")
set(DPRC_EXPECTED_GFN2_PARAMETER_SHA256
  "de0f20e90b592b7b92f107eb672bd3dd29c1096f904d7a472b05693f9238ed1a")
set(DPRC_GENERATED_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/include")
set(DPRC_GENERATED_HARDNESS_HEADER
  "${DPRC_GENERATED_INCLUDE_DIR}/dprc_xtb_hardness.h")
execute_process(
  COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_xtb_hardness.py"
    --gfn1 "${XTBLOOM_SOURCE_DIR}/data/parameters/gfn1.json"
    --gfn2 "${XTBLOOM_SOURCE_DIR}/data/parameters/gfn2.json"
    --xtbloom-revision "${DPRC_EXPECTED_XTBLOOM_REVISION}"
    --expected-gfn1-sha256 "${DPRC_EXPECTED_GFN1_PARAMETER_SHA256}"
    --expected-gfn2-sha256 "${DPRC_EXPECTED_GFN2_PARAMETER_SHA256}"
    --output "${DPRC_GENERATED_HARDNESS_HEADER}"
  RESULT_VARIABLE _dprc_hardness_status
  ERROR_VARIABLE _dprc_hardness_error)
if(NOT _dprc_hardness_status EQUAL 0)
  message(FATAL_ERROR "Could not generate pinned xTB hardness table: ${_dprc_hardness_error}")
endif()

set(MPI_CXX_SKIP_MPICXX TRUE)
find_package(MPI REQUIRED COMPONENTS CXX)

if(DPRC_XTBLOOM_LIBRARY AND NOT EXISTS "${DPRC_XTBLOOM_LIBRARY}")
  message(FATAL_ERROR "DPRC_XTBLOOM_LIBRARY does not exist: ${DPRC_XTBLOOM_LIBRARY}")
endif()
if(NOT WIN32 AND DPRC_XTBLOOM_LIBRARY MATCHES "\\.a$")
  message(FATAL_ERROR
    "DPRC_XTBLOOM_LIBRARY must be a runtime-provided shared library, not a static archive")
endif()
if(DPRC_XTBLOOM_LIBRARY)
  file(REAL_PATH "${DPRC_XTBLOOM_LIBRARY}" _dprc_xtbloom_library_realpath)
  file(SHA256 "${_dprc_xtbloom_library_realpath}" _dprc_xtbloom_library_sha256)
  if(DPRC_EXPECTED_XTBLOOM_LIBRARY_SHA256 AND
     NOT _dprc_xtbloom_library_sha256 STREQUAL DPRC_EXPECTED_XTBLOOM_LIBRARY_SHA256)
    message(FATAL_ERROR
      "xTBloom shared-library SHA-256 ${_dprc_xtbloom_library_sha256} differs from "
      "DPRC_EXPECTED_XTBLOOM_LIBRARY_SHA256=${DPRC_EXPECTED_XTBLOOM_LIBRARY_SHA256}")
  endif()
  set(DPRC_XTBLOOM_LIBRARY_REALPATH "${_dprc_xtbloom_library_realpath}" CACHE INTERNAL
    "Resolved xTBloom shared-library path used by this build" FORCE)
  set(DPRC_XTBLOOM_LIBRARY_SHA256 "${_dprc_xtbloom_library_sha256}" CACHE INTERNAL
    "SHA-256 of the xTBloom shared library used by this build" FORCE)
endif()
if(DPRC_REQUIRE_XTBLOOM_LIBRARY AND NOT DPRC_XTBLOOM_LIBRARY)
  message(FATAL_ERROR
    "DPRC_REQUIRE_XTBLOOM_LIBRARY=ON requires an explicit runtime-provided shared library")
endif()

# dprcplugin owns the LAMMPS pair style and links only DeePMD's public C API.
# The DeePMD LAMMPS plugin and its private C++ ABI are deliberately absent.
set(_dprc_deepmd_enabled OFF)
set(_dprc_deepmd_runtime_test_enabled OFF)
if(DPRC_DEEPMD_C_LIBRARY OR DPRC_DEEPMD_INCLUDE_DIR)
  if(NOT DPRC_DEEPMD_C_LIBRARY OR NOT DPRC_DEEPMD_INCLUDE_DIR)
    message(FATAL_ERROR
      "DPRC_DEEPMD_C_LIBRARY and DPRC_DEEPMD_INCLUDE_DIR must be supplied together")
  endif()
  if(NOT EXISTS "${DPRC_DEEPMD_INCLUDE_DIR}/deepmd/c_api.h")
    message(FATAL_ERROR
      "DPRC_DEEPMD_INCLUDE_DIR must contain deepmd/c_api.h")
  endif()
  if(NOT EXISTS "${DPRC_DEEPMD_C_LIBRARY}")
    message(FATAL_ERROR
      "DPRC_DEEPMD_C_LIBRARY does not exist: ${DPRC_DEEPMD_C_LIBRARY}")
  endif()
  if(NOT WIN32 AND DPRC_DEEPMD_C_LIBRARY MATCHES "\\.a$")
    message(FATAL_ERROR
      "DPRC_DEEPMD_C_LIBRARY must be a runtime-provided shared library")
  endif()
  if(NOT DPRC_EXPECTED_DEEPMD_C_LIBRARY_SHA256)
    message(FATAL_ERROR
      "The DeePMD C API integration requires DPRC_EXPECTED_DEEPMD_C_LIBRARY_SHA256")
  endif()
  if(NOT EXISTS "${DEEPMD_SOURCE_DIR}/source/api_c/include/c_api.h")
    message(FATAL_ERROR
      "DEEPMD_SOURCE_DIR must point to the checkout that produced the C API library")
  endif()
  file(READ "${DPRC_DEEPMD_INCLUDE_DIR}/deepmd/c_api.h"
    _dprc_deepmd_c_header LIMIT 4096)
  if(NOT _dprc_deepmd_c_header MATCHES "#define[ \t]+DP_C_API_VERSION[ \t]+([0-9]+)" OR
     CMAKE_MATCH_1 LESS 31)
    message(FATAL_ERROR
      "dprc/deepmd/batch requires DeePMD C API version 31 or newer")
  endif()
  dprc_verify_git_dependency(
    "DeePMD-kit" "${DEEPMD_SOURCE_DIR}/source" "${DPRC_EXPECTED_DEEPMD_REVISION}")
  file(REAL_PATH "${DPRC_DEEPMD_C_LIBRARY}" _dprc_deepmd_c_library_realpath)
  file(SHA256 "${_dprc_deepmd_c_library_realpath}"
    _dprc_deepmd_c_library_sha256)
  if(NOT _dprc_deepmd_c_library_sha256 STREQUAL
     DPRC_EXPECTED_DEEPMD_C_LIBRARY_SHA256)
    message(FATAL_ERROR
      "DeePMD C API library SHA-256 ${_dprc_deepmd_c_library_sha256} differs from the expected value")
  endif()
  find_package(CUDAToolkit REQUIRED)
  add_library(dprc_deepmd_c SHARED IMPORTED GLOBAL)
  set_target_properties(dprc_deepmd_c PROPERTIES
    IMPORTED_LOCATION "${_dprc_deepmd_c_library_realpath}"
    INTERFACE_INCLUDE_DIRECTORIES "${DPRC_DEEPMD_INCLUDE_DIR}")
  set(_dprc_deepmd_enabled ON)
  message(STATUS
    "DeePMD C API runtime: ${_dprc_deepmd_c_library_realpath} "
    "(SHA-256 ${_dprc_deepmd_c_library_sha256})")

  if(DPRC_DEEPMD_MODEL)
    if(NOT DPRC_LAMMPS_EXECUTABLE OR NOT DPRC_BUILD_TESTING)
      message(FATAL_ERROR
        "DPRC_DEEPMD_MODEL requires DPRC_LAMMPS_EXECUTABLE and DPRC_BUILD_TESTING=ON")
    endif()
    if(NOT EXISTS "${DPRC_DEEPMD_MODEL}" OR
       NOT DPRC_EXPECTED_DEEPMD_MODEL_SHA256)
      message(FATAL_ERROR
        "The DeePMD runtime test requires an existing model and its expected SHA-256")
    endif()
    file(REAL_PATH "${DPRC_DEEPMD_MODEL}" _dprc_deepmd_model_realpath)
    file(SHA256 "${_dprc_deepmd_model_realpath}" _dprc_deepmd_model_sha256)
    if(NOT _dprc_deepmd_model_sha256 STREQUAL
       DPRC_EXPECTED_DEEPMD_MODEL_SHA256)
      message(FATAL_ERROR
        "DeePMD model SHA-256 ${_dprc_deepmd_model_sha256} differs from the expected value")
    endif()
    set(_dprc_deepmd_runtime_test_enabled ON)
  endif()
elseif(DPRC_DEEPMD_MODEL)
  message(FATAL_ERROR
    "DPRC_DEEPMD_MODEL requires DPRC_DEEPMD_C_LIBRARY and DPRC_DEEPMD_INCLUDE_DIR")
elseif(DPRC_REQUIRE_DEEPMD_C_API)
  message(FATAL_ERROR
    "DPRC_REQUIRE_DEEPMD_C_API=ON requires the DeePMD C API include directory and library")
endif()

add_library(dprc_lammps_headers INTERFACE)
target_include_directories(dprc_lammps_headers INTERFACE "${LAMMPS_SOURCE_DIR}")
target_link_libraries(dprc_lammps_headers INTERFACE MPI::MPI_CXX)
if(DPRC_LAMMPS_SIZES STREQUAL "smallbig")
  target_compile_definitions(dprc_lammps_headers INTERFACE LAMMPS_SMALLBIG)
elseif(DPRC_LAMMPS_SIZES STREQUAL "bigbig")
  target_compile_definitions(dprc_lammps_headers INTERFACE LAMMPS_BIGBIG)
else()
  message(FATAL_ERROR "DPRC_LAMMPS_SIZES must be smallbig or bigbig")
endif()

add_library(dprc_batch_core STATIC src/stable_batch.cpp)
target_include_directories(dprc_batch_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_compile_options(dprc_batch_core PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic>)
set_target_properties(dprc_batch_core PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES)

# Standalone classical oracle and backend boundary.  It is intentionally kept
# independent of LAMMPS C++ types so the numerical kernels can be qualified
# before the private Pair/KSpace publication proxies are enabled.
add_library(dprc_classical_batch STATIC src/classical_batch.cpp)
target_include_directories(dprc_classical_batch PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_compile_options(dprc_classical_batch PRIVATE
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU,Clang>>:-Wall;-Wextra;-Wpedantic>)
set_target_properties(dprc_classical_batch PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES)
if(DPRC_ENABLE_CLASSICAL_CUDA)
  target_sources(dprc_classical_batch PRIVATE src/classical_batch_cuda.cu)
  target_include_directories(dprc_classical_batch PRIVATE
    "${DPRC_CUFFT_INCLUDE_DIR}")
  target_link_libraries(dprc_classical_batch PUBLIC "${DPRC_CUFFT_LIBRARY}")
  target_compile_definitions(dprc_classical_batch PRIVATE
    DPRC_HAVE_CLASSICAL_CUDA=1)
  set_target_properties(dprc_classical_batch PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_EXTENSIONS OFF)
endif()

add_library(dprc_classical_partition_broker STATIC
  src/classical_partition_broker.cpp)
target_include_directories(dprc_classical_partition_broker PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(dprc_classical_partition_broker PUBLIC
  dprc_classical_batch MPI::MPI_CXX)
target_compile_options(dprc_classical_partition_broker PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic>)
set_target_properties(dprc_classical_partition_broker PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES)

add_library(dprc_partition_roots STATIC src/partition_roots.cpp)
target_include_directories(dprc_partition_roots PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(dprc_partition_roots PUBLIC MPI::MPI_CXX)
target_compile_options(dprc_partition_roots PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic>)
set_target_properties(dprc_partition_roots PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES)

if(_dprc_deepmd_enabled)
  add_library(dprc_deepmd_executor STATIC src/deepmd_batch_executor.cpp)
  target_include_directories(dprc_deepmd_executor PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_link_libraries(dprc_deepmd_executor PUBLIC
    dprc_deepmd_c CUDA::cudart)
  target_compile_options(dprc_deepmd_executor PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic>)
  set_target_properties(dprc_deepmd_executor PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES)

  add_library(dprc_deepmd_partition_broker STATIC
    src/deepmd_partition_broker.cpp)
  target_include_directories(dprc_deepmd_partition_broker PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_link_libraries(dprc_deepmd_partition_broker PUBLIC
    dprc_deepmd_executor MPI::MPI_CXX)
  target_compile_options(dprc_deepmd_partition_broker PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic>)
  set_target_properties(dprc_deepmd_partition_broker PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES)

  add_library(dprc_lammps_deepmd_pair STATIC
    src/pair_dprc_deepmd_batch.cpp)
  target_include_directories(dprc_lammps_deepmd_pair PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_link_libraries(dprc_lammps_deepmd_pair PUBLIC
    dprc_lammps_headers
    dprc_partition_roots
    dprc_deepmd_partition_broker)
  target_compile_options(dprc_lammps_deepmd_pair PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic>)
  set_target_properties(dprc_lammps_deepmd_pair PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES)
endif()

add_library(dprc_xtb_legacy STATIC src/xtb_legacy_hardness.cpp)
target_include_directories(dprc_xtb_legacy PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/src"
  "${DPRC_GENERATED_INCLUDE_DIR}")
target_compile_options(dprc_xtb_legacy PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic>)
set_target_properties(dprc_xtb_legacy PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES)

add_library(dprc_point_charge_slots STATIC src/point_charge_slots.cpp)
target_include_directories(dprc_point_charge_slots PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_compile_options(dprc_point_charge_slots PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic>)
set_target_properties(dprc_point_charge_slots PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES)

# Compile the xTBloom adapter unconditionally against the pinned public header.
# Linking and numerical tests are enabled only when the user explicitly
# supplies a runtime shared library; no binary is vendored or installed by this
# project.
add_library(dprc_xtbloom_executor STATIC src/xtbloom_plan_executor.cpp)
target_include_directories(dprc_xtbloom_executor PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/src"
  "${XTBLOOM_SOURCE_DIR}/include"
  "${XTBLOOM_GENERATED_INCLUDE_DIR}")
target_link_libraries(dprc_xtbloom_executor PUBLIC dprc_batch_core)
target_compile_options(dprc_xtbloom_executor PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic>)
set_target_properties(dprc_xtbloom_executor PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES)
if(DPRC_XTBLOOM_LIBRARY)
  target_link_libraries(dprc_xtbloom_executor PUBLIC "${DPRC_XTBLOOM_LIBRARY}")
  message(STATUS
    "xTBloom integration library: ${DPRC_XTBLOOM_LIBRARY_REALPATH} "
    "(SHA-256 ${DPRC_XTBLOOM_LIBRARY_SHA256})")
else()
  message(STATUS
    "DPRC_XTBLOOM_LIBRARY was not supplied; public-header adapter compiles but numerical smoke is skipped")
endif()

add_library(dprc_partition_broker STATIC src/xtbloom_partition_broker.cpp)
target_include_directories(dprc_partition_broker PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(dprc_partition_broker PUBLIC
  dprc_xtbloom_executor MPI::MPI_CXX)
target_compile_options(dprc_partition_broker PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic>)
set_target_properties(dprc_partition_broker PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES)

if(DPRC_XTBLOOM_BACKEND STREQUAL "AUTO")
  set(_dprc_xtbloom_backend_value 0)
elseif(DPRC_XTBLOOM_BACKEND STREQUAL "CPU")
  set(_dprc_xtbloom_backend_value 1)
elseif(DPRC_XTBLOOM_BACKEND STREQUAL "CUDA")
  set(_dprc_xtbloom_backend_value 2)
else()
  message(FATAL_ERROR "DPRC_XTBLOOM_BACKEND must be AUTO, CPU, or CUDA")
endif()
if(NOT DPRC_XTBLOOM_DEVICE_ID MATCHES "^-?[0-9]+$" OR DPRC_XTBLOOM_DEVICE_ID LESS -1)
  message(FATAL_ERROR "DPRC_XTBLOOM_DEVICE_ID must be an integer greater than or equal to -1")
endif()
if(NOT DPRC_XTBLOOM_CPU_THREADS MATCHES "^[0-9]+$")
  message(FATAL_ERROR "DPRC_XTBLOOM_CPU_THREADS must be a non-negative integer")
endif()

if(DPRC_XTBLOOM_LIBRARY)
  set(_dprc_fused_manifest
    "${CMAKE_CURRENT_SOURCE_DIR}/config/fused_lammps_sources.json")
  set(_dprc_fused_patch
    "${CMAKE_CURRENT_SOURCE_DIR}/patches/lammps-qmmm-xtb-fused.patch")
  set(_dprc_fused_source_dir
    "${CMAKE_CURRENT_BINARY_DIR}/generated/lammps-qmmm-xtb")
  get_filename_component(_dprc_lammps_repository "${LAMMPS_SOURCE_DIR}/.." ABSOLUTE)
  if(NOT EXISTS "${LAMMPS_SOURCE_DIR}/QMMM-XTB/fix_qmmm_xtb.cpp" OR
     NOT EXISTS "${LAMMPS_SOURCE_DIR}/QMMM-XTB/qmmm_xtb_adapter.h" OR
     NOT EXISTS "${LAMMPS_SOURCE_DIR}/QMMM-XTB/qmmm_xtb_ewald.cpp" OR
     NOT EXISTS "${LAMMPS_SOURCE_DIR}/QMMM-XTB/qmmm_xtb_ewald.h")
    message(FATAL_ERROR
      "The pinned LAMMPS QMMM-XTB package sources are required for qmmm/xtb/dprc")
  endif()

  # Generate only in the build tree. The generator verifies all source, patch,
  # and output hashes so a patch that merely still applies after upstream drift
  # cannot silently enter the plugin.
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${_dprc_fused_manifest}"
    "${_dprc_fused_patch}"
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_fused_lammps_sources.py"
    "${LAMMPS_SOURCE_DIR}/QMMM-XTB/fix_qmmm_xtb.cpp"
    "${LAMMPS_SOURCE_DIR}/QMMM-XTB/fix_qmmm_xtb.h"
    "${LAMMPS_SOURCE_DIR}/QMMM-XTB/pppm_tip4p_xtb.h"
    "${LAMMPS_SOURCE_DIR}/QMMM-XTB/pppm_xtb.h"
    "${LAMMPS_SOURCE_DIR}/QMMM-XTB/qmmm_xtb_ewald.cpp"
    "${LAMMPS_SOURCE_DIR}/QMMM-XTB/qmmm_xtb_ewald.h"
    "${LAMMPS_SOURCE_DIR}/QMMM-XTB/qmmm_xtb_pppm.cpp")
  execute_process(
    COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_fused_lammps_sources.py"
      --repository-root "${CMAKE_CURRENT_SOURCE_DIR}"
      --lammps-root "${_dprc_lammps_repository}"
      --manifest "${_dprc_fused_manifest}"
      --output-dir "${_dprc_fused_source_dir}"
    RESULT_VARIABLE _dprc_fused_generation_status
    ERROR_VARIABLE _dprc_fused_generation_error)
  if(NOT _dprc_fused_generation_status EQUAL 0)
    message(FATAL_ERROR
      "Could not generate pinned fused LAMMPS sources: ${_dprc_fused_generation_error}")
  endif()

  set(_dprc_reference_fix_source
    "${_dprc_fused_source_dir}/fix_qmmm_xtb.cpp")
  set(_dprc_reference_pppm_source
    "${_dprc_fused_source_dir}/qmmm_xtb_pppm.cpp")
  set(_dprc_reference_ewald_source
    "${_dprc_fused_source_dir}/qmmm_xtb_ewald.cpp")

  # Rename every implementation and adapter symbol that would otherwise
  # collide with the built-in qmmm/xtb and PPPM-XTB styles.  The private Ewald
  # implementation is included because a valid plugin host need not build the
  # QMMM-XTB package at all; renaming it also prevents accidental interposition
  # when the host does provide the original class.
  set_source_files_properties("${_dprc_reference_fix_source}" PROPERTIES
    COMPILE_DEFINITIONS
      "FixQMMMXTB=FixDPRCXtbReference;PPPMXTB=PPPMDPRC;PPPMTIP4PXTB=PPPMTIP4PDPRC;QMMMXTBEwald=DPRCXtbEwald;lammps_qmmm_xtb_create=dprc_lammps_xtb_create;lammps_qmmm_xtb_calculate=dprc_lammps_xtb_calculate;lammps_qmmm_xtb_last_error=dprc_lammps_xtb_last_error;lammps_qmmm_xtb_destroy=dprc_lammps_xtb_destroy")
  set_source_files_properties("${_dprc_reference_pppm_source}" PROPERTIES
    COMPILE_DEFINITIONS
      "PPPMXTB=PPPMDPRC;PPPMTIP4PXTB=PPPMTIP4PDPRC;QMMMXTBPPPMHelper=DPRCQMMMReferencePPPMHelper")
  set_source_files_properties("${_dprc_reference_ewald_source}" PROPERTIES
    COMPILE_DEFINITIONS "QMMMXTBEwald=DPRCXtbEwald")

  add_library(dprc_lammps_xtb_fix STATIC
    "${_dprc_reference_fix_source}"
    "${_dprc_reference_pppm_source}"
    "${_dprc_reference_ewald_source}"
    src/fix_dprc_xtb.cpp
    src/xtbloom_lammps_adapter.cpp)
  if(DPRC_ENABLE_CLASSICAL_CUDA)
    target_sources(dprc_lammps_xtb_fix PRIVATE
      src/fix_dprc_classical_batch.cpp
      src/kspace_dprc_batched.cpp
      src/pair_dprc_batched_lj.cpp
      src/pair_dprc_batched_tip4p.cpp)
  endif()
  target_include_directories(dprc_lammps_xtb_fix PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${_dprc_fused_source_dir}"
    "${LAMMPS_SOURCE_DIR}/QMMM-XTB")
  target_include_directories(dprc_lammps_xtb_fix PRIVATE
    "${LAMMPS_SOURCE_DIR}/KSPACE"
    "${LAMMPS_SOURCE_DIR}/MOLECULE"
    "${XTBLOOM_SOURCE_DIR}/include"
    "${XTBLOOM_GENERATED_INCLUDE_DIR}")
  target_link_libraries(dprc_lammps_xtb_fix PUBLIC
    dprc_lammps_headers
    dprc_partition_broker
    dprc_partition_roots
    dprc_point_charge_slots
    dprc_xtb_legacy)
  if(DPRC_ENABLE_CLASSICAL_CUDA)
    target_link_libraries(dprc_lammps_xtb_fix PUBLIC
      dprc_classical_partition_broker)
    target_compile_definitions(dprc_lammps_xtb_fix PRIVATE
      DPRC_HAVE_CLASSICAL_BATCH=1)
  endif()
  target_compile_definitions(dprc_lammps_xtb_fix PRIVATE
    DPRC_XTBLOOM_BACKEND_VALUE=${_dprc_xtbloom_backend_value}
    DPRC_XTBLOOM_DEVICE_ID=${DPRC_XTBLOOM_DEVICE_ID}
    DPRC_XTBLOOM_CPU_THREADS=${DPRC_XTBLOOM_CPU_THREADS})
  if(DPRC_BUILD_TESTING)
    target_compile_definitions(dprc_lammps_xtb_fix PRIVATE
      DPRC_ENABLE_TEST_HOOKS=1)
  endif()
  target_compile_options(dprc_lammps_xtb_fix PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic>)
  set_target_properties(dprc_lammps_xtb_fix PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES)
endif()

add_library(dprcplugin MODULE
  src/command_dprc_info.cpp
  src/plugin.cpp)
target_include_directories(dprcplugin PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/src"
  "${LAMMPS_SOURCE_DIR}/KSPACE"
  "${LAMMPS_SOURCE_DIR}/MOLECULE")
target_link_libraries(dprcplugin PRIVATE dprc_lammps_headers dprc_partition_roots)
target_compile_definitions(dprcplugin PRIVATE DPRC_PROJECT_VERSION="${PROJECT_VERSION}")
if(_dprc_deepmd_enabled)
  target_link_libraries(dprcplugin PRIVATE dprc_lammps_deepmd_pair)
  target_compile_definitions(dprcplugin PRIVATE DPRC_HAVE_DEEPMD_BATCH=1)
endif()
if(DPRC_XTBLOOM_LIBRARY)
  target_link_libraries(dprcplugin PRIVATE dprc_lammps_xtb_fix)
  target_compile_definitions(dprcplugin PRIVATE DPRC_HAVE_XTBLOOM_FIX=1)
  if(DPRC_ENABLE_CLASSICAL_CUDA)
    target_compile_definitions(dprcplugin PRIVATE DPRC_HAVE_CLASSICAL_BATCH=1)
  endif()
  if(DPRC_BUILD_TESTING)
    target_sources(dprcplugin PRIVATE src/command_dprc_test_recovery.cpp)
    target_compile_definitions(dprcplugin PRIVATE DPRC_ENABLE_TEST_HOOKS=1)
  endif()
endif()
target_compile_options(dprcplugin PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic>)
set_target_properties(dprcplugin PROPERTIES
  PREFIX ""
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES)

if(APPLE)
  target_link_options(dprcplugin PRIVATE
    "LINKER:-undefined,dynamic_lookup"
    "LINKER:-exported_symbols_list,${CMAKE_CURRENT_SOURCE_DIR}/cmake/dprcplugin.exports")
  set_property(TARGET dprcplugin APPEND PROPERTY LINK_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/dprcplugin.exports")
elseif(UNIX)
  # LAMMPS loads plugins with RTLD_GLOBAL. Keep every C++ implementation
  # symbol local so another plugin cannot interpose on this private ABI.
  target_link_options(dprcplugin PRIVATE
    "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/cmake/dprcplugin.map")
  set_property(TARGET dprcplugin APPEND PROPERTY LINK_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/dprcplugin.map")
endif()

include(GNUInstallDirs)
install(TARGETS dprcplugin
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}/lammps/plugins"
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}/lammps/plugins")
if(UNIX AND NOT APPLE)
  # Some conda compiler wrappers inject toolchain paths after CMake has
  # computed INSTALL_RPATH. Remove the final ELF RPATH explicitly so an
  # installed plugin never names this build machine.
  install(CODE
    "file(RPATH_REMOVE FILE \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/lammps/plugins/$<TARGET_FILE_NAME:dprcplugin>\")")
endif()
install(FILES README.md LICENSE THIRD_PARTY_NOTICES.md
  DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/lammps-dprc")
install(FILES docs/user-guide.md docs/lammps-build-and-run.md
  DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/lammps-dprc/docs")
install(FILES
    LICENSES/GPL-2.0-only.txt
    LICENSES/GPL-3.0-only.txt
  DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/lammps-dprc/LICENSES")

if(DPRC_BUILD_TESTING)
  include(CTest)

  add_executable(stable_batch_test tests/stable_batch_test.cpp)
  target_link_libraries(stable_batch_test PRIVATE dprc_batch_core)
  add_test(NAME dprc.stable_batch COMMAND stable_batch_test)

  add_executable(classical_batch_test tests/classical_batch_test.cpp)
  target_link_libraries(classical_batch_test PRIVATE dprc_classical_batch)
  if(DPRC_ENABLE_CLASSICAL_CUDA)
    target_compile_definitions(classical_batch_test PRIVATE
      DPRC_TEST_CLASSICAL_CUDA=1)
    target_include_directories(classical_batch_test PRIVATE
      "${DPRC_CUFFT_INCLUDE_DIR}"
      "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
    if(UNIX AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
      # Test-only linker wrapping records the exact planMany batch shapes and
      # execution directions without changing the production CUDA library.
      target_compile_definitions(classical_batch_test PRIVATE
        DPRC_TEST_CUFFT_WRAP=1)
      target_link_options(classical_batch_test PRIVATE
        "-Wl,--wrap=cufftMakePlanMany"
        "-Wl,--wrap=cufftExecZ2Z")
    endif()
  endif()
  add_test(NAME dprc.classical_batch COMMAND classical_batch_test)

  if(MPIEXEC_EXECUTABLE)
    add_executable(classical_partition_broker_test
      tests/classical_partition_broker_test.cpp)
    target_link_libraries(classical_partition_broker_test PRIVATE
      dprc_classical_partition_broker)
    add_test(NAME dprc.classical_partition_broker
      COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" 2
        $<TARGET_FILE:classical_partition_broker_test>)
    set_tests_properties(dprc.classical_partition_broker PROPERTIES TIMEOUT 120)
  endif()

  add_executable(xtbloom_header_smoke tests/xtbloom_header_smoke.c)
  target_include_directories(xtbloom_header_smoke PRIVATE
    "${XTBLOOM_SOURCE_DIR}/include"
    "${XTBLOOM_GENERATED_INCLUDE_DIR}")
  add_test(NAME xtbloom.header COMMAND xtbloom_header_smoke)

  add_executable(xtb_legacy_hardness_test tests/xtb_legacy_hardness_test.cpp)
  target_link_libraries(xtb_legacy_hardness_test PRIVATE dprc_xtb_legacy)
  add_test(NAME xtb.legacy_hardness COMMAND xtb_legacy_hardness_test)
  add_executable(point_charge_slots_test tests/point_charge_slots_test.cpp)
  target_link_libraries(point_charge_slots_test PRIVATE dprc_point_charge_slots)
  add_test(NAME xtb.point_charge_slots COMMAND point_charge_slots_test)
  add_test(NAME xtb.legacy_hardness_generated
    COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_xtb_hardness.py"
      --gfn1 "${XTBLOOM_SOURCE_DIR}/data/parameters/gfn1.json"
      --gfn2 "${XTBLOOM_SOURCE_DIR}/data/parameters/gfn2.json"
      --xtbloom-revision "${DPRC_EXPECTED_XTBLOOM_REVISION}"
      --expected-gfn1-sha256 "${DPRC_EXPECTED_GFN1_PARAMETER_SHA256}"
      --expected-gfn2-sha256 "${DPRC_EXPECTED_GFN2_PARAMETER_SHA256}"
      --output "${DPRC_GENERATED_HARDNESS_HEADER}"
      --check)

  add_test(NAME dependencies.artifact_pins
    COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/check_dependency_pins_test.py")
  add_test(NAME dependencies.fused_source_generator
    COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/fused_source_generator_test.py")
  add_test(NAME licensing.source_payload
    COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/check_license_payload.py"
      --source-root "${CMAKE_CURRENT_SOURCE_DIR}")
  add_test(NAME licensing.install_payload
    COMMAND "${CMAKE_COMMAND}"
      -DBUILD_DIR=${CMAKE_CURRENT_BINARY_DIR}
      -DINSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/install-payload-test
      -DPYTHON_EXECUTABLE=${Python3_EXECUTABLE}
      -DCHECK_SCRIPT=${CMAKE_CURRENT_SOURCE_DIR}/tests/check_license_payload.py
      -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_install_payload.cmake")
  add_test(NAME workload.etpeth
    COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/etpeth_workload_test.py")
  add_test(NAME benchmark.etpeth_matrix
    COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/test_run.py")
  add_test(NAME workload.etpeth_pmf
    COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/etpeth_pmf_test.py")
  add_test(NAME workload.dprc_labels
    COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/dprc_labels_test.py")
  add_test(NAME workload.quick_pbe0_qualification
    COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/quick_pbe0_qualification_test.py")
  add_test(NAME workload.dprc_binary64_io
    COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/dprc_binary64_io_test.py")
  add_test(NAME workload.dprc_correction_io
    COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/dprc_correction_io_test.py")
  add_test(NAME workload.lammps_xtb_oracle_compare
    COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/lammps_xtb_oracle_compare_test.py")
  add_test(NAME workload.deepmd_overlay_input
    COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/deepmd_overlay_input_test.py")
  add_test(NAME workload.deepmd_center_mask_input
    COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/deepmd_center_mask_input_test.py")

  if(UNIX AND NOT APPLE)
    add_test(NAME lammps.plugin.symbols
      COMMAND "${CMAKE_COMMAND}"
        -DNM_EXECUTABLE=${CMAKE_NM}
        -DREADELF_EXECUTABLE=${CMAKE_READELF}
        -DDEEPMD_C_API_ENABLED=${_dprc_deepmd_enabled}
        -DPLUGIN_FILE=$<TARGET_FILE:dprcplugin>
        -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/check_plugin_symbols.cmake")
  endif()

  if(_dprc_deepmd_enabled)
    add_executable(deepmd_c_api_header_smoke
      tests/deepmd_c_api_header_smoke.c)
    target_link_libraries(deepmd_c_api_header_smoke PRIVATE dprc_deepmd_c)
    add_test(NAME deepmd.c_api_header COMMAND deepmd_c_api_header_smoke)
  endif()

  if(DPRC_XTBLOOM_LIBRARY)
    add_test(NAME lammps.fused_sources.generated
      COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_fused_lammps_sources.py"
        --repository-root "${CMAKE_CURRENT_SOURCE_DIR}"
        --lammps-root "${_dprc_lammps_repository}"
        --manifest "${_dprc_fused_manifest}"
        --output-dir "${_dprc_fused_source_dir}"
        --check)

    add_executable(xtbloom_plan_executor_test
      tests/xtbloom_plan_executor_test.cpp)
    target_link_libraries(xtbloom_plan_executor_test PRIVATE
      dprc_xtbloom_executor)
    add_test(NAME xtbloom.plan_qmmm_batch COMMAND xtbloom_plan_executor_test)
    set_tests_properties(xtbloom.plan_qmmm_batch PROPERTIES TIMEOUT 120)

    if(MPIEXEC_EXECUTABLE)
      add_executable(xtbloom_partition_broker_test
        tests/xtbloom_partition_broker_test.cpp)
      target_link_libraries(xtbloom_partition_broker_test PRIVATE
        dprc_partition_broker)
      add_test(NAME xtbloom.partition_broker
        COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" 2
          $<TARGET_FILE:xtbloom_partition_broker_test>)
      set_tests_properties(xtbloom.partition_broker PROPERTIES TIMEOUT 120)
    endif()
  endif()

  if(DPRC_LAMMPS_EXECUTABLE)
    if(NOT EXISTS "${DPRC_LAMMPS_EXECUTABLE}")
      message(FATAL_ERROR "DPRC_LAMMPS_EXECUTABLE does not exist")
    endif()
    add_test(NAME lammps.plugin.load
      COMMAND "${DPRC_LAMMPS_EXECUTABLE}"
        -log none
        -screen none
        -var dprc_plugin $<TARGET_FILE:dprcplugin>
        -var dprc_marker "${CMAKE_CURRENT_BINARY_DIR}/dprc-info.txt"
        -in "${CMAKE_CURRENT_SOURCE_DIR}/tests/in.dprc_info")
    set_tests_properties(lammps.plugin.load PROPERTIES TIMEOUT 30)
    add_test(NAME lammps.plugin.output
      COMMAND "${CMAKE_COMMAND}"
        -DOUTPUT_FILE=${CMAKE_CURRENT_BINARY_DIR}/dprc-info.txt
        -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/check_info_output.cmake")
    set_tests_properties(lammps.plugin.output PROPERTIES DEPENDS lammps.plugin.load)

    if(_dprc_deepmd_runtime_test_enabled AND
       DPRC_ENABLE_KOKKOS_RUNTIME_TESTS)
      add_test(NAME lammps.deepmd_c_api.center_mask
        COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_lammps_deepmd_center_mask.py"
          --lammps "${DPRC_LAMMPS_EXECUTABLE}"
          --dprc-plugin $<TARGET_FILE:dprcplugin>
          --model "${_dprc_deepmd_model_realpath}"
          --deepmd-revision "${DPRC_EXPECTED_DEEPMD_REVISION}"
          --evidence
            "${CMAKE_CURRENT_BINARY_DIR}/deepmd-center-mask-evidence.json")
      set_tests_properties(lammps.deepmd_c_api.center_mask PROPERTIES
        TIMEOUT 180 LABELS "cuda;gpu;kokkos;deepmd")
      if(DPRC_XTBLOOM_LIBRARY)
        add_test(NAME lammps.deepmd_c_api.qmmm_overlay
          COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_lammps_deepmd_overlay.py"
            --lammps "${DPRC_LAMMPS_EXECUTABLE}"
            --dprc-plugin $<TARGET_FILE:dprcplugin>
            --model "${_dprc_deepmd_model_realpath}"
            --deepmd-revision "${DPRC_EXPECTED_DEEPMD_REVISION}"
            --evidence
              "${CMAKE_CURRENT_BINARY_DIR}/deepmd-overlay-evidence.json")
        set_tests_properties(lammps.deepmd_c_api.qmmm_overlay PROPERTIES
          TIMEOUT 240 LABELS "cuda;gpu;qmmm;deepmd")
      endif()
      if(MPIEXEC_EXECUTABLE)
        add_test(NAME lammps.deepmd_c_api.partition_batch
          COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_lammps_deepmd_partition_batch.py"
            --lammps "${DPRC_LAMMPS_EXECUTABLE}"
            --mpiexec "${MPIEXEC_EXECUTABLE}"
            --dprc-plugin $<TARGET_FILE:dprcplugin>
            --model "${_dprc_deepmd_model_realpath}"
            --deepmd-revision "${DPRC_EXPECTED_DEEPMD_REVISION}"
            --evidence
              "${CMAKE_CURRENT_BINARY_DIR}/deepmd-partition-batch-evidence.json")
        set_tests_properties(lammps.deepmd_c_api.partition_batch PROPERTIES
          TIMEOUT 240 LABELS "cuda;gpu;mpi;deepmd")
      endif()
    endif()

    if(DPRC_XTBLOOM_LIBRARY)
      set(_dprc_reference_lammps_arguments)
      set(_dprc_mixed_style_lammps_executable "${DPRC_LAMMPS_EXECUTABLE}")
      if(DPRC_REFERENCE_LAMMPS_EXECUTABLE)
        if(NOT EXISTS "${DPRC_REFERENCE_LAMMPS_EXECUTABLE}")
          message(FATAL_ERROR
            "DPRC_REFERENCE_LAMMPS_EXECUTABLE does not exist")
        endif()
        list(APPEND _dprc_reference_lammps_arguments
          --reference-lammps "${DPRC_REFERENCE_LAMMPS_EXECUTABLE}")
        # The two mixed native/private rejection tests need both style
        # families in one process, so they must run in the QMMM-XTB host.
        set(_dprc_mixed_style_lammps_executable
          "${DPRC_REFERENCE_LAMMPS_EXECUTABLE}")
      endif()
      add_test(NAME lammps.qmmm_xtb.reference_compare
        COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_lammps_qmmm_compare.py"
          ${_dprc_reference_lammps_arguments}
          --lammps "${DPRC_LAMMPS_EXECUTABLE}"
          --plugin $<TARGET_FILE:dprcplugin>
          --input "${CMAKE_CURRENT_SOURCE_DIR}/tests/in.qmmm_xtb_compare"
          --evidence "${CMAKE_CURRENT_BINARY_DIR}/qmmm-reference-evidence.json")
      set_tests_properties(lammps.qmmm_xtb.reference_compare PROPERTIES TIMEOUT 120)
      add_test(NAME lammps.qmmm_xtb.reference_compare_triclinic
        COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_lammps_qmmm_compare.py"
          ${_dprc_reference_lammps_arguments}
          --lammps "${DPRC_LAMMPS_EXECUTABLE}"
          --plugin $<TARGET_FILE:dprcplugin>
          --input "${CMAKE_CURRENT_SOURCE_DIR}/tests/in.qmmm_xtb_compare_triclinic"
          --evidence "${CMAKE_CURRENT_BINARY_DIR}/qmmm-reference-triclinic-evidence.json")
      set_tests_properties(lammps.qmmm_xtb.reference_compare_triclinic PROPERTIES
        TIMEOUT 120)
      add_test(NAME lammps.qmmm_xtb.tip4p_reference_compare
        COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_lammps_qmmm_compare.py"
          ${_dprc_reference_lammps_arguments}
          --lammps "${DPRC_LAMMPS_EXECUTABLE}"
          --plugin $<TARGET_FILE:dprcplugin>
          --input "${CMAKE_CURRENT_SOURCE_DIR}/tests/in.qmmm_xtb_tip4p_compare"
          --natoms 5
          --reference-kspace pppm/tip4p/xtb
          --actual-kspace pppm/tip4p/dprc
          --evidence "${CMAKE_CURRENT_BINARY_DIR}/qmmm-tip4p-reference-evidence.json")
      set_tests_properties(lammps.qmmm_xtb.tip4p_reference_compare PROPERTIES TIMEOUT 120)
      add_test(NAME lammps.qmmm_xtb.tip4p_reference_compare_triclinic
        COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_lammps_qmmm_compare.py"
          ${_dprc_reference_lammps_arguments}
          --lammps "${DPRC_LAMMPS_EXECUTABLE}"
          --plugin $<TARGET_FILE:dprcplugin>
          --input "${CMAKE_CURRENT_SOURCE_DIR}/tests/in.qmmm_xtb_tip4p_compare_triclinic"
          --natoms 5
          --reference-kspace pppm/tip4p/xtb
          --actual-kspace pppm/tip4p/dprc
          --evidence "${CMAKE_CURRENT_BINARY_DIR}/qmmm-tip4p-reference-triclinic-evidence.json")
      set_tests_properties(lammps.qmmm_xtb.tip4p_reference_compare_triclinic PROPERTIES
        TIMEOUT 120)

      add_test(NAME lammps.qmmm_xtb.cutoff_skin_crossing
        COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_lammps_qmmm_compare.py"
          ${_dprc_reference_lammps_arguments}
          --lammps "${DPRC_LAMMPS_EXECUTABLE}"
          --plugin $<TARGET_FILE:dprcplugin>
          --input "${CMAKE_CURRENT_SOURCE_DIR}/tests/in.qmmm_xtb_cutoff_skin_crossing"
          --natoms 5
          --expect-actual-batch-calls 2
          --expect-actual-plan-rebuilds 1
          --expect-actual-fresh-calls 1
          --expect-actual-warm-calls 1
          --expect-actual-capacity-growths 0
          --require-actual-point-padding
          --evidence "${CMAKE_CURRENT_BINARY_DIR}/qmmm-cutoff-skin-crossing-evidence.json")
      set_tests_properties(lammps.qmmm_xtb.cutoff_skin_crossing PROPERTIES
        TIMEOUT 120 ENVIRONMENT "LAMMPS_QMMM_XTB_PROFILE=1")

      add_test(NAME lammps.qmmm_xtb.cutoff_skin_zero_shell
        COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_lammps_qmmm_compare.py"
          ${_dprc_reference_lammps_arguments}
          --lammps "${DPRC_LAMMPS_EXECUTABLE}"
          --plugin $<TARGET_FILE:dprcplugin>
          --input "${CMAKE_CURRENT_SOURCE_DIR}/tests/in.qmmm_xtb_cutoff_skin_zero_shell"
          --natoms 3
          --expect-actual-batch-calls 1
          --expect-actual-plan-rebuilds 1
          --expect-actual-fresh-calls 1
          --expect-actual-warm-calls 0
          --expect-actual-capacity-growths 0
          --require-actual-point-padding
          --evidence "${CMAKE_CURRENT_BINARY_DIR}/qmmm-cutoff-skin-zero-shell-evidence.json")
      set_tests_properties(lammps.qmmm_xtb.cutoff_skin_zero_shell PROPERTIES
        TIMEOUT 120 ENVIRONMENT "LAMMPS_QMMM_XTB_PROFILE=1")

      add_test(NAME lammps.qmmm_xtb.cutoff_skin_epoch_growth
        COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_lammps_qmmm_compare.py"
          ${_dprc_reference_lammps_arguments}
          --lammps "${DPRC_LAMMPS_EXECUTABLE}"
          --plugin $<TARGET_FILE:dprcplugin>
          --input "${CMAKE_CURRENT_SOURCE_DIR}/tests/in.qmmm_xtb_cutoff_skin_epoch_growth"
          --natoms 5
          --expect-actual-batch-calls 2
          --expect-actual-plan-rebuilds 1
          --expect-actual-fresh-calls 1
          --expect-actual-warm-calls 1
          --expect-actual-capacity-growths 0
          --require-actual-point-padding
          --evidence "${CMAKE_CURRENT_BINARY_DIR}/qmmm-cutoff-skin-epoch-growth-evidence.json")
      set_tests_properties(lammps.qmmm_xtb.cutoff_skin_epoch_growth PROPERTIES
        TIMEOUT 120 ENVIRONMENT "LAMMPS_QMMM_XTB_PROFILE=1")

      if(DPRC_ENABLE_CLASSICAL_CUDA)
        add_test(NAME lammps.classical.batched_tip4p_reference_compare_triclinic
          COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_lammps_classical_compare.py"
            --lammps "${DPRC_LAMMPS_EXECUTABLE}"
            --plugin $<TARGET_FILE:dprcplugin>
            --input
              "${CMAKE_CURRENT_SOURCE_DIR}/tests/in.classical_batched_tip4p_compare_triclinic"
            --evidence
              "${CMAKE_CURRENT_BINARY_DIR}/classical-batched-tip4p-triclinic-evidence.json")
        set_tests_properties(
          lammps.classical.batched_tip4p_reference_compare_triclinic
          PROPERTIES TIMEOUT 180 LABELS "cuda;gpu")

        if(MPIEXEC_EXECUTABLE)
          add_test(NAME lammps.classical.batched_tip4p_partition
            COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" 2
              "${DPRC_LAMMPS_EXECUTABLE}"
              -partition 2x1
              -log none
              -screen none
              -var dprc_plugin $<TARGET_FILE:dprcplugin>
              -in
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/in.classical_batched_tip4p_partition")
          set_tests_properties(lammps.classical.batched_tip4p_partition
            PROPERTIES TIMEOUT 180 LABELS "cuda;gpu;mpi")
        endif()

        add_test(NAME lammps.qmmm_xtb.batched_tip4p_reference_compare_triclinic
          COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_lammps_qmmm_compare.py"
            ${_dprc_reference_lammps_arguments}
            --lammps "${DPRC_LAMMPS_EXECUTABLE}"
            --plugin $<TARGET_FILE:dprcplugin>
            --input
              "${CMAKE_CURRENT_SOURCE_DIR}/tests/in.qmmm_xtb_batched_tip4p_compare_triclinic"
            --data-file
              "${CMAKE_CURRENT_SOURCE_DIR}/tests/data.qmmm_xtb_batched_tip4p_triclinic"
            --natoms 5
            --reference-kspace pppm/tip4p/xtb
            --actual-kspace pppm/tip4p/dprc/batch
            --reference-lj lj/cut
            --actual-lj lj/cut/dprc/batch
            --reference-coulomb tip4p/long
            --actual-coulomb tip4p/long/dprc/batch
            --evidence
              "${CMAKE_CURRENT_BINARY_DIR}/qmmm-batched-tip4p-triclinic-evidence.json")
        set_tests_properties(
          lammps.qmmm_xtb.batched_tip4p_reference_compare_triclinic
          PROPERTIES TIMEOUT 180 LABELS "cuda;gpu")

        if(DPRC_ENABLE_KOKKOS_RUNTIME_TESTS)
          add_test(NAME lammps.qmmm_xtb.batched_tip4p_kokkos_reference_compare_triclinic
            COMMAND "${Python3_EXECUTABLE}"
              "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_lammps_qmmm_compare.py"
              ${_dprc_reference_lammps_arguments}
              --lammps "${DPRC_LAMMPS_EXECUTABLE}"
              --plugin $<TARGET_FILE:dprcplugin>
              --input
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/in.qmmm_xtb_batched_tip4p_compare_triclinic"
              --data-file
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/data.qmmm_xtb_batched_tip4p_triclinic"
              --natoms 5
              --reference-kspace pppm/tip4p/xtb
              --actual-kspace pppm/tip4p/dprc/batch
              --reference-lj lj/cut
              --actual-lj lj/cut/dprc/batch
              --reference-coulomb tip4p/long
              --actual-coulomb tip4p/long/dprc/batch
              --actual-hybrid-style hybrid/overlay/kk
              --actual-kokkos
              --expect-actual-mm-only-routing
              --evidence
                "${CMAKE_CURRENT_BINARY_DIR}/qmmm-batched-tip4p-kokkos-triclinic-evidence.json")
          set_tests_properties(
            lammps.qmmm_xtb.batched_tip4p_kokkos_reference_compare_triclinic
            PROPERTIES TIMEOUT 180 LABELS "cuda;gpu;kokkos")
        endif()
      endif()
      add_test(NAME lammps.qmmm_xtb.finite_difference
        COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_lammps_qmmm_finite_difference.py"
          --lammps "${DPRC_LAMMPS_EXECUTABLE}"
          --plugin $<TARGET_FILE:dprcplugin>
          --input "${CMAKE_CURRENT_SOURCE_DIR}/tests/in.qmmm_xtb_finite_difference"
          --evidence "${CMAKE_CURRENT_BINARY_DIR}/qmmm-finite-difference-evidence.json")
      set_tests_properties(lammps.qmmm_xtb.finite_difference PROPERTIES TIMEOUT 120)

      add_test(NAME lammps.qmmm_xtb.repeated_fused_state
        COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_lammps_qmmm_compare.py"
          ${_dprc_reference_lammps_arguments}
          --lammps "${DPRC_LAMMPS_EXECUTABLE}"
          --plugin $<TARGET_FILE:dprcplugin>
          --input "${CMAKE_CURRENT_SOURCE_DIR}/tests/in.qmmm_xtb_repeat_compare"
          --evidence "${CMAKE_CURRENT_BINARY_DIR}/qmmm-repeated-state-evidence.json")
      set_tests_properties(lammps.qmmm_xtb.repeated_fused_state PROPERTIES TIMEOUT 120)

      add_test(NAME lammps.qmmm_xtb.change_box_reference_compare
        COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_lammps_qmmm_compare.py"
          ${_dprc_reference_lammps_arguments}
          --lammps "${DPRC_LAMMPS_EXECUTABLE}"
          --plugin $<TARGET_FILE:dprcplugin>
          --input "${CMAKE_CURRENT_SOURCE_DIR}/tests/in.qmmm_xtb_change_box_compare"
          --evidence "${CMAKE_CURRENT_BINARY_DIR}/qmmm-change-box-evidence.json")
      set_tests_properties(lammps.qmmm_xtb.change_box_reference_compare PROPERTIES
        TIMEOUT 120)

      add_test(NAME lammps.qmmm_xtb.change_box_triclinic_reference_compare
        COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_lammps_qmmm_compare.py"
          ${_dprc_reference_lammps_arguments}
          --lammps "${DPRC_LAMMPS_EXECUTABLE}"
          --plugin $<TARGET_FILE:dprcplugin>
          --input
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/in.qmmm_xtb_change_box_triclinic_compare"
          --evidence
            "${CMAKE_CURRENT_BINARY_DIR}/qmmm-change-box-triclinic-evidence.json")
      set_tests_properties(
        lammps.qmmm_xtb.change_box_triclinic_reference_compare
        PROPERTIES TIMEOUT 120)

      add_test(NAME lammps.qmmm_xtb.failure_transaction_recovery
        COMMAND "${DPRC_LAMMPS_EXECUTABLE}"
          -log none
          -screen none
          -var dprc_plugin $<TARGET_FILE:dprcplugin>
          -in "${CMAKE_CURRENT_SOURCE_DIR}/tests/in.qmmm_xtb_failure_recovery")
      set_tests_properties(lammps.qmmm_xtb.failure_transaction_recovery PROPERTIES
        TIMEOUT 180)

      add_test(NAME lammps.qmmm_xtb.reject_diff_ad
        COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_lammps_expect_failure.py"
          --lammps "${DPRC_LAMMPS_EXECUTABLE}"
          --plugin $<TARGET_FILE:dprcplugin>
          --input "${CMAKE_CURRENT_SOURCE_DIR}/tests/in.qmmm_xtb_diff_ad_reject"
          --qmmm-style qmmm/xtb/dprc
          --kspace-style pppm/dprc
          --expect "requires ik PPPM"
          --evidence "${CMAKE_CURRENT_BINARY_DIR}/qmmm-reject-diff-ad.json")
      set_tests_properties(lammps.qmmm_xtb.reject_diff_ad PROPERTIES TIMEOUT 120)

      add_test(NAME lammps.qmmm_xtb.reject_peratom_kspace
        COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_lammps_expect_failure.py"
          --lammps "${DPRC_LAMMPS_EXECUTABLE}"
          --plugin $<TARGET_FILE:dprcplugin>
          --input "${CMAKE_CURRENT_SOURCE_DIR}/tests/in.qmmm_xtb_peratom_reject"
          --qmmm-style qmmm/xtb/dprc
          --kspace-style pppm/dprc
          --expect "does not support per-atom KSpace output"
          --evidence "${CMAKE_CURRENT_BINARY_DIR}/qmmm-reject-peratom.json")
      set_tests_properties(lammps.qmmm_xtb.reject_peratom_kspace PROPERTIES TIMEOUT 120)

      add_test(NAME lammps.qmmm_xtb.reject_pair_compute_no
        COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_lammps_expect_failure.py"
          --lammps "${DPRC_LAMMPS_EXECUTABLE}"
          --plugin $<TARGET_FILE:dprcplugin>
          --input "${CMAKE_CURRENT_SOURCE_DIR}/tests/in.qmmm_xtb_pair_compute_no"
          --qmmm-style qmmm/xtb/dprc
          --kspace-style pppm/dprc
          --expect "requires pair_modify compute yes"
          --evidence "${CMAKE_CURRENT_BINARY_DIR}/qmmm-reject-pair-compute-no.json")
      set_tests_properties(lammps.qmmm_xtb.reject_pair_compute_no PROPERTIES TIMEOUT 120)

      add_test(NAME lammps.qmmm_xtb.reject_coulomb_substyle_compute_no
        COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_lammps_expect_failure.py"
          --lammps "${DPRC_LAMMPS_EXECUTABLE}"
          --plugin $<TARGET_FILE:dprcplugin>
          --input "${CMAKE_CURRENT_SOURCE_DIR}/tests/in.qmmm_xtb_coulomb_substyle_compute_no"
          --qmmm-style qmmm/xtb/dprc
          --kspace-style pppm/dprc
          --expect "requires its Coulomb pair sub-style to compute"
          --evidence "${CMAKE_CURRENT_BINARY_DIR}/qmmm-reject-coulomb-substyle-compute-no.json")
      set_tests_properties(lammps.qmmm_xtb.reject_coulomb_substyle_compute_no PROPERTIES
        TIMEOUT 120)

      add_test(NAME lammps.qmmm_xtb.reject_private_fix_native_kspace
        COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_lammps_expect_failure.py"
          --lammps "${_dprc_mixed_style_lammps_executable}"
          --plugin $<TARGET_FILE:dprcplugin>
          --input "${CMAKE_CURRENT_SOURCE_DIR}/tests/in.qmmm_xtb_compare"
          --qmmm-style qmmm/xtb/dprc
          --kspace-style pppm/xtb
          --expect "incompatible xTB KSpace implementation"
          --evidence "${CMAKE_CURRENT_BINARY_DIR}/qmmm-reject-private-native.json")
      set_tests_properties(lammps.qmmm_xtb.reject_private_fix_native_kspace PROPERTIES TIMEOUT 120)

      add_test(NAME lammps.qmmm_xtb.reject_native_fix_private_kspace
        COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_lammps_expect_failure.py"
          --lammps "${_dprc_mixed_style_lammps_executable}"
          --plugin $<TARGET_FILE:dprcplugin>
          --input "${CMAKE_CURRENT_SOURCE_DIR}/tests/in.qmmm_xtb_compare"
          --qmmm-style qmmm/xtb
          --kspace-style pppm/dprc
          --expect "incompatible xTB KSpace implementation"
          --evidence "${CMAKE_CURRENT_BINARY_DIR}/qmmm-reject-native-private.json")
      set_tests_properties(lammps.qmmm_xtb.reject_native_fix_private_kspace PROPERTIES TIMEOUT 120)

    endif()

    if(MPIEXEC_EXECUTABLE)
      add_test(NAME lammps.plugin.partition
        COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" 2
          "${DPRC_LAMMPS_EXECUTABLE}"
          -partition 2x1
          -log none
          -screen none
          -var dprc_plugin $<TARGET_FILE:dprcplugin>
          -var dprc_marker "${CMAKE_CURRENT_BINARY_DIR}/dprc-partition.txt"
          -in "${CMAKE_CURRENT_SOURCE_DIR}/tests/in.dprc_info")
      set_tests_properties(lammps.plugin.partition PROPERTIES TIMEOUT 30)
      add_test(NAME lammps.plugin.partition_output
        COMMAND "${CMAKE_COMMAND}"
          -DOUTPUT_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/dprc-partition.txt
          -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/check_partition_output.cmake")
      set_tests_properties(lammps.plugin.partition_output PROPERTIES
        DEPENDS lammps.plugin.partition)

      add_test(NAME lammps.plugin.partition_multirank
        COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" 4
          "${DPRC_LAMMPS_EXECUTABLE}"
          -partition 2x2
          -log none
          -screen none
          -var dprc_plugin $<TARGET_FILE:dprcplugin>
          -var dprc_marker "${CMAKE_CURRENT_BINARY_DIR}/dprc-partition-multirank.txt"
          -in "${CMAKE_CURRENT_SOURCE_DIR}/tests/in.dprc_info")
      set_tests_properties(lammps.plugin.partition_multirank PROPERTIES TIMEOUT 30)
      add_test(NAME lammps.plugin.partition_multirank_output
        COMMAND "${CMAKE_COMMAND}"
          -DOUTPUT_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/dprc-partition-multirank.txt
          -DEXPECTED_WORLD_RANKS=2
          -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/check_partition_output.cmake")
      set_tests_properties(lammps.plugin.partition_multirank_output PROPERTIES
        DEPENDS lammps.plugin.partition_multirank)

      add_test(NAME lammps.plugin.partition_reordered
        COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" 4
          "${DPRC_LAMMPS_EXECUTABLE}"
          -reorder nth 2
          -partition 2x2
          -log none
          -screen none
          -var dprc_plugin $<TARGET_FILE:dprcplugin>
          -var dprc_marker "${CMAKE_CURRENT_BINARY_DIR}/dprc-partition-reordered.txt"
          -in "${CMAKE_CURRENT_SOURCE_DIR}/tests/in.dprc_info")
      set_tests_properties(lammps.plugin.partition_reordered PROPERTIES TIMEOUT 30)
      add_test(NAME lammps.plugin.partition_reordered_output
        COMMAND "${CMAKE_COMMAND}"
          -DOUTPUT_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/dprc-partition-reordered.txt
          -DEXPECTED_WORLD_RANKS=2
          -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/check_partition_output.cmake")
      set_tests_properties(lammps.plugin.partition_reordered_output PROPERTIES
        DEPENDS lammps.plugin.partition_reordered)

      if(DPRC_XTBLOOM_LIBRARY)
        add_test(NAME lammps.qmmm_xtb.partition_ragged_rebuild
          COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" 2
            "${DPRC_LAMMPS_EXECUTABLE}"
            -partition 2x1
            -log none
            -screen none
            -var dprc_plugin $<TARGET_FILE:dprcplugin>
            -in "${CMAKE_CURRENT_SOURCE_DIR}/tests/in.qmmm_xtb_partition")
        set_tests_properties(lammps.qmmm_xtb.partition_ragged_rebuild PROPERTIES
          TIMEOUT 120)

        if(DPRC_ENABLE_CLASSICAL_CUDA)
          add_test(NAME lammps.qmmm_xtb.batched_tip4p_partition
            COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" 2
              "${DPRC_LAMMPS_EXECUTABLE}"
              -partition 2x1
              -log none
              -screen none
              -var dprc_plugin $<TARGET_FILE:dprcplugin>
              -in
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/in.qmmm_xtb_batched_tip4p_partition")
          set_tests_properties(lammps.qmmm_xtb.batched_tip4p_partition
            PROPERTIES TIMEOUT 180 LABELS "cuda;gpu;mpi")
        endif()

        add_test(NAME lammps.qmmm_xtb.mpi_reference_compare
          COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_lammps_qmmm_compare.py"
            ${_dprc_reference_lammps_arguments}
            --lammps "${DPRC_LAMMPS_EXECUTABLE}"
            --plugin $<TARGET_FILE:dprcplugin>
            --input "${CMAKE_CURRENT_SOURCE_DIR}/tests/in.qmmm_xtb_compare"
            --mpi-exec "${MPIEXEC_EXECUTABLE}"
            "--mpi-numproc-flag=${MPIEXEC_NUMPROC_FLAG}"
            --mpi-procs 2
            --evidence "${CMAKE_CURRENT_BINARY_DIR}/qmmm-mpi-reference-evidence.json")
        add_test(NAME lammps.qmmm_xtb.mpi_reference_compare_triclinic
          COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_lammps_qmmm_compare.py"
            ${_dprc_reference_lammps_arguments}
            --lammps "${DPRC_LAMMPS_EXECUTABLE}"
            --plugin $<TARGET_FILE:dprcplugin>
            --input "${CMAKE_CURRENT_SOURCE_DIR}/tests/in.qmmm_xtb_compare_triclinic"
            --mpi-exec "${MPIEXEC_EXECUTABLE}"
            "--mpi-numproc-flag=${MPIEXEC_NUMPROC_FLAG}"
            --mpi-procs 2
            --evidence "${CMAKE_CURRENT_BINARY_DIR}/qmmm-mpi-triclinic-evidence.json")
        add_test(NAME lammps.qmmm_xtb.mpi_tip4p_reference_compare
          COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_lammps_qmmm_compare.py"
            ${_dprc_reference_lammps_arguments}
            --lammps "${DPRC_LAMMPS_EXECUTABLE}"
            --plugin $<TARGET_FILE:dprcplugin>
            --input "${CMAKE_CURRENT_SOURCE_DIR}/tests/in.qmmm_xtb_tip4p_compare"
            --natoms 5
            --reference-kspace pppm/tip4p/xtb
            --actual-kspace pppm/tip4p/dprc
            --mpi-exec "${MPIEXEC_EXECUTABLE}"
            "--mpi-numproc-flag=${MPIEXEC_NUMPROC_FLAG}"
            --mpi-procs 2
            --evidence "${CMAKE_CURRENT_BINARY_DIR}/qmmm-mpi-tip4p-evidence.json")
        add_test(NAME lammps.qmmm_xtb.mpi_tip4p_reference_compare_triclinic
          COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_lammps_qmmm_compare.py"
            ${_dprc_reference_lammps_arguments}
            --lammps "${DPRC_LAMMPS_EXECUTABLE}"
            --plugin $<TARGET_FILE:dprcplugin>
            --input "${CMAKE_CURRENT_SOURCE_DIR}/tests/in.qmmm_xtb_tip4p_compare_triclinic"
            --natoms 5
            --reference-kspace pppm/tip4p/xtb
            --actual-kspace pppm/tip4p/dprc
            --mpi-exec "${MPIEXEC_EXECUTABLE}"
            "--mpi-numproc-flag=${MPIEXEC_NUMPROC_FLAG}"
            --mpi-procs 2
            --evidence "${CMAKE_CURRENT_BINARY_DIR}/qmmm-mpi-tip4p-triclinic-evidence.json")
        set_tests_properties(
          lammps.qmmm_xtb.mpi_reference_compare
          lammps.qmmm_xtb.mpi_reference_compare_triclinic
          lammps.qmmm_xtb.mpi_tip4p_reference_compare
          lammps.qmmm_xtb.mpi_tip4p_reference_compare_triclinic
          PROPERTIES TIMEOUT 240)
      endif()
    endif()
  endif()
endif()
