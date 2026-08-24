if(NOT DEFINED BUILD_DIR OR NOT DEFINED INSTALL_PREFIX OR
   NOT DEFINED PYTHON_EXECUTABLE OR NOT DEFINED CHECK_SCRIPT)
  message(FATAL_ERROR "install payload test arguments are incomplete")
endif()

# This path is fixed below the current build tree by CMakeLists.txt. Removing it
# makes the payload assertion independent of a previous test invocation.
file(REMOVE_RECURSE "${INSTALL_PREFIX}")
execute_process(
  COMMAND "${CMAKE_COMMAND}" --install "${BUILD_DIR}" --prefix "${INSTALL_PREFIX}"
  RESULT_VARIABLE install_status
  OUTPUT_VARIABLE install_output
  ERROR_VARIABLE install_error)
if(NOT install_status EQUAL 0)
  message(FATAL_ERROR
    "test install failed (${install_status}):\n${install_output}\n${install_error}")
endif()

execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" "${CHECK_SCRIPT}"
    --install-prefix "${INSTALL_PREFIX}"
  RESULT_VARIABLE check_status
  OUTPUT_VARIABLE check_output
  ERROR_VARIABLE check_error)
if(NOT check_status EQUAL 0)
  message(FATAL_ERROR
    "installed payload check failed (${check_status}):\n${check_output}\n${check_error}")
endif()
