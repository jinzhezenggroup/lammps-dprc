if(NOT DEFINED NM_EXECUTABLE OR NOT EXISTS "${NM_EXECUTABLE}")
  message(FATAL_ERROR "NM_EXECUTABLE must name the configured symbol inspector")
endif()
if(NOT DEFINED DPRC_PLUGIN OR NOT EXISTS "${DPRC_PLUGIN}")
  message(FATAL_ERROR "DPRC_PLUGIN must name the built LAMMPS-DPRc plugin")
endif()
if(NOT DEFINED DEEPMD_PLUGIN OR NOT EXISTS "${DEEPMD_PLUGIN}")
  message(FATAL_ERROR "DEEPMD_PLUGIN must name the selected DeePMD plugin")
endif()

function(defined_dynamic_symbols library output_variable)
  execute_process(
    COMMAND "${NM_EXECUTABLE}" -D --defined-only --no-demangle -P "${library}"
    RESULT_VARIABLE nm_status
    OUTPUT_VARIABLE nm_output
    ERROR_VARIABLE nm_error)
  if(NOT nm_status EQUAL 0)
    message(FATAL_ERROR "Could not inspect ${library}: ${nm_error}")
  endif()

  string(REPLACE "\n" ";" nm_lines "${nm_output}")
  set(symbols)
  foreach(line IN LISTS nm_lines)
    string(STRIP "${line}" line)
    if(line STREQUAL "")
      continue()
    endif()
    string(REGEX MATCH "^[^ \t]+" symbol "${line}")
    string(REGEX REPLACE "@.*$" "" symbol "${symbol}")
    list(APPEND symbols "${symbol}")
  endforeach()
  list(REMOVE_DUPLICATES symbols)
  list(SORT symbols)
  set(${output_variable} "${symbols}" PARENT_SCOPE)
endfunction()

defined_dynamic_symbols("${DPRC_PLUGIN}" dprc_symbols)
defined_dynamic_symbols("${DEEPMD_PLUGIN}" deepmd_symbols)
set(overlap)
foreach(symbol IN LISTS dprc_symbols)
  if(symbol IN_LIST deepmd_symbols)
    list(APPEND overlap "${symbol}")
  endif()
endforeach()

# Every LAMMPS plugin intentionally exposes this entry point. dlsym() is
# performed on each individual dlopen handle, so the shared name is not a
# style or implementation-symbol collision.
list(REMOVE_ITEM overlap lammpsplugin_init)
if(overlap)
  message(FATAL_ERROR
    "LAMMPS-DPRc and DeePMD plugins define overlapping implementation symbols: ${overlap}")
endif()
