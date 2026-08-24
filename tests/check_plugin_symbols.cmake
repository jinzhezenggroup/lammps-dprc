if(NOT DEFINED NM_EXECUTABLE OR NOT EXISTS "${NM_EXECUTABLE}")
  message(FATAL_ERROR "NM_EXECUTABLE must name the configured symbol inspector")
endif()
if(NOT DEFINED PLUGIN_FILE OR NOT EXISTS "${PLUGIN_FILE}")
  message(FATAL_ERROR "PLUGIN_FILE must name the built LAMMPS-DPRc plugin")
endif()

execute_process(
  COMMAND "${NM_EXECUTABLE}" -D --defined-only --no-demangle -P "${PLUGIN_FILE}"
  RESULT_VARIABLE nm_status
  OUTPUT_VARIABLE nm_output
  ERROR_VARIABLE nm_error)
if(NOT nm_status EQUAL 0)
  message(FATAL_ERROR "Could not inspect plugin dynamic symbols: ${nm_error}")
endif()

string(REPLACE "\n" ";" nm_lines "${nm_output}")
set(exported_symbols)
foreach(line IN LISTS nm_lines)
  string(STRIP "${line}" line)
  if(line STREQUAL "")
    continue()
  endif()
  string(REGEX MATCH "^[^ \t]+" symbol "${line}")
  string(REGEX REPLACE "@.*$" "" symbol "${symbol}")
  list(APPEND exported_symbols "${symbol}")
endforeach()
list(REMOVE_DUPLICATES exported_symbols)
list(SORT exported_symbols)

if(NOT exported_symbols STREQUAL "lammpsplugin_init")
  message(FATAL_ERROR
    "dprcplugin must export only lammpsplugin_init; found: ${exported_symbols}")
endif()

# The production target compiles the pinned reference fix directly. Inspect
# local and undefined symbols too so a missed macro rename cannot silently
# bind to, define, or interpose on LAMMPS's original qmmm/xtb implementation.
execute_process(
  COMMAND "${NM_EXECUTABLE}" --no-demangle -P "${PLUGIN_FILE}"
  RESULT_VARIABLE all_nm_status
  OUTPUT_VARIABLE all_nm_output
  ERROR_VARIABLE all_nm_error)
if(NOT all_nm_status EQUAL 0)
  message(FATAL_ERROR "Could not inspect complete plugin symbol table: ${all_nm_error}")
endif()
if(all_nm_output MATCHES
   "FixQMMMXTB|PPPMXTB|PPPMTIP4PXTB|QMMMXTBEwald|lammps_qmmm_xtb_(create|calculate|destroy)")
  message(FATAL_ERROR
    "dprcplugin contains an original LAMMPS QMMM-XTB implementation symbol")
endif()
