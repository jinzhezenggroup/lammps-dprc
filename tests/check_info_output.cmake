if(NOT EXISTS "${OUTPUT_FILE}")
  message(FATAL_ERROR "dprc/info did not create ${OUTPUT_FILE}")
endif()

file(READ "${OUTPUT_FILE}" output)
if(NOT output MATCHES "LAMMPS-DPRC" OR
   NOT output MATCHES "broker_root=1" OR
   NOT output MATCHES "broker_rank=0" OR
   NOT output MATCHES "broker_size=1" OR
   NOT output MATCHES "stable_slot=0")
  message(FATAL_ERROR "Unexpected dprc/info output: ${output}")
endif()
