if(NOT DEFINED EXPECTED_WORLD_RANKS)
  set(EXPECTED_WORLD_RANKS 1)
endif()

foreach(world RANGE 0 1)
  set(output_file "${OUTPUT_PREFIX}.${world}")
  if(NOT EXISTS "${output_file}")
    message(FATAL_ERROR "dprc/info did not create ${output_file}")
  endif()
  file(READ "${output_file}" output)
  if(NOT output MATCHES "universe_worlds=2" OR
     NOT output MATCHES "world_index=${world}" OR
     NOT output MATCHES "world_ranks=${EXPECTED_WORLD_RANKS}" OR
     NOT output MATCHES "broker_root=1" OR
     NOT output MATCHES "broker_rank=${world}" OR
     NOT output MATCHES "broker_size=2" OR
     NOT output MATCHES "stable_slot=${world}")
    message(FATAL_ERROR "Unexpected partition ${world} output: ${output}")
  endif()
endforeach()
