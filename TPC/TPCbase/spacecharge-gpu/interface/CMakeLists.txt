set(SRCS
    ${SCGPUDIR}/interface/AliTPCPoissonSolverMem.cxx
#    ${CAGPU_O}
   )

# Headers from sources
set(CINTHDRS
 ${SCGPUDIR}/interface/AliTPCPoissonSolverMem.h
 ${SCGPUDIR}/interface/AliCudaLinkDef.h
    )
  
set(HDRS
  )

# Generate the dictionary
# It will create G_ARG1.cxx and G_ARG1.h / ARG1 = function first argument
get_directory_property(incdirs INCLUDE_DIRECTORIES)
generate_dictionary("${MODULE}" "" "${CINTHDRS}" "${incdirs}")

# Generate the ROOT map
# Dependecies
generate_rootmap("${MODULE}" "" "")

# Add a library to the project using the specified source files
add_library_tested(${MODULE} SHARED ${SRCS} G__${MODULE}.cxx)
#CUDA run-time and driver
target_link_libraries(${MODULE} ${CUDA_LIBRARIES} ${LIBCUDA_SO_PATH} AliTPCSpaceCharge3DDriftLine)

# Additional compilation flags
set_target_properties(${MODULE} PROPERTIES COMPILE_FLAGS "")

# Installation
install(TARGETS ${MODULE}
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib)

install(FILES ${HDRS} DESTINATION include)
