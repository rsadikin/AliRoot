CUDA_ADD_LIBRARY(PoissonSolver3DGPU
  PoissonSolver3DGPU.cu
  # lib.cu
  SHARED
  # STATIC
  #OPTIONS -DSTUFF="blah blah"
  #RELEASE --use_fast_math -DNDEBUG
  #DEBUG -g -DDEBUG
  )


INSTALL(FILES PoissonSolver3DGPU.h 
  DESTINATION include)
INSTALL(FILES "${CMAKE_BINARY_DIR}/libPoissonSolver3DGPU.so" 
  DESTINATION lib)
