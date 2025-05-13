
# deprecated
add_executable (test_performance main.cpp)
target_link_libraries (test_performance PRIVATE 
    vsag eval_obj yaml-cpp simd libhdf5_cpp.a libhdf5.a z
    # -Wl,--whole-archive ${PROJECT_SOURCE_DIR}/src/lib/libpnm_engine_static.a  -Wl,--no-whole-archive 
    # -lz 
    # numa
    # dl)
)


# SET(CMAKE_CXX_FLAGS  "-Ofast -lrt -DNDEBUG -std=c++11 -DHAVE_CXX0X -g -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0" )
SET(CMAKE_CXX_FLAGS  "-w")
add_executable(test_vec_cal_demo
    test_vec_cal_demo.cpp
)

target_link_libraries(test_vec_cal_demo
    # -Wl,--whole-archive ${PROJECT_SOURCE_DIR}/src/lib/libpnm_engine_static.a  -Wl,--no-whole-archive 
    # -lz 
    # numa
    # dl
    vsag
)