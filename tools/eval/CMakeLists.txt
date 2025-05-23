
set (eval_srcs
        case/eval_case.cpp
        case/search_eval_case.cpp
        case/build_eval_case.cpp

        exporter/exporter.cpp
        exporter/formatter.cpp

        monitor/monitor.cpp
        monitor/latency_monitor.cpp
        monitor/recall_monitor.cpp
        monitor/memory_peak_monitor.cpp
        monitor/duration_monitor.cpp

        eval_config.cpp
        eval_dataset.cpp
        eval_job.cpp
        )
add_library (eval_obj OBJECT ${eval_srcs})
target_compile_options (eval_obj PRIVATE -fopenmp)
target_link_libraries (eval_obj PRIVATE cpr)
add_dependencies (eval_obj hdf5 spdlog yaml-cpp tabulate cpr)

add_executable (eval_performance main.cpp)
target_compile_options (eval_performance PRIVATE -fopenmp)
target_link_libraries (eval_performance
  PRIVATE
  vsag
  eval_obj
  yaml-cpp
  argparse::argparse
  tabulate
  simd
  libhdf5_cpp.a
  libhdf5.a
  z
)
