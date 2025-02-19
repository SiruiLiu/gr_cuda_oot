# CMake generated Testfile for 
# Source directory: /home/dayao/Projects/gpu/gr-cuda/python/cuda
# Build directory: /home/dayao/Projects/gpu/gr-cuda/build/python/cuda
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(qa_copy "/usr/bin/sh" "qa_copy_test.sh")
set_tests_properties(qa_copy PROPERTIES  _BACKTRACE_TRIPLES "/usr/lib/x86_64-linux-gnu/cmake/gnuradio/GrTest.cmake;119;add_test;/home/dayao/Projects/gpu/gr-cuda/python/cuda/CMakeLists.txt;34;GR_ADD_TEST;/home/dayao/Projects/gpu/gr-cuda/python/cuda/CMakeLists.txt;0;")
add_test(qa_multiply_const "/usr/bin/sh" "qa_multiply_const_test.sh")
set_tests_properties(qa_multiply_const PROPERTIES  _BACKTRACE_TRIPLES "/usr/lib/x86_64-linux-gnu/cmake/gnuradio/GrTest.cmake;119;add_test;/home/dayao/Projects/gpu/gr-cuda/python/cuda/CMakeLists.txt;35;GR_ADD_TEST;/home/dayao/Projects/gpu/gr-cuda/python/cuda/CMakeLists.txt;0;")
add_test(qa_load "/usr/bin/sh" "qa_load_test.sh")
set_tests_properties(qa_load PROPERTIES  _BACKTRACE_TRIPLES "/usr/lib/x86_64-linux-gnu/cmake/gnuradio/GrTest.cmake;119;add_test;/home/dayao/Projects/gpu/gr-cuda/python/cuda/CMakeLists.txt;45;GR_ADD_TEST;/home/dayao/Projects/gpu/gr-cuda/python/cuda/CMakeLists.txt;0;")
subdirs("bindings")
