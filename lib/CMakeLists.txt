# 在 lib 目录下的 CMakeLists.txt 文件
add_library(my_lib STATIC my_lib.cpp)
target_include_directories(my_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
