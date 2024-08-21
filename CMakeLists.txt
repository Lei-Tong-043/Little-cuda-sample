cmake_minimum_required(VERSION 3.10)
project(MyProject VERSION 1.0 LANGUAGES CXX)

# 设置C/C++编译器
# icx && icpx
set(CMAKE_C_COMPILER "/online1/public/support/intel/intel_compiler/2024u1/compiler/2024.1/bin/icx")
set(CMAKE_CXX_COMPILER "/online1/public/support/intel/intel_compiler/2024u1/compiler/2024.1/bin/icpx")

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 添加包含目录
include_directories(include lib)

# 添加子目录
add_subdirectory(lib)

# 添加可执行文件
add_executable(MyProject src/main/main.cpp src/main/helper.cpp)

# 链接库文件
target_link_libraries(MyProject my_lib)
