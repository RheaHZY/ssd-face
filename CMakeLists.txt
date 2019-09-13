cmake_minimum_required(VERSION 3.6)
set(ProjName ssd-face)
project(${ProjName})

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_STANDARD 11)
find_package(OpenCV REQUIRED)
find_package(Caffe  REQUIRED)
include_directories(${Caffe_INCLUDE_DIRS})
AUX_SOURCE_DIRECTORY(src SRC_FILES)
add_executable(${ProjName} ${SRC_FILES})

target_link_libraries(${ProjName} ${OpenCV_LIBS} )
target_link_libraries(${ProjName} ${Caffe_LIBRARIES})