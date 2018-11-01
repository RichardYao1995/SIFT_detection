cmake_minimum_required(VERSION 2.8.3)
project(image_construction)
add_compile_options(-std=c++11)

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(Glew REQUIRED)

set(OpenCV_DIR "/home/yao/Environment/opencv-3.2.0/build")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

add_executable(image_construction src/main.cpp)
target_link_libraries(image_construction ${catkin_LIBRARIES}
					 ${OpenCV_LIBS})
