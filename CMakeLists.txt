cmake_minimum_required(VERSION 3.10)

project(rknn_yolov8_pose_demo)

if (ENABLE_ASAN)
	message(STATUS "BUILD WITH ADDRESS SANITIZER")
	set (CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
	set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
	set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
endif ()

set(rknpu_yolov8-pose_file src/utils/yolov8-pose.cc)

#  设置库架构
set(LIB_ARCH "aarch64")
set(DEVICE_NAME "RK3588")

#  rknn_api 文件夹路径
set(RKNN_API_PATH ${CMAKE_CURRENT_SOURCE_DIR}/librknn_api)
#  rknn_api include 路径
set(RKNN_API_INCLUDE_PATH ${RKNN_API_PATH}/include)
#  rknn_api lib 路径
set(RKNN_API_LIB_PATH ${RKNN_API_PATH}/${LIB_ARCH}/librknnrt.so)
#  yolo_api include 路径
#set(YOLO_API_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/include)
# util_lib
#set(YOLO_API_LIB_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lib)

# 寻找OpenCV库，使用自定义的OpenCV_DIR
find_package(OpenCV REQUIRED)
# 输出OpenCV信息
message(STATUS "include path: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "${OpenCV_VERSION}")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
link_directories( ${CMAKE_CURRENT_SOURCE_DIR}/lib)

# 寻找Eigen3
#set(EIGEN_VERSION 3.3)
#find_package(Eigen3 ${EIGEN_VERSION} QUIET)

# 用来搜索头文件的目录
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${RKNN_API_INCLUDE_PATH}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils
    ${YOLO_API_INCLUDE_PATH}
)



# 测试yolov8_video_track
add_executable(yolov8_video_pose 
src/yolov8_video_pose.cc
src/utils/yolov8-pose.cc
src/utils/file_utils.c
src/utils/image_drawing.c
src/utils/image_utils.c
src/utils/postprocess.cc

)


target_link_libraries(yolov8_video_pose PRIVATE
 m
  rknnrt
  turbojpeg
  rga
  )
