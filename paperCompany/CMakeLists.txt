cmake_minimum_required(VERSION 3.16)
project(PaperDefectDetector)

#set(CMAKE_CXX_COMPILER gcc)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS -Werror=return-type)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
#set(CMAKE_CXX_FLAGS "-Wall")


find_package(PkgConfig REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(GenICamService REQUIRED)
find_package(TensorRT REQUIRED)
find_package(spdlog REQUIRED)
find_package(CUDA REQUIRED)


set(DETECTOR_PATH
        "stream/detectorGPU/config.h"
        "stream/detectorGPU/Detector.cpp"
        "stream/detectorGPU/Detector.h"
        "stream/detectorGPU/engine.cpp"
        "stream/detectorGPU/engine.h"
)
add_compile_definitions(GPU)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_definitions(DEBUG)
endif ()
if (CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_definitions(RELEASE)
endif ()
set(GPU_LIBS ${CUDA_LIBRARIES} ${TensorRT_LIBRARIES})

#set(TYPE "GPU")
#set(DETECTOR_PATH "")
#if (TYPE STREQUAL "CPU")
#set(DETECTOR_PATH
#        "stream/detector/config.h"
#        "stream/detector/Detector.cpp"
#        "stream/detector/Detector.h")
#    add_compile_definitions(CPU)
#else ()
#    find_package(TensorRT REQUIRED)
#    find_package(CUDA REQUIRED)
#    set(DETECTOR_PATH
#            "stream/detectorGPU/config.h"
#            "stream/detectorGPU/Detector.cpp"
#            "stream/detectorGPU/Detector.h"
#            "stream/detectorGPU/engine.cpp"
#            "stream/detectorGPU/engine.h"
#            )
#    add_compile_definitions(GPU)
#
#    set(GPU_LIBS ${CUDA_LIBRARIES} ${TensorRT_LIBRARIES})
#endif ()

include_directories(
        ${CUDA_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        #        ${HTTPLIB_INCLUDE_DIRS}
        #        ${SQLiteCpp_INCLUDE_DIRS}
        ${TensorRT_INCLUDE_DIRS})

#message("DETECTOR PATH: " ${DETECTOR_PATH})
add_executable(ai
        detectorGPU/Detector.cpp
        detectorGPU/config.h
        detectorGPU/Detector.h
        detectorGPU/engine.cpp
        detectorGPU/engine.h
        #        stream/Device/orin.c
        #        stream/Device/jetgpio.h
        main.cpp
        #        stream/request/page-organization/getPageOrganization.cpp
        #        stream/request/camera-settings/getCameraSettings.cpp
        Manager/Detector/CDetectorManager.cpp
        Device/CDevice.cpp
        Manager/Image/CImageManager.cpp
        Network/CNetwork.cpp
        Network/CNetwork.h
        Image/CImage.h
        Device/CDeviceStatus.cpp
        Device/CDeviceStatus.h
        Parameters/CDeviceParams.cpp
        Parameters/CDeviceParams.h
        RingBuffer/CRingBuffer.h
        IWriter/CWriter.cpp
        IWriter/CWriter.h
        constant/constant.h
        colorSensor/uart.cpp)

target_link_libraries(ai
        libMVSDK.so
        ${OpenCV_LIBRARIES}
        ${GPU_LIBS}
        GenICamService::GenICamService
        #        JetsonGPIO
        pthread
        sockpp)
