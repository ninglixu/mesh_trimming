cmake_minimum_required (VERSION 3.1...3.15)
project (Model_Trimming)
set (targetName Trimming)

add_executable (${targetName} trimming.cpp)
include_directories(E:\\3rd\\OpenMesh\ 10.0\\include ./include)
find_package(Eigen3 CONFIG REQUIRED)
target_link_libraries(${targetName} PRIVATE Eigen3::Eigen)

set (OpenMesh_Libs debug "E:\\3rd\\OpenMesh 10.0\\lib\\OpenMeshCored.lib" debug "E:\\3rd\\OpenMesh 10.0\\lib\\OpenMeshToolsd.lib" 
					 optimized "E:\\3rd\\OpenMesh 10.0\\lib\\OpenMeshCore.lib" optimized "E:\\3rd\\OpenMesh 10.0\\lib\\OpenMeshTools.lib")

target_link_libraries( ${targetName} PRIVATE ${OpenMesh_Libs})
# Boost and its components
find_package( Boost)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})
target_link_libraries(${targetName} PRIVATE ${Boost_LIBRARIES})
if ( NOT Boost_FOUND )

  message(STATUS "This project requires the Boost library, and will not be compiled.")

  return()  

endif()

