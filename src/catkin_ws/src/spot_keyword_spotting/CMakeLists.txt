cmake_minimum_required(VERSION 3.0.2)
project(spot_keyword_spotting)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  rospy
  roscpp
  std_msgs
)

find_package(pybind11 REQUIRED)

add_message_files(
  FILES
  VoiceCommand.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime rospy roscpp std_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  keyword_spotting/native/include
)

pybind11_add_module(kws_native
  keyword_spotting/native/src/bindings.cpp
  keyword_spotting/native/src/preprocess.cpp
)

add_dependencies(kws_native ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(kws_native PRIVATE ${catkin_LIBRARIES})
set_target_properties(kws_native PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_BIN_DESTINATION}
)

add_executable(i2s_window_node
  src/i2s_window_node.cpp
)
add_dependencies(i2s_window_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(i2s_window_node ${catkin_LIBRARIES} ${CMAKE_DL_LIBS})
catkin_install_python(PROGRAMS
  scripts/i2s_voice_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(TARGETS i2s_window_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(TARGETS kws_native
  LIBRARY DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY keyword_spotting/ config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
