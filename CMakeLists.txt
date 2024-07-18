cmake_minimum_required(VERSION 3.1)

project(
  MocapAligner
  LANGUAGES CXX
  VERSION 1.0.0)

include(CTest)
enable_testing()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


# Find mc_rtc if we are not building inside mc_rtc itself
if(NOT TARGET mc_rtc::mc_control)
  find_package(mc_rtc REQUIRED)
endif()

# Configure and install the plugin configuration
configure_file(mc_rtc_plugin/etc/MocapAligner.in.yaml
               "${CMAKE_CURRENT_BINARY_DIR}/etc/MocapAligner.yaml")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/etc/MocapAligner.yaml"
        DESTINATION "${MC_PLUGINS_RUNTIME_INSTALL_PREFIX}/etc")

set(plugin_SRC mc_rtc_plugin/src/MocapAligner.cpp)

set(plugin_HDR mc_rtc_plugin/src/MocapAligner.h)

# The add_plugin macro is provided by mc_rtc
add_plugin(MocapAligner "${plugin_SRC}" "${plugin_HDR}")

if(BUILD_TESTING)
  add_subdirectory(mc_rtc_plugin/tests)
endif()
