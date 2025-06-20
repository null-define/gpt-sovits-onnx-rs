cmake_minimum_required(VERSION 3.14)
set(GPT_SOVITS_CC_NAME gpt-sovits-mnn-cc)

project(${GPT_SOVITS_CC_NAME})

# add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/MNN-3.2.0/)

# add_compile_options(-fsanitize=address)
# add_link_options(-fsanitize=address -static-libasan)

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/MNN-3.2.0//build)
aux_source_directory(./cc SRC)
add_library(${GPT_SOVITS_CC_NAME} ${SRC})
target_include_directories(${GPT_SOVITS_CC_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/MNN-3.2.0/include)
target_link_libraries(${GPT_SOVITS_CC_NAME} PRIVATE MNN)

add_executable(simple_test "./test.cc")

target_link_libraries(simple_test PRIVATE ${GPT_SOVITS_CC_NAME}  MNN)
