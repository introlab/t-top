cmake_minimum_required(VERSION 3.5)
project(dnn_utils)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)

# Python Librairies
ament_python_install_package(${PROJECT_NAME})
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../../tools/dnn_training DESTINATION ${PYTHON_INSTALL_DIR})


# Python Nodes
install(PROGRAMS
  scripts/test.py
  DESTINATION lib/${PROJECT_NAME}
)

# Download Weights
set(weights_archive_name Weights.zip)
set(weights_version "4.1.0")
set(weights_md5 "c20518fa511799032e9f2038434fb409")
set(weights_full_url "https://github.com/introlab/t-top/releases/download/DNN_Weights_v${weights_version}/${weights_archive_name}")

set(weights_output_directory ${CMAKE_CURRENT_SOURCE_DIR}/weights)
set(models_output_directory ${CMAKE_CURRENT_SOURCE_DIR}/models)
set(export_script_directory ${CMAKE_CURRENT_SOURCE_DIR}/../../../tools/dnn_training)
set(weights_output_path  ${weights_output_directory}/${weights_archive_name})

# Calculate MD5
if(EXISTS ${weights_output_path})
  file(MD5 ${weights_output_path} FILE_MD5)
else()
  set (FILE_MD5 "")
endif()

set(FORCE_EXPORT "")

if(NOT EXISTS ${weights_output_path} OR NOT FILE_MD5 STREQUAL ${weights_md5})
    # Download archive
    file(DOWNLOAD
        ${weights_full_url}
        ${weights_output_path}
        EXPECTED_MD5 ${weights_md5}
        TIMEOUT 300 #seconds
        TLS_VERIFY ON
        SHOW_PROGRESS
        STATUS DOWNLOAD_STATUS)

    message(STATUS "Download status: ${DOWNLOAD_STATUS}")

    # Handle download failure
    if(NOT DOWNLOAD_STATUS EQUAL 0)
      message(FATAL_ERROR "Download failed: ${DOWNLOAD_STATUS}")
    endif()

    # Extract
    file(ARCHIVE_EXTRACT
          INPUT ${weights_output_path}
          DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/weights)

    set(FORCE_EXPORT "--force_export_if_exists")
endif()

function(generate_models_filenames MY_MODEL_LIST OUTPUT_VAR)
    set(TEMP_LIST "")  # Temporary list
    foreach(ITEM IN LISTS MY_MODEL_LIST)
      list(APPEND TEMP_LIST "${models_output_directory}/${ITEM}.ts.pth")
      # Add trt file on Jetson platforms
      if (EXISTS "/etc/nv_tegra_release")
        list(APPEND TEMP_LIST "${models_output_directory}/${ITEM}.trt.pth")
      endif()
    endforeach()
    set(${OUTPUT_VAR} ${TEMP_LIST} PARENT_SCOPE)
endfunction()

set(first_batch_models
  descriptor_yolo_v4_tiny_coco
  descriptor_yolo_v7_coco
  yolo_v4_coco
  yolo_v4_tiny_coco
)

set(second_batch_models
  yolo_v7_coco
  yolo_v7_tiny_coco
  yolo_v7_objects365
  pose_estimator_efficientnet_b0
  face_descriptor_open_face_e256
  semantic_segmentation_network_coco
  semantic_segmentation_network_kitchen_open_images
  semantic_segmentation_network_person_other_open_images
  ttop_keyword_spotter
  voice_descriptor_extractor
  multiclass_audio_descriptor_extractor
)

generate_models_filenames(first_batch_models first_batch_models_files)
generate_models_filenames(second_batch_models second_batch_models_files)

set(all_models_output ${first_batch_models_files} ${second_batch_models_files})

# First batch
add_custom_command(
  OUTPUT ${first_batch_models_files}
  COMMAND python3 ${export_script_directory}/export_descriptor_yolo.py --dataset_type coco --model_type yolo_v4_tiny --descriptor_size 128 --output_dir ${models_output_directory} --torch_script_filename descriptor_yolo_v4_tiny_coco.ts.pth --trt_filename descriptor_yolo_v4_tiny_coco.trt.pth --model_checkpoint ${weights_output_directory}/descriptor_yolo_v4_tiny_coco.pth --trt_fp16 ${FORCE_EXPORT}
  COMMAND python3 ${export_script_directory}/export_descriptor_yolo.py --dataset_type coco --model_type yolo_v7 --descriptor_size 128 --output_dir ${models_output_directory} --torch_script_filename descriptor_yolo_v7_coco.ts.pth --trt_filename descriptor_yolo_v7_coco.trt.pth --model_checkpoint ${weights_output_directory}/descriptor_yolo_v7_coco.pth --trt_fp16 ${FORCE_EXPORT}
  COMMAND python3 ${export_script_directory}/export_yolo.py --dataset_type coco --model_type yolo_v4 --output_dir ${models_output_directory} --torch_script_filename yolo_v4_coco.ts.pth --trt_filename yolo_v4_coco.trt.pth --model_checkpoint ${weights_output_directory}/yolo_v4_coco.pth --trt_fp16 ${FORCE_EXPORT}
  COMMAND python3 ${export_script_directory}/export_yolo.py --dataset_type coco --model_type yolo_v4_tiny --output_dir ${models_output_directory} --torch_script_filename yolo_v4_tiny_coco.ts.pth --trt_filename yolo_v4_tiny_coco.trt.pth --model_checkpoint ${weights_output_directory}/yolo_v4_tiny_coco.pth --trt_fp16 ${FORCE_EXPORT}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

# Second Batch
add_custom_command(
  OUTPUT ${second_batch_models_files}
  COMMAND python3 ${export_script_directory}/export_yolo.py --dataset_type coco --model_type yolo_v7 --output_dir ${models_output_directory} --torch_script_filename yolo_v7_coco.ts.pth --trt_filename yolo_v7_coco.trt.pth --model_checkpoint ${weights_output_directory}/yolo_v7_coco.pth --trt_fp16 ${FORCE_EXPORT}
  COMMAND python3 ${export_script_directory}/export_yolo.py --dataset_type coco --model_type yolo_v7_tiny --output_dir ${models_output_directory} --torch_script_filename yolo_v7_tiny_coco.ts.pth --trt_filename yolo_v7_tiny_coco.trt.pth --model_checkpoint ${weights_output_directory}/yolo_v7_tiny_coco.pth --trt_fp16 ${FORCE_EXPORT}
  COMMAND python3 ${export_script_directory}/export_yolo.py --dataset_type objects365 --model_type yolo_v7 --output_dir ${models_output_directory} --torch_script_filename yolo_v7_objects365.ts.pth --trt_filename yolo_v7_objects365.trt.pth --model_checkpoint ${weights_output_directory}/yolo_v7_objects365.pth --trt_fp16 ${FORCE_EXPORT}
  COMMAND python3 ${export_script_directory}/export_pose_estimator.py --backbone_type efficientnet_b0 --output_dir ${models_output_directory} --torch_script_filename pose_estimator_efficientnet_b0.ts.pth --trt_filename pose_estimator_efficientnet_b0.trt.pth --model_checkpoint ${weights_output_directory}/pose_estimator_efficientnet_b0.pth --trt_fp16 ${FORCE_EXPORT}
  COMMAND python3 ${export_script_directory}/export_face_descriptor_extractor.py --backbone_type open_face --embedding_size 256 --output_dir ${models_output_directory} --torch_script_filename face_descriptor_open_face_e256.ts.pth --trt_filename face_descriptor_open_face_e256.trt.pth --model_checkpoint ${weights_output_directory}/face_descriptor_open_face_e256.pth --trt_fp16 ${FORCE_EXPORT}
  COMMAND python3 ${export_script_directory}/export_semantic_segmentation_network.py --dataset_type coco --backbone_type stdc1 --channel_scale 1 --output_dir ${models_output_directory} --torch_script_filename semantic_segmentation_network_coco.ts.pth --trt_filename semantic_segmentation_network_coco.trt.pth --model_checkpoint ${weights_output_directory}/semantic_segmentation_network_coco_stdc1_s1.pth --trt_fp16 ${FORCE_EXPORT}
  COMMAND python3 ${export_script_directory}/export_semantic_segmentation_network.py --dataset_type kitchen_open_images --backbone_type stdc1 --channel_scale 1 --output_dir ${models_output_directory} --torch_script_filename semantic_segmentation_network_kitchen_open_images.ts.pth --trt_filename semantic_segmentation_network_kitchen_open_images.trt.pth --model_checkpoint ${weights_output_directory}/semantic_segmentation_network_kitchen_open_images_stdc1_s1.pth --trt_fp16 ${FORCE_EXPORT}
  COMMAND python3 ${export_script_directory}/export_semantic_segmentation_network.py --dataset_type person_other_open_images --backbone_type stdc1 --channel_scale 1 --output_dir ${models_output_directory} --torch_script_filename semantic_segmentation_network_person_other_open_images.ts.pth --trt_filename semantic_segmentation_network_person_other_open_images.trt.pth --model_checkpoint ${weights_output_directory}/semantic_segmentation_network_person_other_open_images_stdc1_s1.pth --trt_fp16 ${FORCE_EXPORT}
  COMMAND python3 ${export_script_directory}/export_keyword_spotter.py --dataset_type ttop_keyword --mfcc_feature_count 20 --output_dir ${models_output_directory} --torch_script_filename ttop_keyword_spotter.ts.pth --trt_filename ttop_keyword_spotter.trt.pth --model_checkpoint ${weights_output_directory}/ttop_keyword_spotter_no_conv_bias_mfcc_20.pth --trt_fp16 ${FORCE_EXPORT}
  COMMAND python3 ${export_script_directory}/export_audio_descriptor_extractor.py --backbone_type open_face_inception --embedding_size 128 --conv_bias --pooling_layer avg --waveform_size 64000 --n_features 128 --n_fft 400 --dataset_class_count 200 --output_dir ${models_output_directory} --torch_script_filename multiclass_audio_descriptor_extractor.ts.pth --trt_filename multiclass_audio_descriptor_extractor.trt.pth --model_checkpoint ${weights_output_directory}/multiclass_audio_descriptor_extractor_open_face_inception_128_64000_128_400_mel_spectrogram.pth --trt_fp16 ${FORCE_EXPORT}
  COMMAND python3 ${export_script_directory}/export_audio_descriptor_extractor.py --backbone_type small_ecapa_tdnn_512 --embedding_size 256 --pooling_layer avg --waveform_size 63840 --n_features 96 --n_fft 480 --output_dir ${models_output_directory} --torch_script_filename voice_descriptor_extractor.ts.pth --trt_filename voice_descriptor_extractor.trt.pth --model_checkpoint ${weights_output_directory}/voice_descriptor_extractor_small_ecapa_tdnn_512_256_64000_96_480_mel_spectrogram.pth --trt_fp16 ${FORCE_EXPORT}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

add_custom_target(
  dnn_utils_models_extractor ALL
  DEPENDS ${all_models_output}
)

install(DIRECTORY models DESTINATION share/${PROJECT_NAME})

ament_package()
