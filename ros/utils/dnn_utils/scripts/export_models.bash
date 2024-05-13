#!/bin/bash

SCRIPT=`realpath $0`
SCRIPT_PATH=`dirname $SCRIPT`

mkdir -p $SCRIPT_PATH/../weights
mkdir -p $SCRIPT_PATH/../models

# Download weights
cd $SCRIPT_PATH/../weights

OLD_TIME=0
if [ -f Weights.zip ]; then
    OLD_TIME=$(stat Weights.zip -c %Y)
fi

if OUT=$(wget -N https://github.com/introlab/t-top/releases/download/DNN_Weights_v4.1.0/Weights.zip 2>&1); then
    # Output to stdout on success
    echo $OUT
else
    # Output to stderr on failure, and fail with non-zero exit code
    RETVAL=$?
    echo $OUT 1>&2
    exit $RETVAL
fi

NEW_TIME=$(stat Weights.zip -c %Y)

if [ $NEW_TIME -gt $OLD_TIME ]; then
    echo "Exporting new weights."
    unzip -o Weights.zip
    FORCE_EXPORT='--force_export_if_exists'
else
    echo "Using current weights."
    FORCE_EXPORT=''
fi

# Export models
cd $SCRIPT_PATH/../../../../tools/dnn_training

# Kill child processes on exit
trap 'jobs -p | xargs -I '{}' kill '{}' &> /dev/null; wait' INT QUIT KILL TERM

set -e

python3 export_descriptor_yolo.py --dataset_type coco --model_type yolo_v4_tiny --descriptor_size 128 --output_dir $SCRIPT_PATH/../models --torch_script_filename descriptor_yolo_v4_tiny_coco.ts.pth --trt_filename descriptor_yolo_v4_tiny_coco.trt.pth --model_checkpoint $SCRIPT_PATH/../weights/descriptor_yolo_v4_tiny_coco.pth --trt_fp16 $FORCE_EXPORT
python3 export_descriptor_yolo.py --dataset_type coco --model_type yolo_v7 --descriptor_size 128 --output_dir $SCRIPT_PATH/../models --torch_script_filename descriptor_yolo_v7_coco.ts.pth --trt_filename descriptor_yolo_v7_coco.trt.pth --model_checkpoint $SCRIPT_PATH/../weights/descriptor_yolo_v7_coco.pth --trt_fp16 $FORCE_EXPORT

python3 export_yolo.py --dataset_type coco --model_type yolo_v4 --output_dir $SCRIPT_PATH/../models --torch_script_filename yolo_v4_coco.ts.pth --trt_filename yolo_v4_coco.trt.pth --model_checkpoint $SCRIPT_PATH/../weights/yolo_v4_coco.pth --trt_fp16 $FORCE_EXPORT
python3 export_yolo.py --dataset_type coco --model_type yolo_v4_tiny --output_dir $SCRIPT_PATH/../models --torch_script_filename yolo_v4_tiny_coco.ts.pth --trt_filename yolo_v4_tiny_coco.trt.pth --model_checkpoint $SCRIPT_PATH/../weights/yolo_v4_tiny_coco.pth --trt_fp16 $FORCE_EXPORT
python3 export_yolo.py --dataset_type coco --model_type yolo_v7 --output_dir $SCRIPT_PATH/../models --torch_script_filename yolo_v7_coco.ts.pth --trt_filename yolo_v7_coco.trt.pth --model_checkpoint $SCRIPT_PATH/../weights/yolo_v7_coco.pth --trt_fp16 $FORCE_EXPORT
python3 export_yolo.py --dataset_type coco --model_type yolo_v7_tiny --output_dir $SCRIPT_PATH/../models --torch_script_filename yolo_v7_tiny_coco.ts.pth --trt_filename yolo_v7_tiny_coco.trt.pth --model_checkpoint $SCRIPT_PATH/../weights/yolo_v7_tiny_coco.pth --trt_fp16 $FORCE_EXPORT

python3 export_yolo.py --dataset_type objects365 --model_type yolo_v7 --output_dir $SCRIPT_PATH/../models --torch_script_filename yolo_v7_objects365.ts.pth --trt_filename yolo_v7_objects365.trt.pth --model_checkpoint $SCRIPT_PATH/../weights/yolo_v7_objects365.pth --trt_fp16 $FORCE_EXPORT

python3 export_pose_estimator.py --backbone_type efficientnet_b0 --output_dir $SCRIPT_PATH/../models --torch_script_filename pose_estimator_efficientnet_b0.ts.pth --trt_filename pose_estimator_efficientnet_b0.trt.pth --model_checkpoint $SCRIPT_PATH/../weights/pose_estimator_efficientnet_b0.pth --trt_fp16 $FORCE_EXPORT

python3 export_face_descriptor_extractor.py --backbone_type open_face --embedding_size 256 --output_dir $SCRIPT_PATH/../models --torch_script_filename face_descriptor_open_face_e256.ts.pth --trt_filename face_descriptor_open_face_e256.trt.pth --model_checkpoint $SCRIPT_PATH/../weights/face_descriptor_open_face_e256.pth --trt_fp16 $FORCE_EXPORT

python3 export_semantic_segmentation_network.py --dataset_type coco --backbone_type stdc1 --channel_scale 1 --output_dir $SCRIPT_PATH/../models --torch_script_filename semantic_segmentation_network_coco.ts.pth --trt_filename semantic_segmentation_network_coco.trt.pth --model_checkpoint $SCRIPT_PATH/../weights/semantic_segmentation_network_coco_stdc1_s1.pth --trt_fp16 $FORCE_EXPORT
python3 export_semantic_segmentation_network.py --dataset_type kitchen_open_images --backbone_type stdc1 --channel_scale 1 --output_dir $SCRIPT_PATH/../models --torch_script_filename semantic_segmentation_network_kitchen_open_images.ts.pth --trt_filename semantic_segmentation_network_kitchen_open_images.trt.pth --model_checkpoint $SCRIPT_PATH/../weights/semantic_segmentation_network_kitchen_open_images_stdc1_s1.pth --trt_fp16 $FORCE_EXPORT
python3 export_semantic_segmentation_network.py --dataset_type person_other_open_images --backbone_type stdc1 --channel_scale 1 --output_dir $SCRIPT_PATH/../models --torch_script_filename semantic_segmentation_network_person_other_open_images.ts.pth --trt_filename semantic_segmentation_network_person_other_open_images.trt.pth --model_checkpoint $SCRIPT_PATH/../weights/semantic_segmentation_network_person_other_open_images_stdc1_s1.pth --trt_fp16 $FORCE_EXPORT

python3 export_keyword_spotter.py --dataset_type ttop_keyword --mfcc_feature_count 20 --output_dir $SCRIPT_PATH/../models --torch_script_filename ttop_keyword_spotter.ts.pth --trt_filename ttop_keyword_spotter.trt.pth --model_checkpoint $SCRIPT_PATH/../weights/ttop_keyword_spotter_no_conv_bias_mfcc_20.pth --trt_fp16 $FORCE_EXPORT

# TODO update multiclass_audio_descriptor_extractor
python3 export_audio_descriptor_extractor.py --backbone_type open_face_inception --embedding_size 128 --conv_bias --pooling_layer avg --waveform_size 64000 --n_features 128 --n_fft 400 --dataset_class_count 200 --output_dir $SCRIPT_PATH/../models --torch_script_filename multiclass_audio_descriptor_extractor.ts.pth --trt_filename multiclass_audio_descriptor_extractor.trt.pth --model_checkpoint $SCRIPT_PATH/../weights/multiclass_audio_descriptor_extractor_open_face_inception_128_64000_128_400_mel_spectrogram.pth --trt_fp16 $FORCE_EXPORT
python3 export_audio_descriptor_extractor.py --backbone_type small_ecapa_tdnn_512 --embedding_size 256 --pooling_layer avg --waveform_size 63840 --n_features 96 --n_fft 480 --output_dir $SCRIPT_PATH/../models --torch_script_filename voice_descriptor_extractor.ts.pth --trt_filename voice_descriptor_extractor.trt.pth --model_checkpoint $SCRIPT_PATH/../weights/voice_descriptor_extractor_small_ecapa_tdnn_512_256_64000_96_480_mel_spectrogram.pth --trt_fp16 $FORCE_EXPORT
