"""
You need to install : https://github.com/NVIDIA-AI-IOT/torch2trt#option-2---with-plugins-experimental
"""

import argparse

from common.file_presence_checker import terminate_if_already_exported

def main():
    parser = argparse.ArgumentParser(description='Export yolo v4')
    parser.add_argument('--model_type', choices=['yolo_v4', 'yolo_v4_tiny', 'yolo_v7', 'yolo_v7_tiny'],
                        help='Choose the model type', required=True)

    parser.add_argument('--output_dir', type=str, help='Choose the output directory', required=True)
    parser.add_argument('--torch_script_filename', type=str, help='Choose the TorchScript filename', required=True)
    parser.add_argument('--trt_filename', type=str, help='Choose the TensorRT filename', required=True)
    parser.add_argument('--model_checkpoint', type=str, help='Choose the model checkpoint file', required=True)

    parser.add_argument('--trt_fp16', action='store_true', help='Choose the model checkpoint file')

    parser.add_argument('--force_export_if_exists', action='store_true')

    args = parser.parse_args()

    terminate_if_already_exported(args.output_dir, args.torch_script_filename, args.trt_filename, args.force_export_if_exists)

    import torch

    from common.model_exporter import export_model



    model = create_model(args.model_type)
    x = torch.ones((1, 3, model.get_image_size()[0], model.get_image_size()[1]))
    export_model(model, args.model_checkpoint, x, args.output_dir, args.torch_script_filename, args.trt_filename,
                 trt_fp16=args.trt_fp16)


def create_model(model_type):
    from object_detection.modules.yolo_v4 import YoloV4
    from object_detection.modules.yolo_v4_tiny import YoloV4Tiny
    from object_detection.modules.yolo_v7 import YoloV7
    from object_detection.modules.yolo_v7_tiny import YoloV7Tiny

    if model_type == 'yolo_v4':
        model = YoloV4()
    elif model_type == 'yolo_v4_tiny':
        model = YoloV4Tiny()
    elif model_type == 'yolo_v7':
        model = YoloV7()
    elif model_type == 'yolo_v7_tiny':
        model = YoloV7Tiny()
    else:
        raise ValueError('Invalid model type')

    return model


if __name__ == '__main__':
    main()
