"""
You need to install : https://github.com/NVIDIA-AI-IOT/torch2trt#option-2---with-plugins-experimental
"""

import argparse

from common.file_presence_checker import terminate_if_already_exported

from backbone.vit import Vit

from audio_descriptor.backbones import Mnasnet0_5, Mnasnet1_0, Resnet18, Resnet34, Resnet50, Resnet101
from audio_descriptor.backbones import OpenFaceInception, ThinResnet34, EcapaTdnn, SmallEcapaTdnn
from audio_descriptor.audio_descriptor_extractor import AudioDescriptorExtractor, AudioDescriptorExtractorVLAD
from audio_descriptor.audio_descriptor_extractor import AudioDescriptorExtractorSAP, AudioDescriptorExtractorPSLA


def main():
    parser = argparse.ArgumentParser(description='Export audio descriptor extractor')
    parser.add_argument('--backbone_type', choices=['mnasnet0.5', 'mnasnet1.0',
                                                    'resnet18', 'resnet34', 'resnet50',
                                                    'open_face_inception', 'thin_resnet_34',
                                                    'ecapa_tdnn_512', 'ecapa_tdnn_1024',
                                                    'small_ecapa_tdnn_128', 'small_ecapa_tdnn_256',
                                                    'small_ecapa_tdnn_512'],
                        help='Choose the backbone type', required=True)
    parser.add_argument('--embedding_size', type=int, help='Set the embedding size', required=True)
    parser.add_argument('--pooling_layer', choices=['avg', 'vlad', 'sap'], help='Set the pooling layer', required=True)
    parser.add_argument('--waveform_size', type=int, help='Set the waveform size', required=True)
    parser.add_argument('--n_features', type=int, help='Set n_features', required=True)
    parser.add_argument('--n_fft', type=int, help='Set n_fft', required=True)

    parser.add_argument('--dataset_class_count', type=int,
                        help='Choose the dataset class count when criterion_type is "cross_entropy_loss" or '
                             '"am_softmax_loss"',
                        default=None)
    parser.add_argument('--normalized_linear', action='store_true',
                        help='Use "NormalizedLinear" instead of "nn.Linear"')
    parser.add_argument('--conv_bias', action='store_true', help='Set bias=True to Conv2d (open_face_inception only)')

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

    image_size = (args.n_features, args.waveform_size // (args.n_fft // 2) + 1)
    model = create_model(args.backbone_type, args.n_features, args.embedding_size, args.dataset_class_count,
                         args.normalized_linear, args.pooling_layer, conv_bias=args.conv_bias)
    x = torch.ones((1, 1, image_size[0], image_size[1]))
    keys_to_remove = ['_classifier._weight'] if args.dataset_class_count is None else []
    export_model(model, args.model_checkpoint, x, args.output_dir, args.torch_script_filename, args.trt_filename,
                 trt_fp16=args.trt_fp16, keys_to_remove=keys_to_remove)


def create_model(backbone_type, n_features, embedding_size,
                 class_count=None, normalized_linear=False, pooling_layer='avg', conv_bias=False):
    pretrained = True
    if backbone_type == 'passt_s_n':
        return Vit((n_features, 1000), embedding_size=embedding_size, class_count=class_count,
                   in_channels=1, depth=12, dropout_rate=0.0, attention_dropout_rate=0.0, output_embeddings=True)
    elif backbone_type == 'passt_s_n_l':
        return Vit((n_features, 1000), embedding_size=embedding_size, class_count=class_count,
                   in_channels=1, depth=7, dropout_rate=0.0, attention_dropout_rate=0.0, output_embeddings=True)

    backbone = create_backbone(backbone_type, n_features, pretrained, conv_bias)
    if pooling_layer == 'avg':
        return AudioDescriptorExtractor(backbone, embedding_size=embedding_size,
                                        class_count=class_count, normalized_linear=normalized_linear)
    elif pooling_layer == 'vlad':
        return AudioDescriptorExtractorVLAD(backbone, embedding_size=embedding_size,
                                            class_count=class_count, normalized_linear=normalized_linear)
    elif pooling_layer == 'sap':
        return AudioDescriptorExtractorSAP(backbone, embedding_size=embedding_size,
                                           class_count=class_count, normalized_linear=normalized_linear)
    elif pooling_layer == 'psla':
        return AudioDescriptorExtractorPSLA(backbone, embedding_size=embedding_size,
                                           class_count=class_count, normalized_linear=normalized_linear)
    else:
        raise ValueError('Invalid pooling layer')


def create_backbone(backbone_type, n_features, pretrained, conv_bias=False):
    if backbone_type == 'mnasnet0.5':
        return Mnasnet0_5(pretrained=pretrained)
    elif backbone_type == 'mnasnet1.0':
        return Mnasnet1_0(pretrained=pretrained)
    elif backbone_type == 'resnet18':
        return Resnet18(pretrained=pretrained)
    elif backbone_type == 'resnet34':
        return Resnet34(pretrained=pretrained)
    elif backbone_type == 'resnet50':
        return Resnet50(pretrained=pretrained)
    elif backbone_type == 'resnet101':
        return Resnet101(pretrained=pretrained)
    elif backbone_type == 'open_face_inception':
        return OpenFaceInception(conv_bias)
    elif backbone_type == 'thin_resnet_34':
        return ThinResnet34()
    elif backbone_type == 'ecapa_tdnn_512':
        return EcapaTdnn(n_features, channels=512)
    elif backbone_type == 'ecapa_tdnn_1024':
        return EcapaTdnn(n_features, channels=1024)
    elif backbone_type == 'small_ecapa_tdnn_128':
        return SmallEcapaTdnn(n_features, channels=128)
    elif backbone_type == 'small_ecapa_tdnn_256':
        return SmallEcapaTdnn(n_features, channels=256)
    elif backbone_type == 'small_ecapa_tdnn_512':
        return SmallEcapaTdnn(n_features, channels=512)
    elif backbone_type == 'small_ecapa_tdnn_1024':
        return SmallEcapaTdnn(n_features, channels=1024)
    else:
        raise ValueError('Invalid backbone type')


if __name__ == '__main__':
    main()
