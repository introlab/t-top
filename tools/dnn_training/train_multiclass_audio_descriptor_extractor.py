import argparse
import os

import torch

from common.program_arguments import save_arguments, print_arguments

from audio_descriptor.backbones import Mnasnet0_5, Mnasnet1_0, Resnet18, Resnet34, Resnet50, OpenFaceInception, VGGLike
from audio_descriptor.backbones import TinyCnn
from audio_descriptor.audio_descriptor_extractor import AudioDescriptorExtractor
from audio_descriptor.trainers import MulticlassAudioDescriptorExtractorTrainer


def main():
    parser = argparse.ArgumentParser(description='Train Backbone')
    parser.add_argument('--use_gpu', action='store_true', help='Use the GPU')
    parser.add_argument('--dataset_root', type=str, help='Choose the dataset root path (FSD50k)', required=True)
    parser.add_argument('--output_path', type=str, help='Choose the output path', required=True)
    parser.add_argument('--backbone_type', choices=['mnasnet0.5', 'mnasnet1.0',
                                                    'resnet18', 'resnet34', 'resnet50',
                                                    'open_face_inception', 'tiny_cnn', 'vgg_like'],
                        help='Choose the backbone type', required=True)
    parser.add_argument('--embedding_size', type=int, help='Set the embedding size', required=True)
    parser.add_argument('--waveform_size', type=int, help='Set the waveform size', required=True)
    parser.add_argument('--n_features', type=int, help='Set n_features', required=True)
    parser.add_argument('--n_fft', type=int, help='Set n_fft', required=True)
    parser.add_argument('--audio_transform_type', choices=['mfcc', 'mel_spectrogram', 'spectrogram'],
                        help='Choose the audio transform type', required=True)
    parser.add_argument('--enable_pitch_shifting', action='store_true', help='Use pitch shifting data augmentation')
    parser.add_argument('--enable_time_stretching', action='store_true', help='Use pitch shifting data augmentation')

    parser.add_argument('--learning_rate', type=float, help='Choose the learning rate', required=True)
    parser.add_argument('--weight_decay', type=float, help='Choose the weight decay', required=True)
    parser.add_argument('--batch_size', type=int, help='Set the batch size for the training', required=True)
    parser.add_argument('--epoch_count', type=int, help='Choose the epoch count', required=True)
    parser.add_argument('--criterion_type', choices=['bce_loss', 'sigmoid_focal_loss'],
                        help='Choose the criterion type', required=True)

    parser.add_argument('--model_checkpoint', type=str, help='Choose the model checkpoint file', default=None)

    args = parser.parse_args()

    model = create_model(args.backbone_type, args.embedding_size)
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')

    output_path = os.path.join(args.output_path, args.backbone_type + '_e' + str(args.embedding_size) +
                               '_' + args.audio_transform_type + '_' + args.criterion_type +
                               '_lr' + str(args.learning_rate) + '_wd' + str(args.weight_decay))
    save_arguments(output_path, args)
    print_arguments(args)

    trainer = MulticlassAudioDescriptorExtractorTrainer(device, model,
                                                        epoch_count=args.epoch_count,
                                                        learning_rate=args.learning_rate,
                                                        weight_decay=args.weight_decay,
                                                        dataset_root=args.dataset_root,
                                                        output_path=output_path,
                                                        batch_size=args.batch_size,
                                                        waveform_size=args.waveform_size,
                                                        n_features=args.n_features,
                                                        n_fft=args.n_fft,
                                                        audio_transform_type=args.audio_transform_type,
                                                        enable_pitch_shifting=args.enable_pitch_shifting,
                                                        enable_time_stretching=args.enable_time_stretching,
                                                        model_checkpoint=args.model_checkpoint)
    trainer.train()


def create_model(backbone_type, embedding_size):
    pretrained = True

    backbone = create_backbone(backbone_type, pretrained)
    return AudioDescriptorExtractor(backbone, embedding_size=embedding_size, class_count=200)


def create_backbone(backbone_type, pretrained):
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
    elif backbone_type == 'open_face_inception':
        return OpenFaceInception()
    elif backbone_type == 'tiny_cnn':
        return TinyCnn()
    elif backbone_type == 'vgg_like':
        return VGGLike()
    else:
        raise ValueError('Invalid backbone type')


if __name__ == '__main__':
    main()
