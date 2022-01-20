import argparse

import torch

from audio_descriptor.backbones import Mnasnet0_5, Mnasnet1_0, Resnet18, Resnet34, Resnet50, OpenFaceInception
from audio_descriptor.audio_descriptor_extractor import AudioDescriptorExtractor, AudioDescriptorExtractorVLAD
from audio_descriptor.trainers import AudioDescriptorExtractorTrainer


def main():
    parser = argparse.ArgumentParser(description='Train Backbone')
    parser.add_argument('--use_gpu', action='store_true', help='Use the GPU')
    parser.add_argument('--dataset_root', type=str, help='Choose the dataset root path', required=True)
    parser.add_argument('--output_path', type=str, help='Choose the output path', required=True)
    parser.add_argument('--backbone_type', choices=['mnasnet0.5', 'mnasnet1.0',
                                                    'resnet18', 'resnet34', 'resnet50',
                                                    'open_face_inception'],
                        help='Choose the backbone type', required=True)
    parser.add_argument('--embedding_size', type=int, help='Set the embedding size', required=True)
    parser.add_argument('--vlad', action='store_true', help='Use VLAD pooling layer')
    parser.add_argument('--waveform_size', type=int, help='Set the waveform size', required=True)
    parser.add_argument('--n_features', type=int, help='Set n_features', required=True)
    parser.add_argument('--n_fft', type=int, help='Set n_fft', required=True)
    parser.add_argument('--audio_transform_type', choices=['mfcc', 'mel_spectrogram'],
                        help='Choose the audio transform type', required=True)
    parser.add_argument('--enable_pitch_shifting', action='store_true', help='Use pitch shifting data augmentation')
    parser.add_argument('--enable_time_stretching', action='store_true', help='Use pitch shifting data augmentation')
    parser.add_argument('--margin', type=float, help='Set the margin', default=0.2)

    parser.add_argument('--learning_rate', type=float, help='Choose the learning rate', required=True)
    parser.add_argument('--batch_size', type=int, help='Set the batch size for the training', required=True)
    parser.add_argument('--epoch_count', type=int, help='Choose the epoch count', required=True)
    parser.add_argument('--criterion_type', choices=['triplet_loss', 'cross_entropy_loss', 'am_softmax_loss'],
                        help='Choose the criterion type', required=True)
    parser.add_argument('--dataset_class_count', type=int,
                        help='Choose the dataset class count when criterion_type is "cross_entropy_loss" or '
                             '"am_softmax_loss"',
                        default=None)

    parser.add_argument('--model_checkpoint', type=str, help='Choose the model checkpoint file', default=None)
    parser.add_argument('--optimizer_checkpoint', type=str, help='Choose the optimizer checkpoint file', default=None)
    parser.add_argument('--scheduler_checkpoint', type=str, help='Choose the scheduler checkpoint file', default=None)

    args = parser.parse_args()

    if args.criterion_type == 'triplet_loss' and args.dataset_class_count is None:
        model = create_model(args.backbone_type, args.embedding_size, vlad=args.vlad)
    elif args.criterion_type == 'cross_entropy_loss' and args.dataset_class_count is not None:
        model = create_model(args.backbone_type, args.embedding_size, args.dataset_class_count, vlad=args.vlad)
    elif args.criterion_type == 'am_softmax_loss' and args.dataset_class_count is not None:
        model = create_model(args.backbone_type, args.embedding_size, args.dataset_class_count,
                             am_softmax_linear=True, vlad=args.vlad)
    else:
        raise ValueError('--dataset_class_count must be used with "cross_entropy_loss" and "am_softmax_loss" criterion '
                         'types')
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')

    trainer = AudioDescriptorExtractorTrainer(device, model,
                                              epoch_count=args.epoch_count,
                                              learning_rate=args.learning_rate,
                                              dataset_root=args.dataset_root,
                                              output_path=args.output_path,
                                              batch_size=args.batch_size,
                                              criterion_type=args.criterion_type,
                                              waveform_size=args.waveform_size,
                                              n_features=args.n_features,
                                              n_fft=args.n_fft,
                                              audio_transform_type=args.audio_transform_type,
                                              enable_pitch_shifting=args.enable_pitch_shifting,
                                              enable_time_stretching=args.enable_time_stretching,
                                              margin=args.margin,
                                              model_checkpoint=args.model_checkpoint,
                                              optimizer_checkpoint=args.optimizer_checkpoint,
                                              scheduler_checkpoint=args.scheduler_checkpoint)
    trainer.train()


def create_model(backbone_type, embedding_size, class_count=None, am_softmax_linear=False, vlad=False):
    pretrained = True

    backbone = create_backbone(backbone_type, pretrained)
    if vlad:
        return AudioDescriptorExtractorVLAD(backbone, embedding_size=embedding_size,
                                            class_count=class_count, am_softmax_linear=am_softmax_linear)
    else:
        return AudioDescriptorExtractor(backbone, embedding_size=embedding_size,
                                        class_count=class_count, am_softmax_linear=am_softmax_linear)


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
    else:
        raise ValueError('Invalid backbone type')


if __name__ == '__main__':
    main()
