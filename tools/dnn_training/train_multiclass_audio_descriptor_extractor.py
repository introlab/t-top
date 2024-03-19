import argparse
import os

import torch

from common.program_arguments import save_arguments, print_arguments

from audio_descriptor.trainers import MulticlassAudioDescriptorExtractorTrainer

from export_audio_descriptor_extractor import create_model


def main():
    parser = argparse.ArgumentParser(description='Train Backbone')
    parser.add_argument('--use_gpu', action='store_true', help='Use the GPU')
    parser.add_argument('--dataset_root', type=str, help='Choose the dataset root path ()', required=True)
    parser.add_argument('--dataset_type', choices=['audio_set', 'fsd50k'], help='Choose the database type',
                        required=True)
    parser.add_argument('--output_path', type=str, help='Choose the output path', required=True)
    parser.add_argument('--backbone_type', choices=['mnasnet0.5', 'mnasnet1.0',
                                                    'resnet18', 'resnet34', 'resnet50', 'resnet101',
                                                    'open_face_inception', 'thin_resnet_34',
                                                    'ecapa_tdnn_512', 'ecapa_tdnn_1024',
                                                    'small_ecapa_tdnn_128', 'small_ecapa_tdnn_256',
                                                    'small_ecapa_tdnn_512',
                                                    'passt_s_n', 'passt_s_n_l'],
                        help='Choose the backbone type', required=True)
    parser.add_argument('--conv_bias', action='store_true', help='Set bias=True to Conv2d (open_face_inception only)')
    parser.add_argument('--embedding_size', type=int, help='Set the embedding size', required=True)
    parser.add_argument('--pooling_layer', choices=['avg', 'vlad', 'sap', 'psla'], help='Set the pooling layer')
    parser.add_argument('--waveform_size', type=int, help='Set the waveform size', required=True)
    parser.add_argument('--n_features', type=int, help='Set n_features', required=True)
    parser.add_argument('--n_fft', type=int, help='Set n_fft', required=True)
    parser.add_argument('--audio_transform_type',
                        choices=['mfcc', 'log_mel_spectrogram', 'mel_spectrogram', 'spectrogram', 'psla'],
                        help='Choose the audio transform type', required=True)
    parser.add_argument('--enable_pitch_shifting', action='store_true', help='Use pitch shifting data augmentation')
    parser.add_argument('--enable_time_stretching', action='store_true', help='Use pitch shifting data augmentation')
    parser.add_argument('--enable_time_masking', action='store_true', help='Use time masking data augmentation')
    parser.add_argument('--enable_frequency_masking', action='store_true', help='Use time masking data augmentation')
    parser.add_argument('--enhanced_targets', action='store_true', help='Use enhanced targets')

    parser.add_argument('--learning_rate', type=float, help='Choose the learning rate', required=True)
    parser.add_argument('--weight_decay', type=float, help='Choose the weight decay', required=True)
    parser.add_argument('--batch_size', type=int, help='Set the batch size for the training', required=True)
    parser.add_argument('--epoch_count', type=int, help='Choose the epoch count', required=True)
    parser.add_argument('--criterion_type', choices=['bce_loss', 'sigmoid_focal_loss'],
                        help='Choose the criterion type', required=True)

    parser.add_argument('--model_checkpoint', type=str, help='Choose the model checkpoint file', default=None)

    args = parser.parse_args()

    if args.dataset_type == 'audio_set':
        class_count = 527
    elif args.dataset_type == 'fsd50k':
        class_count = 200
    else:
        raise ValueError('Invalid dataset type')

    model = create_model(args.backbone_type, args.n_features, args.embedding_size, class_count=class_count,
                         pooling_layer=args.pooling_layer, conv_bias=args.conv_bias)
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')

    output_path = os.path.join(args.output_path, args.backbone_type + '_' + args.dataset_type +
                               '_e' + str(args.embedding_size) + '_' + args.audio_transform_type +
                               '_enhanced' + str(int(args.enhanced_targets)) + '_' + args.criterion_type
                               + '_lr' + str(args.learning_rate) + '_wd' + str(args.weight_decay))
    save_arguments(output_path, args)
    print_arguments(args)

    trainer = MulticlassAudioDescriptorExtractorTrainer(device, model,
                                                        epoch_count=args.epoch_count,
                                                        learning_rate=args.learning_rate,
                                                        weight_decay=args.weight_decay,
                                                        dataset_root=args.dataset_root,
                                                        dataset_type=args.dataset_type,
                                                        output_path=output_path,
                                                        batch_size=args.batch_size,
                                                        waveform_size=args.waveform_size,
                                                        n_features=args.n_features,
                                                        n_fft=args.n_fft,
                                                        audio_transform_type=args.audio_transform_type,
                                                        enable_pitch_shifting=args.enable_pitch_shifting,
                                                        enable_time_stretching=args.enable_time_stretching,
                                                        enable_time_masking=args.enable_time_masking,
                                                        enable_frequency_masking=args.enable_frequency_masking,
                                                        enhanced_targets=args.enhanced_targets,
                                                        model_checkpoint=args.model_checkpoint)
    trainer.train()


if __name__ == '__main__':
    main()
