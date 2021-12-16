import argparse
import os
import random

import scipy.io.wavfile

from keyword_spotting.datasets.folder_extractor import FolderExtractor


class FolderTAndTopExtractor(FolderExtractor):
    def _extract(self, input, output, filename):
        waveform = self._load_waveform(input, filename)
        split_index = self._get_split_index(waveform)
        if split_index is not None:
            self._extract_t_and_top(output, filename, waveform, split_index)
        else:
            print('Warning: {} extraction is invalid'.format(filename))

    def _get_split_index(self, waveform):
        waveform_avg_envelope = self._get_waveform_avg_envelope(waveform)
        rectangles = self._get_rectangles(waveform_avg_envelope)

        if len(rectangles) != 4 or rectangles[2]['value']:
            return None

        return (rectangles[2]['end_index'] + rectangles[2]['start_index']) // 2

    def _extract_t_and_top(self, output, filename, waveform, split_index):
        offset = random.randrange(self._output_max_offset) + self._waveform_padding
        t_waveform = waveform[offset:split_index]

        offset = random.randrange(self._output_max_offset) + self._waveform_padding
        top_waveform = waveform[split_index:-offset]

        basename = os.path.splitext(filename)[0]
        scipy.io.wavfile.write(os.path.join(output, basename + '_t.wav'), self._output_fs, t_waveform)
        scipy.io.wavfile.write(os.path.join(output, basename + '_top.wav'), self._output_fs, top_waveform)


def main():
    parser = argparse.ArgumentParser(description='Extract T-Top from wav files')
    parser.add_argument('--input', type=str, help='Choose the input folder', required=True)
    parser.add_argument('--output', type=str, help='Choose the output folder', required=True)

    parser.add_argument('--noise_threshold', type=float, default=0.1)
    parser.add_argument('--mean_avg_sample_count', type=int, default=1000)
    parser.add_argument('--output_fs', type=int, default=16000)
    parser.add_argument('--waveform_padding', type=int, default=16000)
    parser.add_argument('--output_sample_count', type=int, default=16000)
    parser.add_argument('--output_max_offset', type=int, default=160)

    args = parser.parse_args()

    extractor = FolderTAndTopExtractor(args.noise_threshold, args.mean_avg_sample_count, args.output_fs,
                                       args.waveform_padding, args.output_sample_count, args.output_max_offset)
    extractor.extract(args.input, args.output)


if __name__ == '__main__':
    main()
