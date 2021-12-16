import argparse
import os
import random

import scipy.io.wavfile

from keyword_spotting.datasets.folder_extractor import FolderExtractor


class FolderTTopExtractor(FolderExtractor):
    def __init__(self, noise_threshold, mean_avg_sample_count, output_fs, min_noise_sample_count,
                 waveform_padding, output_sample_count, output_max_offset):
        super(FolderTTopExtractor, self).__init__(noise_threshold, mean_avg_sample_count, output_fs,
                                                  waveform_padding, output_sample_count, output_max_offset)
        self._min_noise_sample_count = min_noise_sample_count

    def _extract(self, input, output, filename):
        waveform = self._load_waveform(input, filename)
        ttop_rectangles = self._get_ttop_rectangles(waveform)
        for i, ttop_rectangle in enumerate(ttop_rectangles):
            self._extract_ttop(output, filename, waveform, ttop_rectangle, i)

    def _get_ttop_rectangles(self, waveform):
        waveform_avg_envelope = self._get_waveform_avg_envelope(waveform)
        rectangles = self._get_rectangles(waveform_avg_envelope)
        merged_rectangles = self._get_merged_rectangles(rectangles)
        return [o for o in merged_rectangles if o['value']]

    def _get_merged_rectangles(self, rectangles):
        merged_rectangles = [rectangles[0]]
        i = 1

        while i < len(rectangles) - 1:
            sample_count = rectangles[i]['end_index'] - rectangles[i]['start_index']

            if sample_count < self._min_noise_sample_count and merged_rectangles[-1]['value']:
                merged_rectangles[-1]['end_index'] = rectangles[i + 1]['end_index']
                i += 1
            else:
                merged_rectangles.append(rectangles[i])

            i += 1

        return merged_rectangles

    def _extract_ttop(self, output, filename, waveform, ttop_rectangle, index):
        ttop_sample_count = ttop_rectangle['end_index'] - ttop_rectangle['start_index']
        padding = (self._output_sample_count - ttop_sample_count) // 2
        start_index = ttop_rectangle['start_index'] - padding
        end_index = ttop_rectangle['end_index'] + padding

        offset = random.randint(-self._output_max_offset, self._output_max_offset)
        start_index += offset
        end_index += offset

        ttop_waveform = waveform[start_index:end_index]
        basename = os.path.splitext(filename)[0]
        scipy.io.wavfile.write(os.path.join(output, basename + '_{}.wav'.format(index)), self._output_fs, ttop_waveform)


def main():
    parser = argparse.ArgumentParser(description='Extract T-Top from wav files')
    parser.add_argument('--input', type=str, help='Choose the input folder', required=True)
    parser.add_argument('--output', type=str, help='Choose the output folder', required=True)

    parser.add_argument('--noise_threshold', type=float, default=0.1)
    parser.add_argument('--mean_avg_sample_count', type=int, default=1000)
    parser.add_argument('--output_fs', type=int, default=16000)
    parser.add_argument('--min_noise_sample_count', type=int, default=16000)
    parser.add_argument('--waveform_padding', type=int, default=16000)
    parser.add_argument('--output_sample_count', type=int, default=16000)
    parser.add_argument('--output_max_offset', type=int, default=1600)

    args = parser.parse_args()

    extractor = FolderTTopExtractor(args.noise_threshold, args.mean_avg_sample_count, args.output_fs,
                                    args.min_noise_sample_count, args.waveform_padding, args.output_sample_count,
                                    args.output_max_offset)
    extractor.extract(args.input, args.output)


if __name__ == '__main__':
    main()
