import os

import torch
import torchaudio.transforms as transforms

from dnn_utils.dnn_model import PACKAGE_PATH, DnnModel
from dnn_utils.audio_transforms import normalize, standardize_every_frame

# TODO set DURATION = 63840, N_MELS = 96 N_FFT = 480
DURATION = 64000
SAMPLING_FREQUENCY = 16000
N_MELS = 128
N_FFT = 400


class MulticlassAudioDescriptorExtractor(DnnModel):
    def __init__(self, inference_type=None):
        torch_script_model_path = os.path.join(PACKAGE_PATH, 'models', 'multiclass_audio_descriptor_extractor.ts.pth')
        tensor_rt_model_path = os.path.join(PACKAGE_PATH, 'models', 'multiclass_audio_descriptor_extractor.trt.pth')
        sample_input = torch.ones((1, 1, N_MELS, int(2 * DURATION / N_FFT) + 1))

        super(MulticlassAudioDescriptorExtractor, self).__init__(torch_script_model_path, tensor_rt_model_path, sample_input,
                                                       inference_type=inference_type)
        self._transform = transforms.MelSpectrogram(sample_rate=SAMPLING_FREQUENCY,
                                                    n_fft=N_FFT,
                                                    n_mels=N_MELS).to(self._device)

    def get_supported_sampling_frequency(self):
        return SAMPLING_FREQUENCY

    def get_supported_duration(self):
        return DURATION

    def get_class_names(self):
        return ['Accelerating_and_revving_and_vroom', 'Accordion', 'Acoustic_guitar', 'Aircraft', 'Alarm', 'Animal',
                'Applause', 'Bark', 'Bass_drum', 'Bass_guitar', 'Bathtub_(filling_or_washing)', 'Bell', 'Bicycle',
                'Bicycle_bell', 'Bird', 'Bird_vocalization_and_bird_call_and_bird_song', 'Boat_and_Water_vehicle',
                'Boiling', 'Boom', 'Bowed_string_instrument', 'Brass_instrument', 'Breathing', 'Burping_and_eructation',
                'Bus', 'Buzz', 'Camera', 'Car', 'Car_passing_by', 'Cat', 'Chatter', 'Cheering', 'Chewing_and_mastication',
                'Chicken_and_rooster', 'Child_speech_and_kid_speaking', 'Chime', 'Chink_and_clink', 'Chirp_and_tweet',
                'Chuckle_and_chortle', 'Church_bell', 'Clapping', 'Clock', 'Coin_(dropping)', 'Computer_keyboard',
                'Conversation', 'Cough', 'Cowbell', 'Crack', 'Crackle', 'Crash_cymbal', 'Cricket', 'Crow', 'Crowd',
                'Crumpling_and_crinkling', 'Crushing', 'Crying_and_sobbing', 'Cupboard_open_or_close',
                'Cutlery_and_silverware', 'Cymbal', 'Dishes_and_pots_and_pans', 'Dog', 'Domestic_animals_and_pets',
                'Domestic_sounds_and_home_sounds', 'Door', 'Doorbell', 'Drawer_open_or_close', 'Drill', 'Drip', 'Drum',
                'Drum_kit', 'Electric_guitar', 'Engine', 'Engine_starting', 'Explosion', 'Fart', 'Female_singing',
                'Female_speech_and_woman_speaking', 'Fill_(with_liquid)', 'Finger_snapping', 'Fire', 'Fireworks',
                'Fixed-wing_aircraft_and_airplane', 'Fowl', 'Frog', 'Frying_(food)', 'Gasp', 'Giggle', 'Glass',
                'Glockenspiel', 'Gong', 'Growling', 'Guitar', 'Gull_and_seagull', 'Gunshot_and_gunfire', 'Gurgling',
                'Hammer', 'Hands', 'Harmonica', 'Harp', 'Hi-hat', 'Hiss', 'Human_group_actions', 'Human_voice',
                'Idling', 'Insect', 'Keyboard_(musical)', 'Keys_jangling', 'Knock', 'Laughter', 'Liquid',
                'Livestock_and_farm_animals_and_working_animals', 'Male_singing', 'Male_speech_and_man_speaking',
                'Mallet_percussion', 'Marimba_and_xylophone', 'Mechanical_fan', 'Mechanisms', 'Meow', 'Microwave_oven',
                'Motor_vehicle_(road)', 'Motorcycle', 'Music', 'Musical_instrument', 'Ocean', 'Organ',
                'Packing_tape_and_duct_tape', 'Percussion', 'Piano', 'Plucked_string_instrument', 'Pour', 'Power_tool',
                'Printer', 'Purr', 'Race_car_and_auto_racing', 'Rail_transport', 'Rain', 'Raindrop', 'Ratchet_and_pawl',
                'Rattle', 'Rattle_(instrument)', 'Respiratory_sounds', 'Ringtone', 'Run', 'Sawing', 'Scissors',
                'Scratching_(performance_technique)', 'Screaming', 'Screech', 'Shatter', 'Shout', 'Sigh', 'Singing',
                'Sink_(filling_or_washing)', 'Siren', 'Skateboard', 'Slam', 'Sliding_door', 'Snare_drum', 'Sneeze',
                'Speech', 'Speech_synthesizer', 'Splash_and_splatter', 'Squeak', 'Stream', 'Strum',
                'Subway_and_metro_and_underground', 'Tabla', 'Tambourine', 'Tap', 'Tearing', 'Telephone',
                'Thump_and_thud', 'Thunder', 'Thunderstorm', 'Tick', 'Tick-tock', 'Toilet_flush', 'Tools',
                'Traffic_noise_and_roadway_noise', 'Train', 'Trickle_and_dribble', 'Truck', 'Trumpet', 'Typewriter',
                'Typing', 'Vehicle', 'Vehicle_horn_and_car_horn_and_honking', 'Walk_and_footsteps', 'Water',
                'Water_tap_and_faucet', 'Waves_and_surf', 'Whispering', 'Whoosh_and_swoosh_and_swish', 'Wild_animals',
                'Wind', 'Wind_chime', 'Wind_instrument_and_woodwind_instrument', 'Wood', 'Writing', 'Yell', 'Zipper_(clothing)']

    def __call__(self, x):
        with torch.no_grad():
            x = x.to(self._device)
            x = normalize(x)
            spectrogram = self._transform(x).unsqueeze(0)
            # spectrogram = standardize_every_frame(spectrogram) TODO use standardize_every_frame
            descriptor, class_scores = super(MulticlassAudioDescriptorExtractor, self).__call__(spectrogram.unsqueeze(0))
            probabilities = torch.sigmoid(class_scores[0])

            return descriptor[0].cpu(), probabilities.cpu()
