from abc import ABC, abstractmethod
from typing import Dict, Tuple
import uuid
import os
import json
import random
from enum import Enum

from google.cloud import texttospeech

import rclpy
import rclpy.node

from piper_ros.srv import GenerateSpeechFromText


class Language(Enum):
    ENGLISH = 'en'
    FRENCH = 'fr'

    @staticmethod
    def from_name(name: str) -> 'Language':
        if name == Language.ENGLISH.value:
            return Language.ENGLISH
        elif name == Language.FRENCH.value:
            return Language.FRENCH
        else:
            raise ValueError(f'Invalid language name ({name})')


class Gender(Enum):
    FEMALE = 'female'
    MALE = 'male'

    @staticmethod
    def from_name(name: str) -> 'Gender':
        if name == Gender.FEMALE.value:
            return Gender.FEMALE
        elif name == Gender.MALE.value:
            return Gender.MALE
        else:
            raise ValueError(f'Invalid gender name ({name})')


class VoiceGenerator(ABC):
    def __init__(self, directory: str, language: Language, gender: Gender, speaking_rate: float):
        self._directory = directory
        self._language = language
        self._gender = gender
        self._speaking_rate = speaking_rate

        os.makedirs(directory, exist_ok=True)

    @abstractmethod
    def generate(self, text: str) -> str:
        pass

    def delete_generated_file(self, file_path: str):
        if os.path.exists(file_path):
            os.remove(file_path)

    def _generate_random_path(self, extension: str) -> str:
        return os.path.join(self._directory, str(uuid.uuid4()) + extension)


class GoogleVoiceGenerator(VoiceGenerator):
    def __init__(self, directory: str, language: Language, gender: Gender, speaking_rate: float):
        super().__init__(directory, language, gender, speaking_rate)
        self._language_code, self._voice_name = self._get_language_code_and_name(language, gender)

    @staticmethod
    def _get_language_code_and_name(language: Language, gender: Gender) -> Tuple[str, str]:
        if language == Language.ENGLISH and gender == Gender.MALE:
            return 'en-US', None
        elif language == Language.ENGLISH and gender == Gender.FEMALE:
            return 'en-US', 'en-US-Standard-G'
        elif language == Language.FRENCH and gender == Gender.MALE:
            return 'fr-CA', 'fr-CA-Standard-B'
        elif language == Language.FRENCH and gender == Gender.FEMALE:
            return 'fr-CA', 'fr-CA-Standard-C'
        else:
            raise ValueError(f'Invalid language and/or gender ({language.value}, {gender.value})')

    def generate(self, text: str) -> str:
        client = texttospeech.TextToSpeechClient()

        synthesis_input = texttospeech.SynthesisInput(text=text)
        if self._voice_name is not None:
            voice = texttospeech.VoiceSelectionParams(language_code=self._language_code, name=self._voice_name)
        else:
            voice = texttospeech.VoiceSelectionParams(language_code=self._language_code,
                                                      ssml_gender=texttospeech.SsmlVoiceGender.MALE)

        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3,
                                                speaking_rate=self._speaking_rate)

        response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)

        file_path = self._generate_random_path('.mp3')
        with open(file_path, 'wb') as file:
            file.write(response.audio_content)

        return file_path


class PiperVoiceGenerator(VoiceGenerator):
    def __init__(self, node: rclpy.node.Node, directory: str, language: Language, gender: Gender, speaking_rate: float):
        super().__init__(directory, language, gender, speaking_rate)

        self._node = node
        self._piper_service = node.create_client(GenerateSpeechFromText, 'piper/generate_speech_from_text')

    def generate(self, text: str) -> str:
        request = GenerateSpeechFromText.Request()
        request.language = self._language.value
        request.gender = self._gender.value
        request.length_scale = 1.0 / self._speaking_rate
        request.text = text
        request.path = self._generate_random_path('.wav')

        future = self._piper_service.call_async(request)
        rclpy.spin_until_future_complete(self._node, future)
        response = future.result()

        if not response.ok:
            raise RuntimeError(response.message)

        return request.path


class CachedVoiceGenerator(VoiceGenerator):
    def __init__(self, voice_generator: VoiceGenerator, cache_size: int):
        super().__init__(voice_generator._directory, voice_generator._language, voice_generator._gender, voice_generator._speaking_rate)

        if cache_size < 1:
            raise ValueError('The cache size must be at least 1.')

        self._voice_generator = voice_generator
        self._cache_size = cache_size

        self._index_path = os.path.join(self._directory, 'index.json')
        self._index = self._load_index()

        self._remove_missing_cache_items()
        while len(self._index) > self._cache_size:
            self._remove_one_cache_item()

    def _load_index(self) -> Dict[str, str]:
        try:
            with open(self._index_path, 'r') as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_index(self, index: Dict[str, str]):
        with open(self._index_path, 'w') as f:
            json.dump(index, f)

    def _remove_missing_cache_items(self):
        for text, path in list(self._index.items()):
            if not os.path.exists(path):
                del self._index[text]

        self._save_index(self._index)

    def _remove_one_cache_item(self):
        text = random.choice(list(self._index.keys()))
        path = self._index[text]
        if os.path.exists(path):
            os.remove(path)

        del self._index[text]

        self._save_index(self._index)

    def generate(self, text: str) -> str:
        key = self._get_cache_key(text)

        if key in self._index:
            return self._index[key]

        if len(self._index) + 1 > self._cache_size:
            self._remove_one_cache_item()

        file_path = self._voice_generator.generate(text)
        self._index[key] = file_path
        self._save_index(self._index)

        return file_path

    def _get_cache_key(self, text: str) -> str:
        return (f'{type(self._voice_generator).__name__}__{self._language.value}__{self._gender.value}__'
                f'{self._speaking_rate}__{text}')

    def delete_generated_file(self, file_path: str):
        pass
