from abc import ABC, abstractmethod
from typing import Dict
import uuid
import os
import json
import random

from google.cloud import texttospeech


class VoiceGenerator(ABC):
    def __init__(self, directory: str, language: str):
        self._directory = directory
        self._language = language

        os.makedirs(directory, exist_ok=True)

    @abstractmethod
    def generate(self, text: str) -> str:
        pass

    def _generate_random_path(self, extension: str) -> str:
        return os.path.join(self._directory, str(uuid.uuid4()) + extension)


class GoogleVoiceGenerator(VoiceGenerator):
    def __init__(self, directory: str, language: str, speaking_rate: float):
        super().__init__(directory, language)
        self._speaking_rate = speaking_rate
        self._language_code = self._get_language_code()

    def generate(self, text: str) -> str:

        client = texttospeech.TextToSpeechClient()

        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(language_code=self._language_code,
                                                  ssml_gender=texttospeech.SsmlVoiceGender.MALE)
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3,
                                                speaking_rate=self._speaking_rate)

        response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)

        file_path = self._generate_random_path('.mp3')
        with open(file_path, 'wb') as file:
            file.write(response.audio_content)

        return file_path

    def _get_language_code(self) -> str:
        if self._language == 'en':
            return 'en-US'
        elif self._language == 'fr':
            return 'fr-CA'
        else:
            raise ValueError(f'Not supported language ({self._language})')


class CachedVoiceGenerator(VoiceGenerator):
    def __init__(self, voice_generator: VoiceGenerator, cache_size: int):
        super().__init__(voice_generator._directory, voice_generator._language)
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
        if text in self._index:
            return self._index[text]

        if len(self._index) + 1 > self._cache_size:
            self._remove_one_cache_item()

        file_path = self._voice_generator.generate(text)
        self._index[text] = file_path
        self._save_index(self._index)

        return file_path
