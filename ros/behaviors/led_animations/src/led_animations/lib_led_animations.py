from abc import ABC, abstractmethod
import random
from typing import List

import numpy as np

from daemon_ros_client.msg import LedColors, LedColor


LED_COUNT = 28


class LedAnimation(ABC):
    def __init__(self, period_s, speed, colors):
        self._period_s = period_s
        self._speed = speed
        self._colors = colors

    @abstractmethod
    def update(self):
        pass

    @staticmethod
    def from_name(name, period_s, speed, colors):
        if name == 'constant':
            return ConstantLedAnimation(period_s, speed, colors)
        elif name == 'rotating_sin':
            return RotatingSinLedAnimation(period_s, speed, colors)
        elif name == 'random':
            return RandomLedAnimation(period_s, speed, colors)
        else:
            raise ValueError('Invalid Name')


class ConstantLedAnimation(LedAnimation):
    def __init__(self, period_s, speed, colors):
        super().__init__(period_s, speed, colors)

        if len(self._colors) == 1:
            self._contant_colors = LedColors()
            for c in self._contant_colors.colors:
                c.red = self._colors[0].red
                c.green = self._colors[0].green
                c.blue = self._colors[0].red
        elif len(self._colors) == LED_COUNT:
            self._contant_colors = LedColors(colors=self._colors)
        else:
            raise ValueError('Invalid colors for the random animation')

    def update(self) -> LedColors:
        return self._contant_colors


class RotatingSinLedAnimation(LedAnimation):
    def __init__(self, period_s, speed, colors):
        super().__init__(period_s, speed, colors)

        if len(self._colors) != 1:
            raise ValueError('Invalid colors for the rotating_sin animation')

        self._phase = 0
        self._x = np.linspace(0, 1, LED_COUNT)

    def update(self) -> LedColors:
        gain = 0.5 * np.cos(2 * np.pi * (self._x + self._phase)) + 0.5

        self._phase += self._period_s * self._speed
        if self._phase > 1.0:
            self._phase -= 1.0

        led_colors = LedColors()
        for i, led_color in enumerate(led_colors.colors):
            led_color.red = int(self._colors[0].red * gain[i])
            led_color.green = int(self._colors[0].green * gain[i])
            led_color.blue = int(self._colors[0].blue * gain[i])

        return led_colors


class RandomLedAnimation(LedAnimation):
    def __init__(self, period_s, speed, colors):
        super().__init__(period_s, speed, colors)

        self._alpha = 0

        if len(self._colors) == 0:
            self._current_colors = self._random_colors()
        elif len(self._colors) == LED_COUNT:
            self._current_colors = self._colors
        else:
            raise ValueError('Invalid colors for the random animation')

        self._next_colors = self._random_colors()

    def update(self) -> LedColors:
        self._alpha += self._period_s * self._speed
        if self._alpha > 1.0:
            self._alpha -= 1.0
            self._current_colors = self._next_colors
            self._next_colors = self._random_colors()

        led_colors = LedColors()
        for i in range(LED_COUNT):
            led_colors.colors[i] = self._interpolate_color(self._current_colors[i], self._next_colors[i])

        return led_colors

    def _interpolate_color(self, current_color, next_color):
        color = LedColor()
        color.red = int(self._alpha * (next_color.red - current_color.red) + current_color.red)
        color.green = int(self._alpha * (next_color.green - current_color.green) + current_color.green)
        color.blue = int(self._alpha * (next_color.blue - current_color.blue) + current_color.blue)
        return color

    def _random_colors(self):
        return [self._random_color() for _ in range(LED_COUNT)]

    def _random_color(self):
        color = LedColor()
        color.red = random.randint(0, 255)
        color.green = random.randint(0, 255)
        color.blue = random.randint(0, 255)
        return color
