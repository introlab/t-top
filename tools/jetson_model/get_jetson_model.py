#! /usr/bin/env python3

from enum import Enum
from pathlib import Path
from typing import Union


def read_nvidia_dts() -> str:
    path = Path("/proc/device-tree/nvidia,dtsfilename")
    return path.read_text()


def get_p_number(text: str) -> str:
    return "-".join(text.split("/")[-1].split("-")[1:3])


class UnknownModel(Enum):
    UNKNOWN_JETSON = "unknown_jetson"
    OTHER = "not_jetson"


class JetsonModel(Enum):
    XAVIER = "xavier"
    ORIN = "orin"


# https://docs.nvidia.com/jetson/archives/r35.3.1/DeveloperGuide/text/IN/QuickStart.html
P_NUMBER_MAPPING = {
    "p3701-0000": JetsonModel.ORIN,
    "p2888-0001": JetsonModel.XAVIER,
    "p2888-0003": JetsonModel.XAVIER,
    "p2888-0005": JetsonModel.XAVIER,
}


def get_model_from_p_number(p_number: str) -> Union[JetsonModel, UnknownModel]:
    if p_number in P_NUMBER_MAPPING:
        return P_NUMBER_MAPPING[p_number]

    return UnknownModel.UNKNOWN_JETSON


def get_model() -> Union[JetsonModel, UnknownModel]:
    try:
        p_number = get_p_number(read_nvidia_dts())
    except FileNotFoundError:
        return UnknownModel.OTHER
    except IndexError:
        return UnknownModel.UNKNOWN_JETSON

    return get_model_from_p_number(p_number)


def main():
    print(get_model().value)


if __name__ == "__main__":
    main()
