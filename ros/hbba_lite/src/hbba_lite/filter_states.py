from abc import ABC, abstractmethod

import rospy

from hbba_lite.srv import SetOnOffFilterState, SetOnOffFilterStateResponse, SetThrottlingFilterState, SetThrottlingFilterStateResponse


class _HbbaFilterState(ABC):
    @abstractmethod
    def check(self):
        return False

    @abstractmethod
    def on_changed(self, callback):
        pass


class OnOffHbbaFilterState(_HbbaFilterState):
    def __init__(self, state_service_name):
        self._state_service = rospy.Service(state_service_name, SetOnOffFilterState, self._state_service_callback)
        self._is_filtering_all_messages = True
        self._user_callback = None

    def _state_service_callback(self, request):
        previous_is_filtering_all_messages = self._is_filtering_all_messages
        self._is_filtering_all_messages = request.is_filtering_all_messages
        if self._user_callback is not None:
            self._user_callback(previous_is_filtering_all_messages, self._is_filtering_all_messages)

        return SetOnOffFilterStateResponse(ok=True)

    def on_changed(self, callback):
        self._user_callback = callback

    def check(self):
        return not self._is_filtering_all_messages

    @property
    def is_filtering_all_messages(self):
        return self._is_filtering_all_messages


class ThrottlingHbbaFilterState(_HbbaFilterState):
    def __init__(self, state_service_name):
        self._state_service = rospy.Service(state_service_name, SetThrottlingFilterState, self._state_service_callback)
        self._is_filtering_all_messages = True
        self._rate = 1
        self._counter = 0
        self._user_callback = None

    def _state_service_callback(self, request):
        if request.rate <= 0 :
            return SetThrottlingFilterStateResponse(ok=False)

        previous_is_filtering_all_messages = self._is_filtering_all_messages
        previous_rate = self._rate
        self._is_filtering_all_messages = request.is_filtering_all_messages
        self._rate = request.rate
        self._counter = 0
        if self._user_callback is not None:
            self._user_callback(previous_is_filtering_all_messages, self._is_filtering_all_messages, previous_rate, self._rate)

        return SetThrottlingFilterStateResponse(ok=True)

    def on_changed(self, callback):
        self._user_callback = callback

    def check(self):
        if self._is_filtering_all_messages:
            return False

        is_ready = False
        if self._counter == 0:
            is_ready = True
        self._counter = (self._counter + 1) % self._rate

        return is_ready

    @property
    def is_filtering_all_messages(self):
        return self._is_filtering_all_messages
