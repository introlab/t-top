import time

class Rate:

    def __init__(self, hz):
        self._last_time_s = time.time()
        self._sleep_duration_s = 1 / hz if hz > 0.0 else 0.0

    def sleep(self):
        curr_time_s = time.time()
        # detect time jumping backwards
        if self._last_time_s > curr_time_s:
            self._last_time_s = curr_time_s

        # calculate sleep interval
        elapsed_s = curr_time_s - self._last_time_s
        time.sleep(self._sleep_duration_s - elapsed_s)
        self._last_time_s = self._last_time_s + self._sleep_duration_s

        # detect time jumping forwards, as well as loops that are
        # inherently too slow
        if curr_time_s - self._last_time_s > self._sleep_duration_s * 2:
            self._last_time_s = curr_time_s
