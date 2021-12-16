import rospy
import message_filters

from hbba_lite.filter_states import OnOffHbbaFilterState, ThrottlingHbbaFilterState


class _HbbaSubscriber:
    def __init__(self, topic_name, data_class, filter_state_class, callback, state_service_name, queue_size):
        if state_service_name is None:
            state_service_name = topic_name + '/filter_state'
        self._callback = callback

        self._filter_state = filter_state_class(state_service_name)
        self._subscriber = rospy.Subscriber(topic_name, data_class, self._subscriber_callback, queue_size=queue_size)

    def _subscriber_callback(self, msg):
        if self._filter_state.check() and self._callback is not None:
            self._callback(msg)

    @property
    def is_filtering_all_messages(self):
        return self._filter_state._is_filtering_all_messages

    def on_filter_state_changed(self, callback):
        self._filter_state.on_changed(callback)

    def unregister(self):
        self._subscriber.unregister()


class OnOffHbbaSubscriber(_HbbaSubscriber):
    def __init__(self, topic_name, data_class, callback=None, state_service_name=None, queue_size=None):
        super(OnOffHbbaSubscriber, self).__init__(topic_name, data_class, OnOffHbbaFilterState,
                                                  callback, state_service_name, queue_size)


class ThrottlingHbbaSubscriber(_HbbaSubscriber):
    def __init__(self, topic_name, data_class, callback=None, state_service_name=None, queue_size=None):
        super(ThrottlingHbbaSubscriber, self).__init__(topic_name, data_class, ThrottlingHbbaFilterState,
                                                       callback, state_service_name, queue_size)


class _HbbaTimeSynchronizer:
    def __init__(self, message_filter_subscribers, filter_state_class, callback, state_service_name,
                queue_size):
        self._callback = callback

        self._filter_state = filter_state_class(state_service_name)
        self._ts = message_filters.TimeSynchronizer(message_filter_subscribers, queue_size=queue_size)
        self._ts.registerCallback(self._ts_callback)

    def _ts_callback(self, *args):
        if self._filter_state.check() and self._callback is not None:
            self._callback(*args)

    @property
    def is_filtering_all_messages(self):
        return self._filter_state._is_filtering_all_messages

    def on_filter_state_changed(self, callback):
        self._filter_state.on_changed(callback)


class OnOffHbbaTimeSynchronizer(_HbbaTimeSynchronizer):
    def __init__(self, message_filter_subscribers, queue_size, callback=None, state_service_name=None):
        super(OnOffHbbaTimeSynchronizer, self).__init__(message_filter_subscribers, OnOffHbbaFilterState,
                                                        callback, state_service_name, queue_size)


class ThrottlingHbbaTimeSynchronizer(_HbbaTimeSynchronizer):
    def __init__(self, message_filter_subscribers, queue_size, callback=None, state_service_name=None):
        super(ThrottlingHbbaTimeSynchronizer, self).__init__(message_filter_subscribers, ThrottlingHbbaFilterState,
                                                        callback, state_service_name, queue_size)


class _HbbaApproximateTimeSynchronizer:
    def __init__(self, message_filter_subscribers, filter_state_class, callback, state_service_name,
                queue_size, slop, allow_headerless):
        self._callback = callback

        self._filter_state = filter_state_class(state_service_name)
        self._ts = message_filters.ApproximateTimeSynchronizer(message_filter_subscribers, queue_size, slop,
                                                               allow_headerless=allow_headerless)
        self._ts.registerCallback(self._ts_callback)

    def _ts_callback(self, *args):
        if self._filter_state.check() and self._callback is not None:
            self._callback(*args)

    @property
    def is_filtering_all_messages(self):
        return self._filter_state._is_filtering_all_messages

    def on_filter_state_changed(self, callback):
        self._filter_state.on_changed(callback)


class OnOffHbbaApproximateTimeSynchronizer(_HbbaApproximateTimeSynchronizer):
    def __init__(self, message_filter_subscribers, queue_size, slop, callback=None, state_service_name=None,
                 allow_headerless=False):
        super(OnOffHbbaApproximateTimeSynchronizer, self).__init__(message_filter_subscribers,
                                                                   OnOffHbbaFilterState,
                                                                   callback, state_service_name,
                                                                   queue_size, slop, allow_headerless)


class ThrottlingHbbaApproximateTimeSynchronizer(_HbbaApproximateTimeSynchronizer):
    def __init__(self, message_filter_subscribers, queue_size, slop, callback=None, state_service_name=None,
                 allow_headerless=False):
        super(ThrottlingHbbaApproximateTimeSynchronizer, self).__init__(message_filter_subscribers,
                                                                        ThrottlingHbbaFilterState,
                                                                        callback, state_service_name,
                                                                        queue_size, slop, allow_headerless)
