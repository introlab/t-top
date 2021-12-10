import rospy

from hbba_lite.filter_states import OnOffHbbaFilterState, ThrottlingHbbaFilterState


class _HbbaPublisher:
    def __init__(self, topic_name, data_class, filter_state_class, state_service_name, queue_size):
        if state_service_name is None:
            state_service_name = topic_name + '/filter_state'

        self._filter_state = filter_state_class(state_service_name)
        self._subscriber = rospy.Publisher(topic_name, data_class, queue_size=queue_size)

    def publish(self, msg):
        if self._filter_state.check():
            self._subscriber.publish(msg)


class OnOffHbbaPublisher(_HbbaPublisher):
    def __init__(self, topic_name, data_class, state_service_name=None, queue_size=None):
        super(OnOffHbbaPublisher, self).__init__(topic_name, data_class, OnOffHbbaFilterState,
                                                 state_service_name, queue_size)


class ThrottlingHbbaPublisher(_HbbaPublisher):
    def __init__(self, topic_name, data_class, callback=None, state_service_name=None, queue_size=None):
        super(ThrottlingHbbaPublisher, self).__init__(topic_name, data_class, ThrottlingHbbaFilterState,
                                                      state_service_name, queue_size)
