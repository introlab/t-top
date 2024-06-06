# led_emotions

This folder contains the node to make T-Top express emotions with its LED strip.

## Nodes

### `led_emotions_node.py`

This node makes T-Top express emotions with it LED strip.

#### Parameters

- `period_s` (double): The period in second of the animation timer. The default value is 0.0333.
- `led_patterns_file` (string): The file path containing the LED patterns

#### Subscribed Topics

- `led_emotions/name` ([std_msgs/String](http://docs.ros.org/en/noetic/api/std_msgs/html/msg/String.html)): The emotion names.

#### Published Topics

- `led_emotions/set_led_colors` ([daemon_ros_client/LedColors](../../daemon_ros_client/msg/LedColors.msg)): The LED colors.

#### Services

- `set_led_colors/filter_state` ([hbba_lite_srvs/SetOnOffFilterState](../../utils/hbba_lite/hbba_lite_srvs/srv/SetOnOffFilterState.srv)): The HBBA filter
  state service to enable or disable the behavior.
