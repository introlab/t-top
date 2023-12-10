# led_animations

This folder contains the node to perform animations with the LED strip.

## Nodes

### `led_animations_node.py`

This node performs animations with the LED strip.

#### Subscribed Topics

- `led_animations/name` ([led_animations/Animation](msg/Animation.msg)): The animation parameters.

#### Published Topics

- `led_animations/set_led_colors` ([daemon_ros_client/LedColors](../../daemon_ros_client/msg/LedColors.msg)): The LED colors.
- `led_animations/done` ([led_animations/Done](msg/Done.msg)): Indicates that the animation is finished.

#### Services

- `set_led_colors/filter_state` ([hbba_lite/SetOnOffFilterState](../../hbba_lite/srv/SetOnOffFilterState.srv)): The HBBA filter
  state service to enable or disable the behavior.
