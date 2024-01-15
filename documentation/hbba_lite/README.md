# HBBA Lite
HBBA (Hybrid Behavior-Based Architecture) is the control architecture of T-Top. A hybrid robot architecture means that perceptual nodes can communicate with the behavior nodes and the planning modules.
T-Top uses [HBBA Lite](https://github.com/introlab/hbba_lite/) implementation instead of the [original one](https://github.com/introlab/HBBA) since HBBA Lite is simpler to use but less complete.

## Core
See the [HBBA Lite documentation](https://github.com/introlab/hbba_lite#documentation).

## T-Top
The following sections present T-Top specific information.

### Resources
- `motor`: The resource to prevent conflicts over the motors.
- `audio`: The resource to prevent conflicts over the sound output.
- `led`: The resource to prevent conflicts over the LEDs.

### Desire Types
- [`Camera3dRecordingDesire`](../../ros/utils/t_top_hbba_lite/include/t_top_hbba_lite/Desires.h#L12): To enable the recording of the 3D camera.
- [`Camera2dWideRecordingDesire`](../../ros/utils/t_top_hbba_lite/include/t_top_hbba_lite/Desires.h#L22): To enable the recording of the 2D Wide camera.
- [`RobotNameDetectorDesire`](../../ros/utils/t_top_hbba_lite/include/t_top_hbba_lite/Desires.h#L32): To enable the robot name detector node.
- [`RobotNameDetectorWithLedStatusDesire`](../../ros/utils/t_top_hbba_lite/include/t_top_hbba_lite/Desires.h#L42): To enable the robot name detector node and have feed on the LED strip.
- [`SlowVideoAnalyzer3dDesire`](../../ros/utils/t_top_hbba_lite/include/t_top_hbba_lite/Desires.h#52): To enable the video analyzer node for the 3D camera at 1 Hz.
- [`FastVideoAnalyzer3dDesire`](../../ros/utils/t_top_hbba_lite/include/t_top_hbba_lite/Desires.h#L62): To enable the video analyzer node for the 3D camera at 5 Hz.
- [`FastVideoAnalyzer3dWithAnalyzedImageDesire`](../../ros/utils/t_top_hbba_lite/include/t_top_hbba_lite/Desires.h#L72): To enable the video analyzer node for the 3D camera at 5 Hz and to publish the analyzed image.
- [`SlowVideoAnalyzer2dWideDesire`](../../ros/utils/t_top_hbba_lite/include/t_top_hbba_lite/Desires.h#L82): To enable the video analyzer node for the 2D wide camera at 1 Hz.
- [`FastVideoAnalyzer2dWideDesire`](../../ros/utils/t_top_hbba_lite/include/t_top_hbba_lite/Desires.h#92): To enable the video analyzer node for the 2D wide camera at 5 Hz.
- [`FastVideoAnalyzer2dWideWithAnalyzedImageDesire`](../../ros/utils/t_top_hbba_lite/include/t_top_hbba_lite/Desires.h#L102): To enable the video analyzer node for the 2D wide camera at 5 Hz and to publish the analyzed image.
- [`AudioAnalyzerDesire`](../../ros/utils/t_top_hbba_lite/include/t_top_hbba_lite/Desires.h#L112): To enable the audio analyzer node.
- [`VadDesire`](../../ros/utils/t_top_hbba_lite/include/t_top_hbba_lite/Desires.h#L122): To enable to VAD node.
- [`SpeechToTextDesire`](../../ros/utils/t_top_hbba_lite/include/t_top_hbba_lite/Desires.h#L132): To enable the speech to text node.
- [`ExploreDesire`](../../ros/utils/t_top_hbba_lite/include/t_top_hbba_lite/Desires.h#L142): To enable the explore node.
- [`FaceAnimationDesire`](../../ros/utils/t_top_hbba_lite/include/t_top_hbba_lite/Desires.h#L152): To change the face animation.
- [`LedEmotionDesire`](../../ros/utils/t_top_hbba_lite/include/t_top_hbba_lite/Desires.h#L175): To enable the led emotion node.
- [`LedAnimationDesire`](../../ros/utils/t_top_hbba_lite/include/t_top_hbba_lite/Desires.h#L197): To enable the led animation node.
- [`SoundFollowingDesire`](../../ros/utils/t_top_hbba_lite/include/t_top_hbba_lite/Desires.h#L245): To enable the node that makes T-Top follow the loudest sound.
- [`NearestFaceFollowingDesire`](../../ros/utils/t_top_hbba_lite/include/t_top_hbba_lite/Desires.h#L255): To enable the node that makes T-Top follow the nearest face.
- [`SpecificFaceFollowingDesire`](../../ros/utils/t_top_hbba_lite/include/t_top_hbba_lite/Desires.h#L265): To enable the node that makes T-Top follow a specific face.
- [`SoundObjectPersonFollowingDesire`](../../ros/utils/t_top_hbba_lite/include/t_top_hbba_lite/Desires.h#L284): To enable the node that makes T-Top follow the loudest sound, the people and the objects.
- [`TalkDesire`](../../ros/utils/t_top_hbba_lite/include/t_top_hbba_lite/Desires.h#L294): To make T-Top talk.
- [`GestureDesire`](../../ros/utils/t_top_hbba_lite/include/t_top_hbba_lite/Desires.h#L313): To make T-Top perform a head gesture.
- [`DanceDesire`](../../ros/utils/t_top_hbba_lite/include/t_top_hbba_lite/Desires.h#L335): To make T-Top dance.
- [`PlaySoundDesire`](../../ros/utils/t_top_hbba_lite/include/t_top_hbba_lite/Desires.h#L345): To make T-Top play a sound file.
- [`TelepresenceDesire`](../../ros/utils/t_top_hbba_lite/include/t_top_hbba_lite/Desires.h#L364): To make T-Top perform a video call
- [`TeleoperationDesire`](../../ros/utils/t_top_hbba_lite/include/t_top_hbba_lite/Desires.h#L374): To enable remote control of T-Top.
- [`TooCloseReactionDesire`](../../ros/utils/t_top_hbba_lite/include/t_top_hbba_lite/Desires.h#L384): To enable the too near reaction behavior.

### Strategies
The strategies are declared in the following files: [Strategies.h](../../ros/utils/t_top_hbba_lite/include/t_top_hbba_lite/Strategies.h) and [Strategies.cpp](../../ros/utils/t_top_hbba_lite/src/Strategies.cpp).
