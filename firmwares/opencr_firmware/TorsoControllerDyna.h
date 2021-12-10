#ifndef TORSO_CONTROLLER_DYNA_H
#define TORSO_CONTROLLER_DYNA_H

#include <DynamixelWorkbench.h>

class TorsoControllerDyna {
  DynamixelWorkbench& m_dynamixelWorkbench;

  bool m_isZeroOffsetFound;
  float m_zeroOffset;

public:
  TorsoControllerDyna(DynamixelWorkbench& dynamixelWorkbench);
  ~TorsoControllerDyna();

  void init();

  void setOrientation(float orientation);
  float readOrientation();

private:
  void findZeroOffset();
  float getOrientationFromDynamixelPosition(float dynamixelPosition);
  float getNewDynamixelPositionFromOrientationDelta(float dynamixelPosition, float orientationDelta);
};

#endif

