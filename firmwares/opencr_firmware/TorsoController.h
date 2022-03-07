#ifndef TORSO_CONTROLLER_DYNA_H
#define TORSO_CONTROLLER_DYNA_H

#include <DynamixelWorkbench.h>

class TorsoController {
  DynamixelWorkbench& m_dynamixelWorkbench;

  bool m_isZeroOffsetFound;
  float m_zeroOffset;

public:
  TorsoController(DynamixelWorkbench& dynamixelWorkbench);
  ~TorsoController();

  void init();

  void setOrientation(float orientation);
  float readOrientation();

private:
  void findZeroOffset();
  float getOrientationFromDynamixelPosition(float dynamixelPosition);
  float getNewDynamixelPositionFromOrientationDelta(float dynamixelPosition, float orientationDelta);
};

#endif
