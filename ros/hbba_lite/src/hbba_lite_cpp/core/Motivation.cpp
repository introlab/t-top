#include <hbba_lite/core/Motivation.h>

using namespace std;

Motivation::Motivation(shared_ptr<DesireSet> desireSet) : m_desireSet(move(desireSet)) {}
