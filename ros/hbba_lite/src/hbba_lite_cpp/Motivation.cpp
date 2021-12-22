#include <hbba_lite/Motivation.h>

using namespace std;

Motivation::Motivation(shared_ptr<DesireSet> desireSet) : m_desireSet(move(desireSet))
{
}
