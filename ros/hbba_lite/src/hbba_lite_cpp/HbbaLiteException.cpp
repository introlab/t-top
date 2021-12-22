#include <hbba_lite/HbbaLiteException.h>

using namespace std;

HbbaLiteException::HbbaLiteException(const string& message) : runtime_error(message)
{
}
