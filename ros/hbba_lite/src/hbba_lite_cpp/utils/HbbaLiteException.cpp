#include <hbba_lite/utils/HbbaLiteException.h>

using namespace std;

HbbaLiteException::HbbaLiteException(const string& message) : runtime_error(message) {}
