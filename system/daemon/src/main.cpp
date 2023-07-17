#include "DaemonApp.h"

int main(int argc, char* argv[])
{
    DaemonApp app(argc, argv);
    return app.exec();
}
