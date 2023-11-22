
#include "DaemonRosClientNode.h"
#include <signal.h>
#include <unistd.h>

void catchUnixSignals(std::initializer_list<int> quitSignals)
{
    auto handler = [](int sig) -> void { QCoreApplication::quit(); };

    sigset_t blockingMask;
    sigemptyset(&blockingMask);
    for (auto sig : quitSignals)
    {
        sigaddset(&blockingMask, sig);
    }

    struct sigaction sa;
    sa.sa_handler = handler;
    sa.sa_mask = blockingMask;
    sa.sa_flags = 0;

    for (auto sig : quitSignals)
    {
        sigaction(sig, &sa, nullptr);
    }
}


int main(int argc, char** argv)
{
    ros::init(argc, argv, "daemon_ros_client_node");

    ros::NodeHandle nodeHandle;
    ros::NodeHandle privateNodeHandle("~");

    DaemonRosClientNodeConfiguration configuration;

    if (!privateNodeHandle.getParam("base_link_torso_base_delta_z", configuration.baseLinkTorsoBaseDeltaZ))
    {
        ROS_ERROR("The parameter base_link_torso_base_delta_z is required.");
        return -1;
    }

    // Run ROS in background
    ros::AsyncSpinner spinner(1);
    spinner.start();

    catchUnixSignals({SIGQUIT, SIGINT, SIGTERM, SIGHUP});

    // Initialize and start Qt App
    DaemonRosClientNode app(argc, argv, nodeHandle, configuration);
    app.exec();

    app.cleanup();

    // Stop ROS spinner
    spinner.stop();
}
