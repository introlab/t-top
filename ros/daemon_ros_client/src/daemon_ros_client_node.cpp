
#include "DaemonRosClientNode.h"

#include <QCoreApplication>

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
    QCoreApplication app(argc, argv);

    rclcpp::init(argc, argv);

    auto node = std::make_shared<DaemonRosClientNode>();

    catchUnixSignals({SIGQUIT, SIGINT, SIGTERM, SIGHUP});

    // Run ROS in background
    rclcpp::executors::SingleThreadedExecutor rosExecutor;
    rosExecutor.add_node(node);
    std::thread spinThread([&rosExecutor]() { rosExecutor.spin(); });

    // Initialize and start Qt App
    app.exec();

    node->cleanup();
    rosExecutor.cancel();
    spinThread.join();

    rclcpp::shutdown();
}
