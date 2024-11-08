#include <hbba_lite/core/DesireSet.h>
#include <hbba_lite/core/GecodeSolver.h>
#include <hbba_lite/core/HbbaLite.h>
#include <hbba_lite/core/RosFilterPool.h>
#include <memory>
#include <opentera_webrtc_ros_msgs/srv/set_string.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <t_top_hbba_lite/Strategies.h>
#include <optional>

#include "utils.h"

constexpr bool WAIT_FOR_SERVICE = true;

class Node
{
public:
    Node();

    void run()
    {
        rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);
        executor.add_node(m_node);
        executor.spin();
    }

    rclcpp::Logger get_logger() const { return m_node->get_logger(); }

private:
    std::shared_ptr<rclcpp::Node> m_node;
    std::shared_ptr<DesireSet> m_desireSet;
    std::shared_ptr<RosFilterPool> m_filterPool;

    std::unique_ptr<HbbaLite> m_hbbaLite;

    std::optional<uint64_t> m_activeMovementModeId;

    rclcpp::Service<opentera_webrtc_ros_msgs::srv::SetString>::SharedPtr m_setMovementModeService;
    rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr m_listActiveStrategiesService;
    rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr m_listActiveDesiresService;

    void listActiveStrategiesCb(
        const std_srvs::srv::SetBool::Request::ConstSharedPtr& request,
        const std_srvs::srv::SetBool::Response::SharedPtr& response);
    void listActiveDesiresCb(
        const std_srvs::srv::SetBool::Request::ConstSharedPtr& request,
        const std_srvs::srv::SetBool::Response::SharedPtr& response);

    void setMovementModeCb(
        const opentera_webrtc_ros_msgs::srv::SetString::Request::ConstSharedPtr& request,
        const opentera_webrtc_ros_msgs::srv::SetString::Response::SharedPtr& response);

    template<typename MovementMode>
    void setMovementModeDesire();

    std::vector<std::unique_ptr<BaseStrategy>> makeStrategies();
};

std::vector<std::unique_ptr<BaseStrategy>> Node::makeStrategies()
{
    std::vector<std::unique_ptr<BaseStrategy>> strategies;

    strategies.emplace_back(createTelepresenceStrategy(m_filterPool));
    strategies.emplace_back(createTeleoperationStrategy(m_filterPool));
    strategies.emplace_back(createSoundFollowingStrategy(m_filterPool));
    strategies.emplace_back(createNearestFaceFollowingStrategy(m_filterPool));

    return strategies;
}

Node::Node()
    : m_node{std::make_shared<rclcpp::Node>("t_top_opentera_node")},
      m_desireSet{std::make_shared<DesireSet>()},
      m_filterPool{std::make_shared<RosFilterPool>(m_node, WAIT_FOR_SERVICE)},
      m_hbbaLite{std::make_unique<HbbaLite>(
          m_desireSet,
          makeStrategies(),
          std::unordered_map<std::string, uint16_t>{{"sound", 1}},
          std::make_unique<GecodeSolver>())},
      m_activeMovementModeId{},
      m_setMovementModeService{m_node->create_service<opentera_webrtc_ros_msgs::srv::SetString>(
          "set_movement_mode",
          bind_this<opentera_webrtc_ros_msgs::srv::SetString>(this, &Node::setMovementModeCb))},
      m_listActiveStrategiesService{m_node->create_service<std_srvs::srv::SetBool>(
          "hbba/list_active_strategies",
          bind_this<std_srvs::srv::SetBool>(this, &Node::listActiveStrategiesCb))},
      m_listActiveDesiresService{m_node->create_service<std_srvs::srv::SetBool>(
          "hbba/list_active_desires",
          bind_this<std_srvs::srv::SetBool>(this, &Node::listActiveDesiresCb))}
{
    auto telepresenceDesire = std::make_unique<TelepresenceDesire>();
    m_desireSet->addDesire(std::move(telepresenceDesire));

    setMovementModeDesire<TeleoperationDesire>();
}

void Node::listActiveStrategiesCb(
    [[maybe_unused]] const std_srvs::srv::SetBool::Request::ConstSharedPtr& request,
    const std_srvs::srv::SetBool::Response::SharedPtr& response)
{
    response->message = "";
    response->success = true;

    for (const auto& strategy : m_hbbaLite->getActiveStrategies())
    {
        response->message += strategy + ";";
    }
}
void Node::listActiveDesiresCb(
    [[maybe_unused]] const std_srvs::srv::SetBool::Request::ConstSharedPtr& request,
    const std_srvs::srv::SetBool::Response::SharedPtr& response)
{
    response->message = "";
    response->success = true;

    for (const auto& desire : m_hbbaLite->getActiveDesireNames())
    {
        response->message += desire + ";";
    }
}

void Node::setMovementModeCb(
    const opentera_webrtc_ros_msgs::srv::SetString::Request::ConstSharedPtr& request,
    const opentera_webrtc_ros_msgs::srv::SetString::Response::SharedPtr& response)
{
    response->success = false;
    response->message = "Unknown movement mode";

    const auto set_success = [&response]()
    {
        response->message = "";
        response->success = true;
    };

    if (request->data == "teleop")
    {
        setMovementModeDesire<TeleoperationDesire>();
        set_success();
    }
    else if (request->data == "sound")
    {
        setMovementModeDesire<SoundFollowingDesire>();
        set_success();
    }
    else if (request->data == "face")
    {
        setMovementModeDesire<NearestFaceFollowingDesire>();
        set_success();
    }
}

template<typename MovementMode>
void Node::setMovementModeDesire()
{
    auto transaction = m_desireSet->beginTransaction();
    if (m_activeMovementModeId && m_desireSet->contains(m_activeMovementModeId.value()))
    {
        m_desireSet->removeDesire(m_activeMovementModeId.value());
    }

    auto movementModeDesire = std::make_unique<MovementMode>();
    m_activeMovementModeId = movementModeDesire->id();
    m_desireSet->addDesire(std::move(movementModeDesire));
}

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    Node node;

    try
    {
        node.run();
    }
    catch (const std::exception& e)
    {
        RCLCPP_ERROR_STREAM(node.get_logger(), "T-Top OpenTera Node crashed (" << e.what() << ")");
    }

    return 0;
}
