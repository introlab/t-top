#include <hbba_lite/core/DesireSet.h>
#include <hbba_lite/core/GecodeSolver.h>
#include <hbba_lite/core/HbbaLite.h>
#include <hbba_lite/core/RosFilterPool.h>
#include <memory>
#include <opentera_webrtc_ros_msgs/SetString.h>
#include <ros/ros.h>
#include <std_srvs/SetBool.h>
#include <t_top_hbba_lite/Strategies.h>

using namespace std;

constexpr bool WAIT_FOR_SERVICE = true;

class Node
{
public:
    explicit Node(ros::NodeHandle& nodeHandle);
    ~Node() = default;

    static void run() { ros::spin(); }

private:
    ros::NodeHandle& m_nodeHandle;
    std::shared_ptr<DesireSet> m_desireSet;
    std::shared_ptr<RosFilterPool> m_filterPool;

    std::unique_ptr<HbbaLite> m_hbbaLite;

    optional<uint64_t> m_activeMovementModeId;

    ros::ServiceServer m_setMovementModeService;
    ros::ServiceServer m_listActiveStrategiesService;
    ros::ServiceServer m_listActiveDesiresService;

    bool listActiveStrategiesCb(std_srvs::SetBool::Request& request, std_srvs::SetBool::Response& response);
    bool listActiveDesiresCb(std_srvs::SetBool::Request& request, std_srvs::SetBool::Response& response);

    bool setMovementModeCb(
        opentera_webrtc_ros_msgs::SetString::Request& request,
        opentera_webrtc_ros_msgs::SetString::Response& response);

    template<typename MovementMode>
    void setMovementModeDesire();
};

Node::Node(ros::NodeHandle& nodeHandle)
    : m_nodeHandle{nodeHandle},
      m_desireSet{make_shared<DesireSet>()},
      m_filterPool{make_shared<RosFilterPool>(nodeHandle, WAIT_FOR_SERVICE)}
{
    vector<unique_ptr<BaseStrategy>> strategies;
    strategies.emplace_back(createTelepresenceStrategy(m_filterPool));
    strategies.emplace_back(createTeleoperationStrategy(m_filterPool));
    strategies.emplace_back(createSoundFollowingStrategy(m_filterPool));
    strategies.emplace_back(createNearestFaceFollowingStrategy(m_filterPool));

    m_hbbaLite = make_unique<HbbaLite>(
        m_desireSet,
        move(strategies),
        std::unordered_map<std::string, uint16_t>{{"sound", 1}},
        make_unique<GecodeSolver>());

    auto telepresenceDesire = make_unique<TelepresenceDesire>();
    m_desireSet->addDesire(std::move(telepresenceDesire));

    setMovementModeDesire<TeleoperationDesire>();

    m_setMovementModeService = m_nodeHandle.advertiseService("set_movement_mode", &Node::setMovementModeCb, this);

    m_listActiveStrategiesService =
        m_nodeHandle.advertiseService("hbba/list_active_strategies", &Node::listActiveStrategiesCb, this);
    m_listActiveDesiresService =
        m_nodeHandle.advertiseService("hbba/list_active_desires", &Node::listActiveDesiresCb, this);
};

bool Node::listActiveStrategiesCb(std_srvs::SetBool::Request& request, std_srvs::SetBool::Response& response)
{
    response.message = "";
    response.success = true;

    for (const auto& strategy : m_hbbaLite->getActiveStrategies())
    {
        response.message += strategy + ";";
    }

    return true;
}
bool Node::listActiveDesiresCb(std_srvs::SetBool::Request& request, std_srvs::SetBool::Response& response)
{
    response.message = "";
    response.success = true;

    for (const auto& desire : m_hbbaLite->getActiveDesireNames())
    {
        response.message += desire + ";";
    }

    return true;
}

bool Node::setMovementModeCb(
    opentera_webrtc_ros_msgs::SetString::Request& request,
    opentera_webrtc_ros_msgs::SetString::Response& response)
{
    response.message = "";
    response.success = true;

    if (request.data == "teleop")
    {
        setMovementModeDesire<TeleoperationDesire>();
    }
    else if (request.data == "sound")
    {
        setMovementModeDesire<SoundFollowingDesire>();
    }
    else if (request.data == "face")
    {
        setMovementModeDesire<NearestFaceFollowingDesire>();
    }
    else
    {
        response.success = false;
        response.message = "Unknown movement mode";
    }

    return true;
}

template<typename MovementMode>
void Node::setMovementModeDesire()
{
    auto transaction = m_desireSet->beginTransaction();
    if (m_activeMovementModeId && m_desireSet->contains(m_activeMovementModeId.value()))
    {
        m_desireSet->removeDesire(m_activeMovementModeId.value());
    }

    auto movementModeDesire = make_unique<MovementMode>();
    m_activeMovementModeId = movementModeDesire->id();
    m_desireSet->addDesire(std::move(movementModeDesire));
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "t_top_opentera_node");
    ros::NodeHandle nodeHandle;
    ros::NodeHandle privateNodeHandle("~");

    Node node{nodeHandle};
    Node::run();

    return 0;
}
