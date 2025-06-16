#ifndef CHATBOT__TOOLS__CHATBOT_TOOLS_HPP
#define CHATBOT__TOOLS__CHATBOT_TOOLS_HPP

#include <rclcpp/rclcpp.hpp>
#include <behavior_srvs/srv/chat_tools_function_call.hpp>
#include <daemon_ros_client/msg/base_status.hpp>
#include <std_msgs/msg/u_int8.hpp>
#include <cloud_data/srv/current_local_weather_open_meteo.hpp>
#include <cloud_data/srv/local_weather_forecast_open_meteo.hpp>
#include <perception_msgs/srv/perceive_objects.hpp>

class ChatbotTools
{
public:
    ChatbotTools(std::shared_ptr<rclcpp::Node> node, rclcpp::CallbackGroup::SharedPtr callbackGroup);
    ~ChatbotTools() = default;

private:
    rclcpp::Node::SharedPtr node_;
    rclcpp::CallbackGroup::SharedPtr callbackGroup_;

    rclcpp::Service<behavior_srvs::srv::ChatToolsFunctionCall>::SharedPtr service_volume_up_;
    rclcpp::Service<behavior_srvs::srv::ChatToolsFunctionCall>::SharedPtr service_volume_down_;
    rclcpp::Service<behavior_srvs::srv::ChatToolsFunctionCall>::SharedPtr service_get_weather_;
    rclcpp::Service<behavior_srvs::srv::ChatToolsFunctionCall>::SharedPtr service_get_forecast_;
    rclcpp::Service<behavior_srvs::srv::ChatToolsFunctionCall>::SharedPtr service_get_perceived_objects_;
    rclcpp::Service<behavior_srvs::srv::ChatToolsFunctionCall>::SharedPtr service_get_date_;

    rclcpp::Client<cloud_data::srv::CurrentLocalWeatherOpenMeteo>::SharedPtr weather_client_;
    rclcpp::Client<cloud_data::srv::LocalWeatherForecastOpenMeteo>::SharedPtr forecast_client_;
    rclcpp::Client<perception_msgs::srv::PerceiveObjects>::SharedPtr perceive_objects_client_;

    rclcpp::Publisher<std_msgs::msg::UInt8>::SharedPtr volume_publisher_;
    rclcpp::Subscription<daemon_ros_client::msg::BaseStatus>::SharedPtr base_status_subscriber_;

    daemon_ros_client::msg::BaseStatus::SharedPtr base_status_msg_;

    void on_base_status_(const daemon_ros_client::msg::BaseStatus::SharedPtr msg);

    void publish_volume(uint8_t volume);

    void handle_volume_up_request(
        const std::shared_ptr<behavior_srvs::srv::ChatToolsFunctionCall::Request> request,
        const std::shared_ptr<behavior_srvs::srv::ChatToolsFunctionCall::Response> response);
    void handle_volume_down_request(
        const std::shared_ptr<behavior_srvs::srv::ChatToolsFunctionCall::Request> request,
        const std::shared_ptr<behavior_srvs::srv::ChatToolsFunctionCall::Response> response);
    void handle_get_weather_request(
        const std::shared_ptr<behavior_srvs::srv::ChatToolsFunctionCall::Request>,
        const std::shared_ptr<behavior_srvs::srv::ChatToolsFunctionCall::Response> response);
    void handle_get_forecast_request(
        const std::shared_ptr<behavior_srvs::srv::ChatToolsFunctionCall::Request> request,
        const std::shared_ptr<behavior_srvs::srv::ChatToolsFunctionCall::Response> response);
    void handle_perceive_objects_request(
        const std::shared_ptr<behavior_srvs::srv::ChatToolsFunctionCall::Request>,
        const std::shared_ptr<behavior_srvs::srv::ChatToolsFunctionCall::Response> response);
    void handle_get_date_request(
        const std::shared_ptr<behavior_srvs::srv::ChatToolsFunctionCall::Request>,
        const std::shared_ptr<behavior_srvs::srv::ChatToolsFunctionCall::Response> response);
};

#endif
