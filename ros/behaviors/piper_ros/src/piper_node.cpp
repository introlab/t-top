#include <piper_ros/srv/generate_speech_from_text.hpp>

#include <rclcpp/rclcpp.hpp>

#include <ament_index_cpp/get_package_share_directory.hpp>

#include <piper.hpp>

#include <optional>

enum class Language
{
    ENGLISH,
    FRENCH
};

std::optional<Language> languageFromString(const std::string& str)
{
    if (str == "en")
    {
        return Language::ENGLISH;
    }
    else if (str == "fr")
    {
        return Language::FRENCH;
    }
    else
    {
        return std::nullopt;
    }
}


enum class Gender
{
    FEMALE,
    MALE
};

std::optional<Gender> genderFromString(const std::string& str)
{
    if (str == "female")
    {
        return Gender::FEMALE;
    }
    else if (str == "male")
    {
        return Gender::MALE;
    }
    else
    {
        return std::nullopt;
    }
}


class PiperNode : public rclcpp::Node
{
    bool m_useGpuIfAvailable;

    rclcpp::Service<piper_ros::srv::GenerateSpeechFromText>::SharedPtr m_generateSpeechFromTextService;

    piper::PiperConfig m_piperConfig;
    piper::Voice m_englishFemaleVoice;
    piper::Voice m_englishMaleVoice;
    piper::Voice m_frenchFemaleVoice;
    piper::Voice m_frenchMaleVoice;

public:
    explicit PiperNode() : rclcpp::Node("piper_node")
    {
        m_useGpuIfAvailable = declare_parameter("use_gpu_if_available", false);

        m_piperConfig.useESpeak = true;
        m_piperConfig.eSpeakDataPath = ESPEAK_NG_DATA_PATH;

        m_englishFemaleVoice = loadVoice("en_US-amy-low");
        m_englishMaleVoice = loadVoice("en_US-ryan-low");
        m_frenchFemaleVoice = loadVoice("fr_FR-siwis-low");
        m_frenchMaleVoice = loadVoice("fr_FR-gilles-low");

        piper::initialize(m_piperConfig);

        m_generateSpeechFromTextService = create_service<piper_ros::srv::GenerateSpeechFromText>(
            "piper/generate_speech_from_text",
                [this] (
                const std::shared_ptr<piper_ros::srv::GenerateSpeechFromText::Request> request,
                std::shared_ptr<piper_ros::srv::GenerateSpeechFromText::Response> response)
                {
                    generateSpeechFromTextServiceCallback(request, response);
                });
    }

    void run() { rclcpp::spin(shared_from_this()); }

private:
    piper::Voice loadVoice(const char* filename)
    {
        std::string modelFolder = ament_index_cpp::get_package_share_directory("piper_ros") + "/models/";

        piper::Voice voice;
        voice.session.options.SetInterOpNumThreads(1);
        voice.session.options.SetIntraOpNumThreads(1);
        voice.session.options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        voice.session.options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

#ifdef ONNXRUNTIME_CUDA_PROVIDER_ENABLED
        if (m_useGpuIfAvailable)
        {
            OrtCUDAProviderOptions cudaOptions;
            cudaOptions.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;
            voice.session.options.AppendExecutionProvider_CUDA(cudaOptions);
        }
#else
        if (m_useGpuIfAvailable)
        {
            RCLCPP_WARN(get_logger(), "CUDA is not supported.");
        }
#endif

        std::optional<piper::SpeakerId> speakerId;
        piper::loadVoice(
            m_piperConfig,
            modelFolder + filename + ".onnx",
            modelFolder + filename + ".onnx.json",
            voice,
            speakerId);

        return voice;
    }

    void generateSpeechFromTextServiceCallback(
        const std::shared_ptr<piper_ros::srv::GenerateSpeechFromText::Request> request,
        std::shared_ptr<piper_ros::srv::GenerateSpeechFromText::Response> response)
    {
        std::optional<Language> language = languageFromString(request->language);
        if (language == std::nullopt)
        {
            response->ok = false;
            response->message = "Invalid language";
        }

        std::optional<Gender> gender = genderFromString(request->gender);
        if (gender == std::nullopt)
        {
            response->ok = false;
            response->message = "Invalid gender";
        }

        generateSpeechFromText(*language, *gender, request->length_scale, request->text, request->path);
        response->ok = true;
    }

    void generateSpeechFromText(
        Language language,
        Gender gender,
        float lengthScale,
        const std::string& text,
        const std::string& path)
    {
        piper::Voice& voice = voiceFromLanguageAndGender(language, gender);
        voice.synthesisConfig.lengthScale = lengthScale;

        std::ofstream audioFile(path, std::ios::binary);
        piper::SynthesisResult result = {0};

        piper::textToWavFile(m_piperConfig, voice, text, audioFile, result);
    }

    piper::Voice& voiceFromLanguageAndGender(Language language, Gender gender)
    {
        if (language == Language::ENGLISH && gender == Gender::FEMALE)
        {
            return m_englishFemaleVoice;
        }
        else if (language == Language::ENGLISH && gender == Gender::MALE)
        {
            return m_englishMaleVoice;
        }
        else if (language == Language::FRENCH && gender == Gender::FEMALE)
        {
            return m_frenchFemaleVoice;
        }
        else if (language == Language::FRENCH && gender == Gender::MALE)
        {
            return m_frenchMaleVoice;
        }
        else
        {
            throw std::runtime_error("Invalid language and/or gender");
        }
    }
};


int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    auto node = std::make_shared<PiperNode>();
    node->run();

    rclcpp::shutdown();

    return 0;
}
