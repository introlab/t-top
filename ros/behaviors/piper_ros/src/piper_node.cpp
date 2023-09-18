#include <piper_ros/GenerateSpeechFromText.h>

#include <ros/ros.h>
#include <ros/package.h>

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


class PiperNode
{
    ros::NodeHandle m_nodeHandle;
    ros::ServiceServer m_generateSpeechFromTextService;

    piper::PiperConfig m_piperConfig;
    piper::Voice m_englishFemaleVoice;
    piper::Voice m_englishMaleVoice;
    piper::Voice m_frenchFemaleVoice;
    piper::Voice m_frenchMaleVoice;

public:
    PiperNode()
    {
        m_piperConfig.useESpeak = true;
        m_piperConfig.eSpeakDataPath = ESPEAK_NG_DATA_PATH;

        m_englishFemaleVoice = loadVoiceFromLanguageAndGender("en_US-amy-low");
        m_englishMaleVoice = loadVoiceFromLanguageAndGender("en_US-ryan-low");
        m_frenchFemaleVoice = loadVoiceFromLanguageAndGender("fr_FR-siwis-low");
        m_frenchMaleVoice = loadVoiceFromLanguageAndGender("fr_FR-gilles-low");

        piper::initialize(m_piperConfig);


        m_generateSpeechFromTextService = m_nodeHandle.advertiseService("piper/generate_speech_from_text",
            &PiperNode::generateSpeechFromTextServiceCallback,
            this);
    }

    void run()
    {
        ros::spin();
    }

private:
    piper::Voice loadVoiceFromLanguageAndGender(const char* filename)
    {
        std::string modelFolder = ros::package::getPath("piper_ros") + "/models/";

        piper::Voice voice;
        voice.session.options.SetInterOpNumThreads(1);
        voice.session.options.SetIntraOpNumThreads(1);
        voice.session.options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        voice.session.options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

#ifdef ONNXRUNTIME_CUDA_PROVIDER_ENABLED
        OrtCUDAProviderOptions cudaOptions;
        cudaOptions.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;
        voice.session.options.AppendExecutionProvider_CUDA(cudaOptions);
#endif

        std::optional<piper::SpeakerId> speakerId;
        loadVoice(m_piperConfig, modelFolder + filename + ".onnx", modelFolder + filename + ".onnx.json", voice, speakerId);

        return voice;
    }

    bool generateSpeechFromTextServiceCallback(piper_ros::GenerateSpeechFromText::Request& request,
        piper_ros::GenerateSpeechFromText::Response& response)
    {
        std::optional<Language> language = languageFromString(request.language);
        if (language == std::nullopt)
        {
            response.ok = false;
            response.message = "Invalid language";
            return true;
        }

        std::optional<Gender> gender = genderFromString(request.gender);
        if (gender == std::nullopt)
        {
            response.ok = false;
            response.message = "Invalid gender";
            return true;
        }

        generateSpeechFromText(*language, *gender, request.length_scale, request.text, request.path);
        response.ok = true;

        return true;
    }

    void generateSpeechFromText(Language language,
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


int main(int argc, char **argv)
{
    ros::init(argc, argv, "piper_node");

    PiperNode node;
    node.run();

    return 0;
}
