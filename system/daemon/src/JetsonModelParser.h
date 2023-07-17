#ifndef _JETSON_MODEL_PARSER_H_
#define _JETSON_MODEL_PARSER_H_

#include <string>

enum class JetsonModel
{
    XAVIER,
    ORIN,
    UNKNOWN
};

JetsonModel get_jetson_model();

std::string get_jetson_model_name(JetsonModel model);

#endif  // _JETSON_MODEL_PARSER_H_
