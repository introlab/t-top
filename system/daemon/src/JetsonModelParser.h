#ifndef _JETSON_MODEL_PARSER_H_
#define _JETSON_MODEL_PARSER_H_

enum class JetsonModel
{
    XAVIER,
    ORIN,
    UNKNOWN
};

JetsonModel get_jetson_model();

#endif  // _JETSON_MODEL_PARSER_H_
