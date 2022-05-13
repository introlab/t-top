#ifndef EGO_NOISE_REDUCTION_MATH_H
#define EGO_NOISE_REDUCTION_MATH_H

#include <MusicBeatDetector/Utils/Exception/NotSupportedException.h>

#include <armadillo>

template<class T>
inline T hann(std::size_t length)
{
    if (length == 0)
    {
        THROW_NOT_SUPPORTED_EXCEPTION("The length must be greater than 0.");
    }

    const std::size_t N = length - 1;
    T n = arma::regspace<T>(0, N);
    return 0.5 - 0.5 * arma::cos(2 * M_PI * n / N);
}

#endif
