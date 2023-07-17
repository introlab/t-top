#include "RrcUtils.h"

void initRandom()
{
    randomSeed(analogRead(12));
}

void smBusRandomDelay()
{
    constexpr int MIN_TRIAL_DELAY_US = 500;
    constexpr int MAX_TRIAL_DELAY_US = 2000;

    delayMicroseconds(random(MIN_TRIAL_DELAY_US, MAX_TRIAL_DELAY_US));
}
