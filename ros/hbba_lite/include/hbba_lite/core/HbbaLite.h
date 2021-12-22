#ifndef HBBA_LITE_CORE_HBBA_LITE_H
#define HBBA_LITE_CORE_HBBA_LITE_H

#include <hbba_lite/utils/ClassMacros.h>
#include <hbba_lite/utils/BinarySemaphore.h>

#include <hbba_lite/core/DesireSet.h>
#include <hbba_lite/core/Strategy.h>
#include <hbba_lite/core/Solver.h>

#include <atomic>
#include <unordered_map>
#include <memory>
#include <vector>
#include <typeindex>
#include <thread>
#include <mutex>

class HbbaLite : public DesireSetObserver
{
    std::shared_ptr<DesireSet> m_desireSet;
    std::unordered_map<std::type_index, std::vector<std::unique_ptr<BaseStrategy>>> m_strategiesByDesireType;
    std::unordered_map<std::string, uint16_t> m_resourcesByNames;
    std::unique_ptr<Solver> m_solver;

    std::mutex m_pendingDesiresMutex;
    BinarySemaphore m_pendingDesiresSemaphore;
    std::vector<std::unique_ptr<Desire>> m_pendingDesires;

    std::atomic_bool m_stopped;
    std::unique_ptr<std::thread> m_thread;

public:
    HbbaLite(std::shared_ptr<DesireSet> desireSet,
        std::vector<std::unique_ptr<BaseStrategy>> strategies,
        std::unordered_map<std::string, uint16_t> resourcesByNames,
        std::unique_ptr<Solver> solver);
    ~HbbaLite();

    DECLARE_NOT_COPYABLE(HbbaLite);
    DECLARE_NOT_MOVABLE(HbbaLite);

    void onDesireSetChanged(const std::vector<std::unique_ptr<Desire>>& enabledDesires) override;

private:
    void checkStrategyResources(std::type_index desireType, const std::unordered_map<std::string, uint16_t>& resourcesByNames);

    void run();
    void updateStrategies(std::vector<std::unique_ptr<Desire>> desires);
};

#endif
