#ifndef HBBA_LITE_CORE_DESIRE_SET_H
#define HBBA_LITE_CORE_DESIRE_SET_H

#include <hbba_lite/utils/ClassMacros.h>
#include <hbba_lite/core/Desire.h>

#include <memory>
#include <unordered_map>
#include <mutex>
#include <vector>
#include <unordered_set>

class DesireSet;

class DesireSetTransaction
{
    DesireSet& m_desireSet;
    std::unique_lock<std::recursive_mutex> m_lock;

private:
    DesireSetTransaction(DesireSet& desireSet, std::unique_lock<std::recursive_mutex>&& lock);
    DECLARE_NOT_COPYABLE(DesireSetTransaction);

public:
    DesireSetTransaction(DesireSetTransaction&&) = default;
    ~DesireSetTransaction();

    friend DesireSet;
};

class DesireSetObserver
{
public:
    virtual void onDesireSetChanged(const std::vector<std::unique_ptr<Desire>>& enabledDesires) = 0;
};

class DesireSet
{
    std::unordered_map<uint64_t, std::unique_ptr<Desire>> m_desiresById;
    std::unordered_set<DesireSetObserver*> m_observers;

    std::mutex m_observerMutex;
    std::recursive_mutex m_desireMutex;
    bool m_isTransactionStarted;
    bool m_hasChanged;

public:
    DesireSet();
    DECLARE_NOT_COPYABLE(DesireSet);
    DECLARE_NOT_MOVABLE(DesireSet);

    void addObserver(DesireSetObserver* observer);
    void removeObserver(DesireSetObserver* observer);

    DesireSetTransaction beginTransaction();

    void addDesire(std::unique_ptr<Desire>&& desire);
    void removeDesire(uint64_t id);
    void removeDesires(std::type_index type);
    bool contains(uint64_t id);
    void clear();

    void enableAllDesires();
    void disableAllDesires();

private:
    void endTransaction(std::unique_lock<std::recursive_mutex> lock);
    void callObservers(std::unique_lock<std::recursive_mutex> desireLock);

    std::vector<std::unique_ptr<Desire>> getEnabledDesires();
    friend DesireSetTransaction;
};

#endif
