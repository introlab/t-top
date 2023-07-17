#ifndef _PROCESS_UTILS_H_
#define _PROCESS_UTILS_H_

#include <cstdint>
#include <QString>
#include <QList>

#ifdef __linux__

QList<uint32_t> listPidsMatchingTheCriteria(const QString& searchedText);
void shutdownProcessesAndWait(const QList<uint32_t>& pids, qint64 timeoutSec);

#endif

#endif  //_PROCESS_UTILS_H_
