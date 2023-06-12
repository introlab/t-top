#include "ProcessUtils.h"
#include <QDir>
#include <QFile>
#include <QTextStream>
#include <QProcess>
#include <QElapsedTimer>
#include <QThread>


#ifdef __linux__

QList<uint32_t> listPidsMatchingTheCriteria(const QString& searchedText)
{
    QList<uint32_t> pids;
    QDir procDir("/proc");

    for (const QString& processIdStr : procDir.entryList(QDir::Dirs))
    {
        QFile f("/proc/" + processIdStr + "/cmdline");
        if (!f.open(QFile::ReadOnly | QFile::Text))
        {
            continue;
        }

        QTextStream in(&f);
        const QString& cmdline = in.readAll();

        if (cmdline.contains(searchedText))
        {
            pids.append(processIdStr.toULong());
        }
    }

    return pids;
}

bool anyProcessExists(const QList<uint32_t>& pids)
{
    for (uint32_t pid : pids)
    {
        QDir dir("/proc/" + QString::number(pid));
        if (dir.exists())
        {
            return true;
        }
    }

    return false;
}

void shutdownProcessesAndWait(const QList<uint32_t>& pids, qint64 timeoutSec)
{
    QElapsedTimer timer;
    timer.start();

    for (uint32_t pid : pids)
    {
        QProcess::startDetached("kill", {"-s", "SIGINT", QString::number(pid)});
    }

    while (true)
    {
        if (timer.hasExpired(timeoutSec * 1000u) || !anyProcessExists(pids))
        {
            break;
        }

        QThread::sleep(1);  // Sleep one sec
    }
}

#endif
