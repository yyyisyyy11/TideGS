"""Export real CUPTI execution intervals and host waits for one process/window."""

import csv
import sqlite3
import sys

path, pid, begin, finish = sys.argv[1:]
pid, begin, finish = int(pid), int(begin), int(finish)
conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
assert conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"
processes = conn.execute("SELECT DISTINCT globalPid FROM PROCESSES WHERE pid=?", (pid,)).fetchall()
assert len(processes) == 1, processes
global_pid = processes[0][0]
main_tid = global_pid | pid
copy_types = dict(conn.execute("SELECT id, label FROM ENUM_CUDA_MEMCPY_OPER"))
records = []

for start, end, device, stream, name, correlation in conn.execute(
    "SELECT k.start,k.end,k.deviceId,k.streamId,s.value,k.correlationId "
    "FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName=s.id "
    "WHERE k.globalPid=? AND k.end>? AND k.start<? AND k.end>k.start",
    (global_pid, begin, finish),
):
    category = "nccl" if "nccl" in name.lower() else "compute"
    records.append((start, end, "GPU", category, name, device, stream, 0, correlation))
for start, end, device, stream, size, kind, correlation in conn.execute(
    "SELECT start,end,deviceId,streamId,bytes,copyKind,correlationId "
    "FROM CUPTI_ACTIVITY_KIND_MEMCPY "
    "WHERE globalPid=? AND end>? AND start<? AND end>start",
    (global_pid, begin, finish),
):
    records.append((start, end, "GPU", "copy", copy_types[kind], device, stream, size, correlation))
for start, end, device, stream, size, correlation in conn.execute(
    "SELECT start,end,deviceId,streamId,bytes,correlationId "
    "FROM CUPTI_ACTIVITY_KIND_MEMSET "
    "WHERE globalPid=? AND end>? AND start<? AND end>start",
    (global_pid, begin, finish),
):
    records.append((start, end, "GPU", "compute", "CUDA memset", device, stream, size, correlation))
for start, end, name, correlation, tid, rowid in conn.execute(
    "SELECT r.start,r.end,s.value,r.correlationId,r.globalTid,r.rowid "
    "FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId=s.id "
    "WHERE (r.globalTid & -16777216)=? AND r.end>? AND r.start<? "
    "AND r.end-r.start>=100000 AND s.value LIKE '%Synchronize%'",
    (global_pid, begin, finish),
):
    records.append((start, end, "CPU", "wait", name, -1, -1, 0, correlation, tid, rowid,
                    "main" if tid == main_tid else "worker"))
for start, end, name, tid, rowid in conn.execute(
    "SELECT r.start,r.end,s.value,r.globalTid,r.rowid FROM OSRT_API r JOIN StringIds s ON r.nameId=s.id "
    "WHERE (r.globalTid & -16777216)=? AND r.end>? AND r.start<? AND r.end-r.start>=100000 "
    "AND (s.value LIKE '%wait%' OR s.value LIKE '%poll%')",
    (global_pid, begin, finish),
):
    records.append((start, end, "CPU", "os_wait", name, -1, -1, 0, -1, tid, rowid,
                    "main" if tid == main_tid else "worker"))
records = [row if len(row) > 9 else (*row, None, None, None) for row in records]
records.sort(key=lambda row: (row[0], row[1], row[2], row[3]))
assert records, "No CUDA events in selected window"
writer = csv.writer(sys.stdout)
writer.writerow(["start_ns", "end_ns", "lane", "category", "name", "device_id", "stream_id", "bytes", "correlation_id", "global_tid", "source_rowid", "thread_role"])
writer.writerows(records)
