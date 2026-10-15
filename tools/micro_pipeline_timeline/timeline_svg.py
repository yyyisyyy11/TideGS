"""Render verified four/eight-GPU data with the same timeline semantics."""

from xml.etree.ElementTree import Element, SubElement, tostring


# Match nsys-4gpu-three-steady-batches.csv.html across HTML and PNG.
COLORS = {
    "compute": "hsl(43 65% 52%)",
    "nccl": "hsl(86 65% 52%)",
    "copy": "hsl(129 65% 52%)",
    "preview": "hsl(215 65% 52%)",
    "finalize": "hsl(258 65% 52%)",
    "camera": "hsl(113 65% 52%)",
    "phase": "hsl(172 65% 52%)",
    "wait": "hsl(0 65% 52%)",
    "ssd": "hsl(328 65% 52%)",
}
LEGEND = [
    ("step", "batch step"),
    ("compute", "CUDA compute / memset"),
    ("nccl", "collective (NCCL)"),
    ("copy", "memory copy"),
    ("preview", "plan preview / prefetch"),
    ("finalize", "plan finalize"),
    ("camera", "camera load / wait"),
    ("phase", "other CPU calls"),
    ("wait", "CPU blocking API"),
    ("ssd", "SSD read service"),
]


def phase_style(phase):
    return {
        "Predictive prefetch": ("preview", "plan preview / prefetch"),
        "Plan finalize": ("finalize", "plan finalize"),
        "Plan preview": ("preview", "plan preview"),
        "Camera load / wait": ("camera", "camera load / wait"),
        "Camera H2D submission": ("copy", "camera H2D submission"),
    }.get(phase["name"], ("phase", phase["name"]))


def display_labels(data):
    meta, batches = data["metadata"], data["batches"]
    world = meta.get("world_size", 8)
    process = f"rank {meta['rank']}" if meta.get("rank") is not None else f"PID {meta['pid']}"
    first, last = batches[0]["number"], batches[-1]["number"]
    selected = f"B{first}–{last}"
    bsz = meta.get("global_bsz", world*16)
    compact = f"{world} GPUs · {process}"
    mode = "micro" if meta.get("mode") == "pc1" else meta.get("mode")
    if mode in ("off", "micro"):
        compact = f"{mode.upper()} · {compact}"
    kind = 'long-read' if meta.get('window_kind') == 'long-read' else 'steady'
    return {
        "title": f"{compact} · {len(batches)} consecutive {kind} batches · GPU / CPU / SSD",
        "compact_title": compact,
        "subtitle": f"{selected} · global bsz {bsz} · relative time; 0 = B{first} camera scheduling",
        "medium_subtitle": f"{selected} · bsz {bsz} · 0 = B{first} start",
        "compact_subtitle": f"{selected} · bsz {bsz}",
        "page_title": f"{world}-GPU GPU CPU SSD timeline",
        "description": f"{len(batches)} aligned batch cycles on {process}. "
            "GPU compute, NCCL and copies use actual CUDA execution in separate rows, "
            "with opacity equal to activity in each 20 ms bin. CPU calls and blocking "
            "APIs occupy separate rows. SSD marks are recorded storage read calls from "
            "this process, including its worker threads; they exclude foreground "
            "read_batch wrappers and SSD writes. Dashed lines are complete batch boundaries. "
            "Hover or tap an event for its English annotation and duration. "
            + meta.get("gpu_capture_warning", ""),
    }


def render_static(data, width=1180):
    left, right = 92, width - 28
    span = data["metadata"]["duration_seconds"]
    plot = Element("g")
    # Four phase/wait pairs fit inside the existing CPU lane (148..226).
    # Unknown workers have their own row, above the SSD lane beginning at 230.
    cpu_rows = {role: 152 + 18 * index
                for index, role in enumerate(("main", "comm", "autograd", "unknown"))}

    def cpu_role(row):
        role = row.get("thread_role")
        return role if role in cpu_rows else "unknown"

    def add(tag, text=None, **attrs):
        node = SubElement(plot, tag, {k.replace("_", "-"): str(v) for k, v in attrs.items()})
        if text is not None:
            node.text = str(text)
        return node

    def x(t):
        return left + max(0, min(span, t)) / span * (right - left)

    def event(start, end, tip, y, height, kind, opacity=.86, *, thread_role=None):
        # Clip geometry, but report the original duration in the annotation.
        node = add("rect", **{
            "class": "event", "data-tooltip": tip,
            "data-start": start, "data-end": end, "data-kind": kind,
            "x": x(start), "y": y, "width": x(end) - x(start), "height": height,
            "rx": 3, "ry": 3, "opacity": opacity,
            "fill": "var(--muted)" if kind == "step" else f"var(--timeline-{kind})",
        })
        if thread_role is not None:
            node.set("data-thread-role", thread_role)
        SubElement(node, "title").text = tip

    def interval(start, end):
        return f"{start:.3f}–{end:.3f} s · {(end-start)*1000:.1f} ms"

    labels = display_labels(data)
    add("text", labels["title"],
        x=left, y=24, **{"class": "chart-title"})
    add("text", labels["subtitle"],
        x=left, y=45, **{"class": "chart-subtitle"})
    for y in (66, 148, 230):
        add("rect", x=left, y=y, width=right-left, height=78,
            fill="var(--muted)", opacity=.22, **{"class": "lane-bg"})
    for i in range(9):
        t = span * i / 8
        add("line", x1=x(t), x2=x(t), y1=66, y2=312,
            **{"class": "grid", "data-time": t, "data-tick-index": i})
        add("text", f"{t:.1f}s", x=x(t), y=334,
            text_anchor="start" if i == 0 else "end" if i == 8 else "middle",
            **{"class": "tick", "data-time": t, "data-tick-index": i})
    for label, y in (("GPU", 104), ("CPU", 186), ("SSD", 268)):
        add("text", label, x=80, y=y, text_anchor="end", **{"class": "lane-label"})
    for batch in data["batches"]:
        event(batch["start"], batch["end"],
              f"batch {batch['number']} · iter {batch['iteration']} · "
              f"{interval(batch['start'], batch['end'])} · camera scheduling to next scheduling",
              152, 72, "step", .28)

    for row in data["gpu_bins"]:
        label = {"compute": "CUDA compute / memset", "nccl": "collective (NCCL)",
                 "copy": "memory copy (H2D / D2H / D2D)"}[row["kind"]]
        ms = (row["end"] - row["start"]) * 1000
        tip = (f"{label} · {row['start']:.3f}–{row['end']:.3f} s · "
               f"active {ms*row['fraction']:.1f} / {ms:.1f} ms · "
               f"time coverage {row['fraction']*100:.1f}%")
        if row["kind"] == "nccl":
            tip += " · includes possible peer waits"
        event(row["start"], row["end"], tip,
              {"compute": 76, "nccl": 98, "copy": 120}[row["kind"]],
              20, row["kind"], row["fraction"])
    for phase in data["cpu_phases"]:
        kind, label = phase_style(phase)
        tip = f"{label} · {interval(phase['start'], phase['end'])} · {phase['source']}"
        if phase.get("detail"):
            tip += f" · {phase['detail']}"
        if phase["name"] == "Predictive prefetch":
            tip += " · outer call includes planning, prefetch submission and waits"
        role = cpu_role(phase)
        tip += f" · {role} thread {phase.get('thread_id', phase.get('global_tid', 'unknown'))}"
        event(phase["start"], phase["end"], tip, cpu_rows[role], 10, kind, thread_role=role)
    for wait in data["cpu_waits"]:
        role = cpu_role(wait)
        event(wait["start"], wait["end"],
              f"CPU blocking API · {wait['name']} · {interval(wait['start'], wait['end'])} · "
              f"{role} thread {wait.get('thread_id', wait.get('global_tid', 'unknown'))}",
              cpu_rows[role] + 11, 5, "wait", .65, thread_role=role)
    for read in data["ssd_reads"]:
        details = [f"SSD read service · {interval(read['start'], read['end'])}"]
        if read.get("read_kind"):
            details.append(f"{read['read_kind']} read")
        if read.get("target_iteration") is not None:
            details.append(f"target iter {read['target_iteration']}")
        if read.get("blocks") is not None:
            details.append(f"{read['blocks']} blocks")
        if read.get("bytes") is not None:
            details.append(f"{read['bytes']/1048576:.1f} MiB payload")
        if read.get("thread_id") is not None:
            details.append(f"thread {read['thread_id']}")
        if read.get("detail"):
            details.append(read["detail"])
        details.append("storage call; excludes foreground read_batch and SSD writeback")
        event(read["start"], read["end"],
              " · ".join(details),
              250, 36, "ssd")
    if not data["ssd_reads"]:
        add("text", "No SSD read calls recorded in this process/window", x=left+8, y=274,
            **{"class": "ssd-empty"})

    for batch in data["batches"]:
        add("line", x1=x(batch["start"]), x2=x(batch["start"]), y1=60, y2=312,
            **{"class": "batch-line", "data-time": batch["start"]})
        label = f"batch {batch['number']} · {batch['duration']:.3f}s"
        add("text", label, x=x(batch["start"])+4, y=56,
            **{"class": "batch-label", "data-time": batch["start"],
               "data-offset": 4, "data-batch-number": batch["number"], "data-full-label": label})
    add("line", x1=x(span), x2=x(span), y1=60, y2=312,
        **{"class": "batch-line", "data-time": span})
    add("text", "Relative time (s)", x=(left+right)/2, y=364,
        text_anchor="middle", **{"class": "axis-title", "data-axis": "x"})
    return "".join(tostring(child, encoding="unicode") for child in plot), 372
