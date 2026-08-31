"""Rolling-horizon driver for EOLES-Dispatch.

Builds and solves a sequence of overlapping 3-day windows instead of one
whole-year perfect-foresight model, to remove look-ahead bias from the MCP
series used as a LEAR feature.

Each window covers [day-1, day, day+1] but commits only the middle day:
the day before acts as a warm-up buffer that absorbs boundary-condition
artifacts (unknown pre-window on/off state, SOC, ...), and the day after
acts as a look-ahead buffer that avoids end-of-horizon effects (e.g. a
storage unit draining itself right before the window ends because it gets
no credit for leftover energy). Only the middle day's results are trusted
and stitched into the final output series.
"""


def make_rolling_windows(hours):
    """Split a sorted, consecutive hour range into overlapping 3-day windows.

    Args:
        hours: Sorted list/array of consecutive POSIX hours (int) spanning
            the whole backtest period, with no gaps, and a length that is a
            whole number of 24-hour calendar days (POSIX/UTC hours have no
            DST gaps, so this always holds for a period produced by
            compute_hour_mappings).

    Returns:
        List of dicts, one per calendar day, in chronological order of the
        *committed* day:
            {
                "day_index": int,           # 0-based index into calendar days
                "window_hours": [int, ...], # hours to build the model over
                "committed_hours": [int, ...],  # this window's middle day
            }
        The first and last windows have no previous/next day to draw on and
        fall back to a 2-day window (missing that side's buffer) -- they are
        the only ones with no real prior state / no look-ahead.
    """
    hours = sorted(hours)
    if len(hours) % 24 != 0:
        raise ValueError(
            f"expected a whole number of 24h calendar days, got {len(hours)} hours"
        )

    days = [hours[i : i + 24] for i in range(0, len(hours), 24)]
    n_days = len(days)

    windows = []
    for d in range(n_days):
        window_hours = []
        if d > 0:
            window_hours += days[d - 1]
        window_hours += days[d]
        if d < n_days - 1:
            window_hours += days[d + 1]

        windows.append(
            {
                "day_index": d,
                "window_hours": window_hours,
                "committed_hours": list(days[d]),
            }
        )

    return windows


def state_source_day_index(day_index):
    """Which committed day's end-state a window's initial conditions come from.

    Window `day_index` covers [day_index - 1, day_index, day_index + 1] and
    commits day_index. Its first modeled hour is the start of day
    (day_index - 1), so it needs the real state at the end of day
    (day_index - 2) -- which was committed by the window at day_index - 2,
    not by the immediately preceding window (whose committed day,
    day_index - 1, gets *re-solved* here as this window's own warm-up
    buffer, not reused directly).

    Returns None when no such day exists yet (the first two windows of the
    whole backtest), meaning the cold-start defaults in build_model apply.
    """
    source = day_index - 2
    return source if source >= 0 else None
