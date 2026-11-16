"""One high-resolution local clock domain for capture, sync and deadlines.

Python 3.12 on Windows implements time.monotonic_ns with GetTickCount64
(15.625 ms resolution). perf_counter uses the monotonic QueryPerformanceCounter.
Old recorded timestamps are never converted by this module. The standalone
comma bridge intentionally retains its native clock; ping/pong maps domains.
"""
import time as _time
import sys

def monotonic_ns():
    return _time.perf_counter_ns()

def monotonic():
    return _time.perf_counter()

def get_clock_info(name):
    return _time.get_clock_info('perf_counter' if name=='monotonic' else name)

def describe():
    info=_time.get_clock_info('perf_counter')
    legacy=_time.get_clock_info('monotonic')
    return {'schema_version':1,'domain':'python_perf_counter','implementation':info.implementation,
            'nominal_resolution_ns':int(round(info.resolution*1e9)),
            'monotonic':bool(info.monotonic),'adjustable':bool(info.adjustable),
            'python_version':sys.version.split()[0],
            'legacy_monotonic_resolution_ns':int(round(legacy.resolution*1e9)),
            'scope':'Local clock only. Recorded synchronization is still required for sensor clocks; nominal resolution is not sensor timing accuracy.'}

def __getattr__(name):
    return getattr(_time,name)
