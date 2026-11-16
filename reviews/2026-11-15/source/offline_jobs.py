"""Cooperative cancellation lets live capture preempt offline learning work."""
class WorkPaused(Exception):pass

def check_work(cancel):
    if cancel is not None and cancel():raise WorkPaused('Offline work paused for live capture or shutdown')
