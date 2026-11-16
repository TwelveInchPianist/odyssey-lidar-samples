"""Claude fork v19: hands-free operation for the car tablet (Collin, 2026-10-20: "warnings default on, automated recording,
automated stopping"). Enabled only by config keys, so Astra's defaults and tests are unchanged:

  always_warn  (bool)  at every launch, forced warnings + speech are switched on (an existing forced_experiment block is required)
  auto_record  (bool)  a supervisor thread starts a live recording when lidar packets arrive and stops it when the drive ends

Auto-start: while no recording runs, probe the lidar UDP port for 1 s every few seconds; packets on 2 consecutive probes -> start.
Auto-stop:  live recording with no raw lidar data packet for NO_LIDAR_STOP_S (c099 v30d; was: no processed frame), or the comma
            reporting ignition off for IGNITION_OFF_STOP_S -> stop (the normal post-drive review follows).
Manual Stop: a Stop pressed while the lidar is still sending holds auto-start until the lidar has gone quiet once (next drive).

c099 v30d (2026-10-21 live forensics §2): the no-lidar rule read frame_age_s, the age of the last PROCESSED frame, so a dead frame
thread looked like a missing lidar: the session would auto-stop ~62 s after the crash and auto-restart 20 s later, in a loop. It now
reads the raw packet age (Session.snapshot lidar_packet_age_s). Every supervisor transition is written to
<recording>/autodrive-events.jsonl (autodrive used to write nothing, so starts and stops had to be reconstructed from timing).

Inference failures keep retrying while the live session is eligible. Repeated failures
use a longer restart interval, never a permanent manual-reset lockout. Raw-lidar loss,
ignition-off, manual Stop and shutdown retain their normal authority.
"""
import socket, threading, time
from collections import deque
import runtime_clock   # c099 v30d: event stamps in the recording clock domain; timers unchanged

PROBE_EVERY_S = 3.0
PROBE_LISTEN_S = 1.0
PROBES_TO_START = 2
NO_LIDAR_STOP_S = 60.0
IGNITION_OFF_STOP_S = 120.0
AFTER_STOP_PAUSE_S = 20.0
EVENT_BACKLOG = 32   # c099 v30d: transitions kept while no recording is open; written at the top of the next one
ENGINE_RESTARTS_FAST = 3   # thereafter back off, never exhaust recovery
ENGINE_RESTART_BACKOFF_S = 60.0
ENGINE_RESTART_WINDOW_S = 1800.0   # c099 v30d review: ... per 30 min, so a rare transient failure on a later drive is still recovered
ENGINE_RESTART_GAP_S = 30.0   # c099 v30d review: at least this long between automatic engine restarts
ENGINE_RESTART_HOLD_S = 60.0   # c099 v30d review: no lidar probe while the restarted session is being opened (the app queues its start)


def enforce_always_warn(config):
    """Switch forced warnings and speech on; returns True if the config changed."""
    if not config.get('always_warn'): return False
    changed = False
    fe = config.get('forced_experiment')
    if isinstance(fe, dict) and fe.get('enabled') is not True: fe['enabled'] = True; changed = True
    if config.get('audio_enabled') is not True: config['audio_enabled'] = True; changed = True
    return changed


def lidar_present(config):
    """True if at least one UDP packet arrives on the lidar data port within PROBE_LISTEN_S (only while no recording holds it)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((config.get('bind_ip') or '0.0.0.0', int(config.get('msop_port') or 6699)))
        s.settimeout(PROBE_LISTEN_S)
        try:
            s.recvfrom(2048); return True
        except socket.timeout:
            return False
    except OSError:
        return False
    finally:
        s.close()


class AutoDrive:
    def __init__(self, app):
        self.app = app; self.manual_hold = False; self.state = 'waiting for the lidar'
        self.kind = 'waiting'; self.backlog = deque(maxlen=EVENT_BACKLOG); self.event_lock = threading.Lock(); self.inputs = {}   # c099 v30d
        self.engine_restarts = []; self.restart_hold_until = 0.0   # c099 v30d review: monotonic times of automatic engine restarts
        self.thread = threading.Thread(target=self.run, daemon=True, name='autodrive')

    def start(self):
        if self.app.config.get('auto_record'): self.thread.start()

    def manual_stop(self):
        self.manual_hold = True
        # c099 v30d: runs in the /stop HTTP thread before app.stop(), so the line lands in the still-open recording. Guarded: a hand
        # Stop must never fail because of event logging.
        try: self._note('manual_stop', 'Stop pressed by hand', self._live(), self.inputs)
        except Exception: pass

    def _live(self):
        s = self.app.session
        # c099 v30d: a session in closeout (stop_event set) is no longer live, so a manual Stop is not followed by a false 'recording'
        # transition or a repeated app.stop()
        return s if (s and not s.stopped and s.mode == 'live' and not s.stop_event.is_set()) else None

    def _note(self, kind, reason, s=None, snap=None):
        """c099 v30d: record one supervisor transition (kind changed) and write the backlog into the live recording, oldest first.
        Never raises; a transition with no recording open stays in the bounded backlog until the next recording."""
        try:
            with self.event_lock:
                if kind == self.kind: return
                snap = snap or {}
                self.backlog.append({'kind': 'autodrive_transition', 'schema_version': 1,
                                     'wall_ns': runtime_clock.time_ns(), 'monotonic_ns': runtime_clock.monotonic_ns(), 'clock': 'python_perf_counter',
                                     'from': self.kind, 'to': kind, 'reason': reason, 'state': self.state,
                                     'frame_age_s': snap.get('frame_age_s'), 'lidar_packet_age_s': snap.get('lidar_packet_age_s'),
                                     'ignition_on': snap.get('ignition_on'), 'manual_hold': self.manual_hold,
                                     'session_id': (s.config or {}).get('session_id') if s is not None else None})
                self.kind = kind
                while s is not None and self.backlog and s.autodrive_event(self.backlog[0]): self.backlog.popleft()
        except Exception:
            pass

    def _engine_restart(self, s, snap, now):
        """c099 v30d review: bounded automatic restart of a live session whose engine failed (module docstring). True = restart requested."""
        self.engine_restarts = [t for t in self.engine_restarts if now - t < ENGINE_RESTART_WINDOW_S]
        used = len(self.engine_restarts); error = str((snap.get('engine_failure') or {}).get('error') or 'engine failure')[:160]
        gap = ENGINE_RESTART_BACKOFF_S if used >= ENGINE_RESTARTS_FAST else ENGINE_RESTART_GAP_S
        wait = gap - (now - self.engine_restarts[-1]) if self.engine_restarts else 0.0
        if wait > 0:
            self.state = 'engine error: automatic restart in %.0f s (%d recent attempts)' % (wait, used)
            self._note('engine_restart_wait', self.state, s, snap); return False
        self.engine_restarts.append(now); self.restart_hold_until = now + ENGINE_RESTART_HOLD_S
        self.state = 'engine error: restarting automatically (attempt %d): %s' % (used + 1, error)
        self._note('engine_restart', self.state, s, snap)   # lands in the failed recording, before its closeout
        self.app.stop()
        # Application.start queues this while the failed session closes (busy -> pending_start) and opens it right after
        try: self.app.start('live')
        except Exception as e:
            self.state = 'engine restart could not start: ' + str(e); self._note('start_failed', self.state)
        return True

    def run(self):
        hits = 0; ign_off_since = None; no_lidar_since = None; last_stop = 0.0
        while not self.app.shutdown.is_set() and not self.app.closing:
            try:
                if getattr(self.app,'standby_pending',False):
                    self.state='standby — recording paused until wake';time.sleep(.5);continue
                s = self._live()
                if s is None:
                    ign_off_since = no_lidar_since = None
                    other = self.app.session and not self.app.session.stopped   # e.g. a demo started by hand: leave it alone
                    # c099 v30d review: no lidar probe while an engine restart's session is being opened (the probe binds the lidar port)
                    if other or self.app.starting or time.monotonic() - last_stop < AFTER_STOP_PAUSE_S or time.monotonic() < self.restart_hold_until:
                        hits = 0; time.sleep(PROBE_EVERY_S); continue
                    present = lidar_present(self.app.config)
                    if not present: self.manual_hold = False
                    hits = hits + 1 if present and not self.manual_hold else 0
                    self.state = 'held after a manual stop' if self.manual_hold else 'lidar seen, starting' if hits else 'waiting for the lidar'
                    self._note('held' if self.manual_hold else 'lidar_seen' if hits else 'waiting', self.state)   # c099 v30d
                    if hits >= PROBES_TO_START:
                        hits = 0; self.state = 'recording (started automatically)'
                        try: self.app.start('live')
                        except Exception as e: self.state = 'could not start: ' + str(e)
                        self._note('start_failed' if self.state.startswith('could not') else 'auto_start', self.state, self._live())   # c099 v30d
                    time.sleep(PROBE_EVERY_S); continue
                snap = s.supervisor_snapshot() if hasattr(s,'supervisor_snapshot') else s.snapshot()
                now = time.monotonic(); self.restart_hold_until = 0.0
                age = snap.get('frame_age_s')
                raw = snap.get('lidar_packet_age_s', age)   # c099 v30d: raw packets decide 'no lidar'; falls back to frame age only if the key is absent
                self.inputs = {k: snap.get(k) for k in ('frame_age_s', 'lidar_packet_age_s', 'ignition_on')}   # c099 v30d: for manual_stop's event
                no_lidar_since = (no_lidar_since or now) if (raw is None or raw > 2.0) else None
                ign = snap.get('ignition_on')
                ign_off_since = (ign_off_since or now) if ign is False else None
                why = ('no lidar for %.0f s' % (now - no_lidar_since)) if no_lidar_since and now - no_lidar_since >= NO_LIDAR_STOP_S else \
                      ('ignition off for %.0f s' % (now - ign_off_since)) if ign_off_since and now - ign_off_since >= IGNITION_OFF_STOP_S else None
                if why:
                    self.state = 'stopped automatically: ' + why; last_stop = now
                    self._note('auto_stop', self.state, s, snap)   # c099 v30d: written before closeout begins
                    self.app.stop()
                elif snap.get('engine_failure') is not None:
                    self._engine_restart(s, snap, now)   # c099 v30d review: bounded engine restart (the raw-packet rule never stops this session)
                else:
                    self.state = 'recording'
                    self._note('recording', {'auto_start': 'running after an automatic start', 'engine_restart': 'running after an automatic engine restart'}.get(self.kind, 'running, started by hand'), s, snap)   # c099 v30d
            except Exception as e:  # never let the supervisor die
                self.state = 'supervisor error: ' + str(e)
                try: self._note('supervisor_error', self.state, self._live())   # c099 v30d
                except Exception: pass
            time.sleep(2.0)
