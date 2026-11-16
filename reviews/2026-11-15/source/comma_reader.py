"""Read-only observation of the inspected msgq ring format; no subscriber slots.

Native subscribers in msgq 9beb84a have a 15-reader limit and evict one another
when exceeded. This observer never registers, writes shared memory, or signals
publishers/readers. It accepts only complete snapshots of new messages from a
stable writer. Unsupported binaries/layouts fail closed rather than falling back
to a subscriber that could affect the vehicle's processes.
"""
import hashlib
import mmap
import os
from pathlib import Path
import re
import struct

HEADER_SIZE = 384
MAX_READERS = 15
DEFAULT_SIZE = 1024 * 1024
SUPPORTED_NATIVE_SHA256 = {
    '441a5391c2370c5ec12be1150f6f616237f8e5f5a89726762e07b1a61493990f',
}


def verify_native_layout(native_module):
    digest = hashlib.sha256(Path(native_module.__file__).read_bytes()).hexdigest()
    if digest not in SUPPORTED_NATIVE_SHA256:
        raise RuntimeError('Unsupported msgq binary: read-only telemetry withheld; inspect layout before enabling this build')
    return digest


class RingSnapshot:
    """Single-threaded cursor over a read-only mapping; latest complete event."""
    def __init__(self, mapping):
        self.mapping = mapping
        self.size = len(mapping) - HEADER_SIZE
        if self.size < 16 or self.size % 8:
            raise ValueError('Invalid msgq ring size')
        self.cursor = None
        self.writer = None
        self.stats = {'received': 0, 'overruns': 0, 'races': 0, 'malformed': 0, 'writer_resets': 0}

    def header(self):
        count, pointer, writer = struct.unpack('<3Q', self.mapping[:24])
        if count > MAX_READERS or (pointer & 0xffffffff) > self.size or (pointer & 7):
            raise ValueError('Unsupported or invalid msgq header')
        return pointer, writer

    @staticmethod
    def messages(data, require_wrap=False):
        at = 0
        last = None
        while at < len(data):
            if len(data) - at < 8:
                raise ValueError('Truncated message length')
            length = struct.unpack_from('<q', data, at)[0]
            if length == -1:
                if not require_wrap:
                    raise ValueError('Unexpected wrap marker')
                return last
            if length <= 0 or length > len(data) - at - 8:
                raise ValueError('Incomplete or invalid message')
            end = (at + 8 + length + 7) & ~7
            if end > len(data):
                raise ValueError('Incomplete message alignment')
            last = data[at + 8:at + 8 + length]
            at = end
        if require_wrap:
            raise ValueError('Missing wrap marker')
        return last

    def receive(self):
        pointer, writer = self.header()
        if not writer or writer != self.writer or self.cursor is None:
            if self.writer is not None and writer != self.writer:
                self.stats['writer_resets'] += 1
            self.writer, self.cursor = writer, pointer
            return None  # Never replay buffered data on attach/restart.
        if pointer == self.cursor:
            return None
        pos, old_pos = pointer & 0xffffffff, self.cursor & 0xffffffff
        cycles = ((pointer >> 32) - (self.cursor >> 32)) & 0xffffffff
        if cycles > 1 or (cycles == 0 and pos < old_pos) or (cycles == 1 and pos >= old_pos):
            self.stats['overruns'] += 1
            self.cursor = pointer
            return None
        try:
            if cycles:
                # Only the new cycle is needed for a latest-value observer.
                # A writer can already be overwriting the old unread tail.
                head = bytes(self.mapping[HEADER_SIZE:HEADER_SIZE + pos])
                payload = self.messages(head)
            else:
                payload = self.messages(bytes(self.mapping[HEADER_SIZE + old_pos:HEADER_SIZE + pos]))
        except ValueError:
            # A concurrent wrap can look malformed; neither case yields data.
            self.stats['malformed'] += 1
            self.cursor = pointer
            return None
        if self.header() != (pointer, writer):
            self.stats['races'] += 1
            return None
        self.cursor = pointer
        if payload is not None:
            self.stats['received'] += 1
        return payload


class ReadOnlyMessageReader:
    """Minimal SubMaster-shaped facade used only by the external FCW bridge."""
    def __init__(self, services, messaging, registry, root=None):
        self.services = services
        self.messaging = messaging
        self.registry = registry
        self.updated = dict.fromkeys(services, False)
        self.valid = dict.fromkeys(services, False)
        self.logMonoTime = dict.fromkeys(services, 0)
        self.data = {}
        self.rings = {}
        self.files = {}
        prefix = os.environ.get('OPENPILOT_PREFIX', '')
        if prefix and not re.fullmatch(r'[A-Za-z0-9_-]+', prefix):
            raise ValueError('Unsupported msgq namespace')
        self.root = Path(root) if root is not None else Path('/dev/shm/msgq_' + prefix)
        self.prefixed = bool(prefix) if root is None else True

    def __getitem__(self, name):
        return self.data[name]

    def update(self, timeout=0):
        self.updated = dict.fromkeys(self.services, False)
        for name in self.services:
            if not re.fullmatch(r'[A-Za-z0-9_]+', name):
                raise ValueError('Invalid service name')
            path = self.root / name if self.prefixed else Path(str(self.root) + name)
            if name not in self.rings:
                try:
                    file = path.open('rb')
                except FileNotFoundError:
                    continue
                expected = getattr(self.registry[name], 'queue_size', 0) or DEFAULT_SIZE
                if os.fstat(file.fileno()).st_size != HEADER_SIZE + expected:
                    file.close()
                    raise ValueError('Unsupported msgq queue size for ' + name)
                mapping = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
                self.files[name] = (file, mapping)
                self.rings[name] = RingSnapshot(mapping)
            file, mapping = self.files[name]
            stat = path.stat()
            if stat.st_size != len(mapping) or stat.st_ino != os.fstat(file.fileno()).st_ino:
                raise ValueError('msgq queue changed during observation: ' + name)
            payload = self.rings[name].receive()
            if payload is None:
                continue
            event = self.messaging.log_from_bytes(payload)
            if event.which() != name or event.logMonoTime <= 0:
                raise ValueError('msgq event identity/timestamp mismatch')
            if event.logMonoTime <= self.logMonoTime[name]:
                continue
            self.data[name] = getattr(event, name)
            self.valid[name] = bool(event.valid)
            self.logMonoTime[name] = int(event.logMonoTime)
            self.updated[name] = True

    def close(self):
        for file, mapping in self.files.values():
            mapping.close()
            file.close()
        self.files.clear()
        self.rings.clear()
