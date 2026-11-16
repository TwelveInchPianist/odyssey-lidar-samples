"""Original M1 UDP protocol; geometry follows RoboSense rs_driver (BSD-3).

Not an M1 Plus/M2 decoder. See licenses/RoboSense.txt and SOURCES.txt.
"""
import math
import socket
import struct
import numpy as np

MSOP_ID = bytes.fromhex('55aa5aa5')
DIFOP_ID = bytes.fromhex('a5ff005a11115555')
CHANNEL = np.dtype([('distance', '>u2'), ('pitch', '>u2'), ('yaw', '>u2'),
                    ('intensity', 'u1'), ('attribute', 'u1'), ('elongation', 'u1')])
BLOCK = np.dtype([('offset', 'u1'), ('return_seq', 'u1'), ('channels', CHANNEL, (5,))])
PACKET = np.dtype([('header', 'u1', (32,)), ('blocks', BLOCK, (25,)), ('tail', 'u1', (3,))])

def valid_msop(p):
    return len(p) == 1210 and p[:4] == MSOP_ID and p[30] == 0x10 and p[8] in (0, 4, 5, 6)

def metadata(p):
    return {'sequence': int.from_bytes(p[4:6], 'big'), 'protocol': p[6:8].hex(),
            'return_mode': p[8], 'time_mode': p[9],
            'lidar_time_ns': int.from_bytes(p[10:16], 'big') * 1_000_000_000
                             + int.from_bytes(p[16:20], 'big') * 1000,
            'temperature_c': p[31] - 80}

def decode(packets):
    """Return Nx4 XYZ/intensity, each block retains timing in the raw PCAP.

    Caller validates headers. Angles use a 32768 offset, not signed int16.
    Range filter follows SDK 0.2..200m; performance below sensor specification
    (0.5m) is not implied.
    """
    return decode_packet_array(np.frombuffer(b''.join(packets),dtype=PACKET))[0]


def decode_packet_array(packet_array, with_time=False):
    """One parsed packet array supplies both coordinates and matching timestamps.

    Preserve the existing float arithmetic and range/angle mask. Trigonometry
    is performed only on retained returns; invalid returns never reach output.
    """
    a = packet_array['blocks']['channels']
    d = a['distance'].astype(np.float32).ravel() * 0.005
    pitch = (a['pitch'].astype(np.int32).ravel() - 32768) * (math.pi / 18000)
    yaw = (a['yaw'].astype(np.int32).ravel() - 32768) * (math.pi / 18000)
    good = (d >= 0.2) & (d <= 200) & (np.abs(pitch) <= math.pi/2) & (np.abs(yaw) <= math.pi)
    d=d[good];pitch=pitch[good];yaw=yaw[good]
    xy = d * np.cos(pitch)
    points=np.column_stack((xy*np.cos(yaw), xy*np.sin(yaw), d*np.sin(pitch),
                            a['intensity'].ravel()[good])).astype(np.float32)
    if not with_time:return points,None
    h=packet_array['header'].astype(np.int64)
    seconds=np.sum(h[:,10:16]*np.array([1<<40,1<<32,1<<24,1<<16,1<<8,1]),axis=1)
    micros=np.sum(h[:,16:20]*np.array([1<<24,1<<16,1<<8,1]),axis=1)
    maximum=np.iinfo(np.int64).max
    if np.any(seconds>(maximum-micros*1000)//1_000_000_000):
        raise OverflowError('M1 timestamp cannot be represented as int64 nanoseconds')
    base=seconds*1_000_000_000+micros*1000
    stamps=np.repeat((base[:,None]+packet_array['blocks']['offset'].astype(np.int64)*1000).ravel(),5)[good]
    return points,stamps

def checksum(data):
    if len(data) % 2:
        data += b'\0'
    total = sum(struct.unpack('!%dH' % (len(data)//2), data))
    while total >> 16:
        total = (total & 65535) + (total >> 16)
    return (~total) & 65535

class PcapWriter:
    """DLT_RAW PCAP, intact UDP payloads; IP/UDP headers reconstructed.

    This is application-level recording, not an Ethernet wire capture.
    Packet timestamps are laptop receipt wall time. UDP checksum zero (IPv4).
    """
    def __init__(self, path, destination='192.168.1.102'):
        self.file = open(path, 'xb', buffering=1024*1024)
        self.destination = socket.inet_aton(destination)
        self.file.write(struct.pack('<IHHIIII', 0xa1b2c3d4, 2, 4, 0, 0, 65535, 101))
        self.bytes = 24

    def write(self, payload, source, port, wall_ns):
        udp = struct.pack('!HHHH', source[1], port, 8+len(payload), 0) + payload
        hdr = struct.pack('!BBHHHBBH4s4s', 0x45, 0, 20+len(udp), 0, 0, 64, 17, 0,
                          socket.inet_aton(source[0]), self.destination)
        hdr = hdr[:10] + struct.pack('!H', checksum(hdr)) + hdr[12:]
        ip = hdr + udp
        self.file.write(struct.pack('<IIII', wall_ns//1_000_000_000,
                                    (wall_ns % 1_000_000_000)//1000, len(ip), len(ip)) + ip)
        self.bytes += 16 + len(ip)

    def close(self):
        self.file.close()

def read_pcap(path):
    """Classic PCAP: raw IPv4 or Ethernet, optional VLAN. No pcapng/fragments."""
    with open(path, 'rb') as f:
        h = f.read(24)
        kinds = {b'\xd4\xc3\xb2\xa1': ('<',1000), b'\xa1\xb2\xc3\xd4':('>',1000),
                 b'\x4d\x3c\xb2\xa1':('<',1), b'\xa1\xb2\x3c\x4d':('>',1)}
        if len(h) != 24 or h[:4] not in kinds:
            raise ValueError('Use classic .pcap, not .pcapng (convert with Wireshark Save As).')
        endian, scale = kinds[h[:4]]
        link = struct.unpack(endian+'I', h[20:24])[0]
        if link not in (1, 101):
            raise ValueError('Supported PCAP links: Ethernet or raw IPv4.')
        while True:
            record = f.read(16)
            if not record:
                break
            if len(record) != 16:
                raise ValueError('Truncated PCAP record header')
            sec, sub, length, original = struct.unpack(endian+'IIII', record)
            if length > 1_000_000:
                raise ValueError('Invalid PCAP record length')
            ip = f.read(length)
            if len(ip) != length:
                raise ValueError('Truncated PCAP packet')
            if link == 1:
                if len(ip) < 14:
                    continue
                ethertype = int.from_bytes(ip[12:14], 'big')
                offset = 14
                while ethertype in (0x8100, 0x88a8) and len(ip) >= offset+4:
                    ethertype = int.from_bytes(ip[offset+2:offset+4], 'big')
                    offset += 4
                if ethertype != 0x0800:
                    continue
                ip = ip[offset:]
            if len(ip) < 28 or ip[0] >> 4 != 4 or ip[9] != 17:
                continue
            ihl = (ip[0] & 15)*4
            if ihl < 20 or len(ip) < ihl+8 or int.from_bytes(ip[6:8], 'big') & 0x3fff:
                continue
            srcport, dstport, size, _ = struct.unpack('!HHHH', ip[ihl:ihl+8])
            if size < 8 or len(ip) < ihl+size:
                continue
            yield sec*1_000_000_000+sub*scale, (socket.inet_ntoa(ip[12:16]), srcport), dstport, ip[ihl+8:ihl+size]

def make_demo_packet(sequence, phase=0):
    """Synthetic front wall + sloping ground. Never represents measured data."""
    p = bytearray(1210)
    p[:4] = MSOP_ID
    struct.pack_into('!H', p, 4, sequence)
    p[8], p[30], p[31] = 4, 16, 110
    for block in range(25):
        off = 32 + block*47
        p[off] = block*5
        for c in range(5):
            yaw = -58 + ((sequence*125+block*5+c) % 600)*116/599
            pitch = -12 + ((sequence//5+block) % 50)*24/49
            d = min(28/max(0.2, math.cos(math.radians(yaw))),
                    1.8/max(0.001, -math.sin(math.radians(pitch))))
            if abs(yaw-8*math.sin(phase)) < 5 and abs(pitch) < 5:
                d = 12
            struct.pack_into('!HHHB', p, off+2+c*9, round(d/0.005),
                             round(pitch*100)+32768, round(yaw*100)+32768, 110+c*25)
    return bytes(p)
