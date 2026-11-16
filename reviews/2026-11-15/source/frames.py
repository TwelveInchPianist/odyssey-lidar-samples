"""Original M1 scan boundaries and immutable per-point device timestamps."""
import numpy as np
from m1_protocol import valid_msop,metadata,decode,decode_packet_array,PACKET

def decode_with_time(packets):
    p=np.frombuffer(b''.join(packets),dtype=PACKET)
    return decode_packet_array(p,with_time=True)

class FrameAssembler:
    def __init__(self):
        self.packets={};self.arrivals={};self.expected=0;self.started=False
        self.bad_packets=0;self.duplicate_packets=0;self.partial_frames=0

    def finish(self):
        if not self.packets:return None
        seq=sorted(self.packets)
        packets=[self.packets[i] for i in seq]
        point_count=len(packets)*125
        timestamps=[metadata(p)['lidar_time_ns'] for p in packets]
        span=(max(timestamps)-min(timestamps))/1e9
        valid=(timestamps[0]>0 and .01<span<.15 and np.all(np.diff(timestamps)>=0))
        quality={'completeness':len(seq)/self.expected,'timestamp_valid':bool(valid),
                 'packets':len(seq),'expected_packets':self.expected,'device_span_s':span,
                 'first_receipt_ns':min(self.arrivals.values()),'last_receipt_ns':max(self.arrivals.values()),
                 'first_device_ns':timestamps[0],'last_device_ns':timestamps[-1],
                 'min_receipt_minus_device_ns':min(self.arrivals[i]-stamp for i,stamp in zip(seq,timestamps))}
        if len(seq)<self.expected:self.partial_frames+=1
        self.packets={};self.arrivals={};self.started=False
        return packets,quality

    def push(self,packet,arrival_ns):
        if not valid_msop(packet):self.bad_packets+=1;return []
        meta=metadata(packet)
        if int.from_bytes(packet[10:16],'big')>2**32 or int.from_bytes(packet[16:20],'big')>=1_000_000:
            self.bad_packets+=1;return []
        seq=meta['sequence'];expected=1260 if meta['return_mode']==0 else 630
        if not 1<=seq<=expected:self.bad_packets+=1;return []
        out=[]
        if seq==1:
            if self.started and self.packets.get(1)==packet:
                self.duplicate_packets+=1;return []
            prior=self.finish()
            if prior:out.append(prior)
            self.started=True;self.expected=expected
        if not self.started:return out  # discard partial first scan after connecting
        if expected!=self.expected:self.bad_packets+=1;return out
        if seq in self.packets:self.duplicate_packets+=1;return out
        if self.packets:
            # An accepted scan always starts with sequence 1. Avoid searching
            # every accumulated key for every packet (quadratic per scan).
            first=metadata(self.packets[1])['lidar_time_ns']
            if meta['lidar_time_ns']<first or meta['lidar_time_ns']-first>150_000_000:
                self.bad_packets+=1;return out
        self.packets[seq]=packet;self.arrivals[seq]=arrival_ns
        if seq==self.expected:
            out.append(self.finish())
        return out
