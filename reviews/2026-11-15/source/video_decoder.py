"""Decode timestamped access units in a process separate from OpenCV inference."""
from fractions import Fraction
import time
from video_transport import Continuity

PIXEL_CONVERSION='pyav-bgr-bilinear-accurate-bitexact-v1'


def native_bgr(frame):
    """Same standard color conversion on ARM/x86; no enhancement or brightening."""
    from av.video.reformatter import Interpolation
    flags=Interpolation.BILINEAR|Interpolation.ACCURATE_RND|Interpolation.BITEXACT
    return frame.to_ndarray(format='bgr24',interpolation=flags)


class VideoDecoder:
    def __init__(self,hw_device=None):
        import av
        self.av=av;self.continuity=Continuity();self.codecs={};self.metadata={}
        if hw_device not in (None,'d3d11va','videotoolbox'):raise ValueError('Unsupported explicit video device')
        self.hw_device=hw_device;self.hardware_active={}

    def reset(self,service):
        self.codecs.pop(service,None);self.metadata.pop(service,None)

    def decode(self,meta,payload):
        if meta.get('codec')!='fullHEVC':raise ValueError('Only native road HEVC is accepted')
        service=meta['service'];continuity=self.continuity.accept(meta)
        if continuity['reset_decoder']:self.reset(service)
        if not continuity['ready']:return [],continuity['reason']
        if service not in self.codecs:
            options={}
            if self.hw_device:
                from av.codec.hwaccel import HWAccel
                options['hwaccel']=HWAccel(self.hw_device,allow_software_fallback=False,is_hw_owned=False)
            codec=self.av.CodecContext.create('hevc','r',**options)
            codec.thread_count=1 if self.hw_device else 2
            if not self.hw_device:codec.thread_type='SLICE'
            self.codecs[service]=codec;self.metadata[service]={}
        packet=self.av.Packet(payload);packet.pts=meta['source_ns'];packet.dts=packet.pts
        packet.time_base=Fraction(1,1_000_000_000)
        self.metadata[service][packet.pts]=dict(meta)
        if len(self.metadata[service])>32:
            self.reset(service);self.continuity.states[service]['ready']=False
            return [],'decoder_timestamp_backlog'
        start=time.perf_counter_ns();out=[]
        try:frames=self.codecs[service].decode(packet)
        except Exception:
            self.reset(service);self.continuity.states[service]['ready']=False
            return [],'decoder_error_waiting_for_keyframe'
        self.hardware_active[service]=bool(getattr(self.codecs[service],'is_hwaccel',False))
        if frames and self.hw_device and not self.hardware_active[service]:
            raise RuntimeError('Requested video acceleration silently fell back')
        for frame in frames:
            if frame.pts not in self.metadata[service]:
                self.reset(service);self.continuity.states[service]['ready']=False
                return [],'decoder_did_not_preserve_source_identity'
            identity=self.metadata[service].pop(frame.pts)
            if (frame.width,frame.height)!=(identity['width'],identity['height']):
                self.reset(service);self.continuity.states[service]['ready']=False
                return [],'decoded_shape_disagrees_with_source'
            out.append({'metadata':{**identity,'pixel_conversion':PIXEL_CONVERSION},'bgr':native_bgr(frame),
                        'decode_ms':(time.perf_counter_ns()-start)/1e6})
        return out,None
