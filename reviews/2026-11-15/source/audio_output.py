"""Pre-rendered local speech: no network TTS, no stale utterance queue."""
from pathlib import Path
import os
import runtime_clock as time
import wave

PHRASES=['OBSTACLE AHEAD','OBSTACLE LEFT','OBSTACLE RIGHT','ONCOMING OBSTACLE AHEAD',
         'ONCOMING OBSTACLE LEFT','ONCOMING OBSTACLE RIGHT','WARNING SYSTEM UNAVAILABLE','AUDIO TEST',
         'WARNING RANGE LIMITED']
AWARENESS_PHRASES={noun+' '+direction for noun in ('PERSON','PEOPLE','CYCLIST','CYCLISTS','BICYCLE','BICYCLES','MOTORCYCLE','MOTORCYCLES','SCOOTER','SCOOTERS','ANIMAL','ANIMALS','STOPPED VEHICLE','STOPPED VEHICLES','OBJECT','OBJECTS') for direction in ('AHEAD','LEFT','RIGHT')}
PHRASES+=sorted(AWARENESS_PHRASES)
STATUS_PHRASES={'WARNING SYSTEM UNAVAILABLE','WARNING RANGE LIMITED'}

class Speaker:
    def __init__(self,assets):
        self.assets=Path(assets);self.last=None;self.last_rejection=None;self.last_interruption=None;self.busy_until_ns=0
    def say(self,phrase,created_ns=None):
        now=time.monotonic_ns();self.last_rejection=None;self.last_interruption=None
        if phrase not in PHRASES:self.last_rejection='unknown_phrase';return False
        if created_ns is not None and now-created_ns>500_000_000:self.last_rejection='expired';return False
        # Status speech never cuts off an obstacle warning (or another status phrase).
        # The caller may retry a current status after playback; no stale utterance queue.
        if phrase in STATUS_PHRASES and now<self.busy_until_ns:
            self.last_rejection='deferred_until_audio_idle';return False
        path=self.assets/(phrase.replace(' ','_')+'.wav')
        if not path.exists():raise FileNotFoundError('Missing speech asset: '+str(path))
        try:
            with wave.open(str(path),'rb') as wav:duration_ns=int(wav.getnframes()/wav.getframerate()*1e9)
        except wave.Error as e:raise RuntimeError('Invalid speech asset: '+str(path)) from e
        if os.name=='nt':
            import winsound
            winsound.PlaySound(str(path),winsound.SND_FILENAME|winsound.SND_ASYNC|winsound.SND_NODEFAULT)
        if self.last and now<self.busy_until_ns:self.last_interruption={**self.last,'interrupted_ns':now,'replaced_by':phrase}
        self.last={'phrase':phrase,'at_ns':now};self.busy_until_ns=now+duration_ns+100_000_000
        return True
    def stop(self):
        self.busy_until_ns=0
        if os.name=='nt':
            import winsound
            winsound.PlaySound(None,0)
