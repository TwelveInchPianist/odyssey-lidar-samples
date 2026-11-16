"""Isolated compiled-copy diagnostic. Only creates its own temporary fixtures."""
import hashlib,json,tempfile
from pathlib import Path
from session_sync import SessionSync

def run():
    with tempfile.TemporaryDirectory(prefix='odyssey-sync-test-') as temporary:
        root=Path(temporary);records=root/'recordings';source=records/'synthetic-session';source.mkdir(parents=True)
        (source/'session.json').write_text('{"capture_mode":"test","note":"temporary synthetic sync diagnostic"}')
        (source/'final-state.json').write_text('{}');(source/'offline-status.json').write_text('{"state":"complete"}')
        payload=bytes(range(256))*8192;(source/'lidar-00000.pcap').write_bytes(payload)
        destination=root/'destination';state={'busy':True,'ssid':'Home-test'}
        profile={'enabled':True,'home_ssid':'Home-test','destination':str(destination)}
        worker=SessionSync(records,root,lambda:profile,lambda:state['busy'],ssid_provider=lambda:{'ssids':[state['ssid']],'error':None})
        assert worker.run_once()['status']=='paused' and not destination.exists()
        state.update(busy=False,ssid='Other-test');assert worker.run_once()['status']=='paused' and not destination.exists()
        state['ssid']='Home-test';result=worker.run_once();assert result['status']=='idle',result
        assert (source/'lidar-00000.pcap').read_bytes()==payload
        assert (destination/'synthetic-session/lidar-00000.pcap').read_bytes()==payload
        assert worker.run_once()['sessions'][0]['copied_files']==0
        return {'passed':True,'scope':'synthetic temporary folders and mocked SSID; no real user data or cloud writes',
                'recording_and_wrong_network_blocked':True,'copy_sha256':hashlib.sha256(payload).hexdigest(),
                'original_retained':True,'incremental_repeat_copied_zero':True}
