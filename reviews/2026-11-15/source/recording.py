"""Read-only recorded-frame iterator shared by offline map and exam modules.

Context lookup uses a disposable disk index of byte offsets. Long route logs do
not need to fit in RAM; source recordings are never rewritten or truncated.
"""
import json,sqlite3,tempfile
from pathlib import Path
import numpy as np
from frames import FrameAssembler,decode_with_time
from m1_protocol import read_pcap
from offline_jobs import check_work


def frames(folder,cancel=None,strict=False,diagnostics=None):
    folder=Path(folder);cfg=json.loads((folder/'session.json').read_text(encoding='utf-8-sig'))
    diagnostics={} if diagnostics is None else diagnostics
    diagnostics.update(unmatched_frames=0,contexts_without_raw_frames=0)
    path=folder/'frame-contexts.jsonl'
    if not path.exists():return
    with tempfile.TemporaryDirectory(prefix='odyssey-frame-index-') as temporary:
        db=sqlite3.connect(Path(temporary)/'offsets.sqlite')
        try:
            # This index is disposable; crash recovery uses the original JSONL.
            db.execute('PRAGMA journal_mode=OFF');db.execute('PRAGMA cache_size=-4096')
            db.execute('CREATE TABLE contexts(sequence INTEGER PRIMARY KEY,device_key TEXT,offset INTEGER,length INTEGER)')
            with path.open('rb') as context_file:
                line_number=0
                while True:
                    check_work(cancel);offset=context_file.tell();line=context_file.readline()
                    if not line:break
                    line_number+=1
                    try:
                        record=json.loads(line);key=record['quality']['first_device_ns']
                        if type(key) is not int:raise ValueError('Invalid device timestamp')
                        db.execute('INSERT INTO contexts VALUES(?,?,?,?)',(line_number,str(key),offset,len(line)))
                    except (ValueError,KeyError,TypeError):
                        if strict:raise ValueError('Malformed frame context; replay coverage cannot be established')
                db.execute('CREATE INDEX device_order ON contexts(device_key,sequence)');db.commit()
                assembler=FrameAssembler()
                for capture in sorted(folder.glob('lidar-*.pcap')):
                    for wall,source,port,payload in read_pcap(capture):
                        check_work(cancel)
                        if port!=cfg['msop_port'] or source[0]!=cfg['sensor_ip']:continue
                        for packets,q in assembler.push(payload,wall):
                            row=db.execute('SELECT sequence,offset,length FROM contexts WHERE device_key=? ORDER BY sequence LIMIT 1',(str(q['first_device_ns']),)).fetchone()
                            if not row:diagnostics['unmatched_frames']+=1;continue
                            sequence,offset,length=row;db.execute('DELETE FROM contexts WHERE sequence=?',(sequence,))
                            context_file.seek(offset);record=json.loads(context_file.read(length))
                            points,stamps=decode_with_time(packets)
                            if not len(points):continue
                            ctx=record['context'];ctx['path']=np.asarray(ctx.get('path',[]),dtype=float).reshape(-1,2)
                            quality={**record['quality'],'context_line':sequence,'raw_capture_completion_part':capture.name,'raw_capture_last_wall_ns':wall}
                            yield points,stamps,ctx,quality
        finally:
            try:diagnostics['contexts_without_raw_frames']=db.execute('SELECT COUNT(*) FROM contexts').fetchone()[0]
            except sqlite3.Error:pass
            db.close()
