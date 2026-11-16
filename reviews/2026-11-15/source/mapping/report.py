"""Small offline route overview from the user's own measurements only."""
import html,json,math
from pathlib import Path
import numpy as np
from .geodesy import geodetic_to_ecef,ecef_to_enu

def route_section(folder,summary,map_review=None):
    folder=Path(folder);lines=[];basis='local wheel/motion estimate; drift is not corrected in this overview'
    quality_note='<p>GPS was recorded as an approximate route hint. Its accuracy or measurement timing was not verified, so it was not used as a precise pose constraint.</p>' if summary.get('gps_state')=='coarse_trace' else ''
    try:
        atlas_path=folder/'atlas-route.json'
        if atlas_path.exists() and map_review and map_review.get('status')=='complete':
            atlas=json.loads(atlas_path.read_text(encoding='utf-8'))
            if atlas.get('revision')==map_review.get('revision'):
                areas={}
                for segment in atlas.get('segments',[]):
                    p=[v['position_m'][:2] for v in segment.get('poses',[])]
                    if p:areas.setdefault(segment['component'],[]).append(np.asarray(p,float))
                if areas:
                    text='<h2>Your route in the learned map</h2><p>Corrected keyframe positions from your own lidar and motion evidence. Each area below has its own coordinate frame; gaps remain separate. These estimates are not surveyed ground truth.</p>'+quality_note
                    for number,paths in enumerate(areas.values(),1):text+=f'<p>Mapped area {number}</p>'+trace_svg(paths)
                    return text+'<p><a href="atlas-route.json">Corrected route keyframes</a> · <a href="route-gps.geojson">Original GPS trace</a> · <a href="route-review.json">Route diagnostics</a> · <a href="map-review.json">Map revision report</a></p>'
        gps=folder/'route-gps.geojson'
        if gps.exists():
            data=json.loads(gps.read_text(encoding='utf-8'));origin=None
            for feature in data.get('features',[]):
                geometry=feature.get('geometry',{});coordinates=geometry.get('coordinates',[])
                if geometry.get('type')=='Point':coordinates=[coordinates]
                if not coordinates:continue
                if origin is None:origin=[coordinates[0][1],coordinates[0][0],0.]
                sampled=coordinates[::max(1,math.ceil(len(coordinates)/800))]
                if sampled[-1]!=coordinates[-1]:sampled.append(coordinates[-1])
                line=[ecef_to_enu(geodetic_to_ecef(p[1],p[0],0),origin)[:2] for p in sampled]
                lines.append(np.asarray(line))
            if lines:basis='recorded receiver positions; reported accuracy varies, and gaps remain separate'
        if not lines:
            local=folder/'route-trajectory.jsonl';parts={};counts={};last={}
            if local.exists():
                with local.open(encoding='utf-8') as f:
                    for raw in f:
                        try:row=json.loads(raw)
                        except ValueError:continue
                        key=(row['stream_epoch'],row['segment_id']);point=[row['x_m'],row['y_m']];last[key]=point
                        counts[key]=counts.get(key,0)+1
                        if counts[key]%20==1:parts.setdefault(key,[]).append(point)
                for key,points in parts.items():
                    if points[-1]!=last[key]:points.append(last[key])
                    lines.append(np.asarray(points,float))
        lines=[line for line in lines if len(line) and np.isfinite(line).all()]
    except (OSError,ValueError,KeyError,TypeError):lines=[]
    intro='<h2>Your recorded route</h2><p>'+html.escape(basis)+'.</p>'
    intro+=f'<p>{summary.get("local_segments",0)} valid motion segments; {summary.get("valid_local_distance_m",0)/1000:.2f} km of recorded local travel. GPS: {html.escape(summary.get("gps_state","unavailable").replace("_"," "))}.</p>'+quality_note
    if lines:
        intro+=trace_svg(lines)+'<p>Overview only. Disconnected local segments can share an arbitrary origin; this drawing does not establish that they are the same place.</p>'
    else:intro+='<p>No usable route trace was recorded. This does not prevent preserving available lidar evidence.</p>'
    intro+='<p><a href="route-review.json">Route recording diagnostics</a> · <a href="route-gps.geojson">Export recorded GPS trace</a> · <a href="map-review.json">Map matching and revision report</a></p>'
    return intro

def trace_svg(lines):
    lines=[np.asarray(line,float) for line in lines if len(line)]
    if not lines or any(not np.isfinite(line).all() for line in lines):return '<p>Trace unavailable.</p>'
    points=np.concatenate(lines);lo=points.min(0);hi=points.max(0);center=(lo+hi)/2;scale=min(600/max(hi[0]-lo[0],1),280/max(hi[1]-lo[1],1))
    text='<svg viewBox="0 0 640 320" role="img" aria-label="Your recorded route segments" style="width:100%;max-width:800px;background:#edf3f5;border-radius:8px">'
    for line in lines:
        last=line[-1].copy();line=line[::max(1,math.ceil(len(line)/1000))]
        if not np.array_equal(line[-1],last):line=np.vstack((line,last))
        xy=(line-center)*scale;xy[:,0]+=320;xy[:,1]=160-xy[:,1]
        if len(xy)==1:text+=f'<circle cx="{xy[0,0]:.1f}" cy="{xy[0,1]:.1f}" r="3" fill="#147563"/>'
        else:text+='<polyline points="'+' '.join(f'{x:.1f},{y:.1f}' for x,y in xy)+'" stroke="#147563" stroke-width="2.5" fill="none"/>'
    return text+'</svg>'
