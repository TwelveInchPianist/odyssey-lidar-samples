"""Approximate solar gating from fresh GPS; never presents image brightness as lux."""
import math,datetime
def solar_elevation(utc_seconds,latitude,longitude):
    if not all(math.isfinite(x) for x in (utc_seconds,latitude,longitude)) or not -90<=latitude<=90 or not -180<=longitude<=180:raise ValueError('Invalid solar input')
    t=datetime.datetime.fromtimestamp(utc_seconds,datetime.timezone.utc)
    hour=t.hour+t.minute/60+t.second/3600
    gamma=2*math.pi/365*(t.timetuple().tm_yday-1+(hour-12)/24)
    eq=229.18*(.000075+.001868*math.cos(gamma)-.032077*math.sin(gamma)-.014615*math.cos(2*gamma)-.040849*math.sin(2*gamma))
    decl=.006918-.399912*math.cos(gamma)+.070257*math.sin(gamma)-.006758*math.cos(2*gamma)+.000907*math.sin(2*gamma)-.002697*math.cos(3*gamma)+.00148*math.sin(3*gamma)
    angle=math.radians(((hour*60+eq+4*longitude)%1440)/4-180);lat=math.radians(latitude)
    return math.degrees(math.asin(max(-1,min(1,math.sin(lat)*math.sin(decl)+math.cos(lat)*math.cos(decl)*math.cos(angle)))))

def light_evidence(mode,messages,now_ns):
    if mode in ('off','on'):return {'basis':'manual','dark':mode=='on','source_ns':now_ns}
    if mode!='auto':return None
    for name in ('gpsLocationExternal','gpsLocation'):
        row=messages.get(name,{})
        age=now_ns-row.get('receipt_ns',0);d=row.get('data',{})
        if not row.get('valid') or not 0<=age<=60_000_000_000:continue
        if d.get('hasFix') is not True and not (isinstance(d.get('flags'),int) and d['flags']&1):continue
        try:
            accuracy=float(d['horizontalAccuracy']);seconds=float(d['unixTimestampMillis'])/1000
            if not math.isfinite(accuracy) or not 0<=accuracy<=100 or not 1577836800<=seconds<=4102444800:continue
            elevation=solar_elevation(seconds,float(d['latitude']),float(d['longitude']))
        except (KeyError,TypeError,ValueError,OverflowError):continue
        return {'basis':'solar_elevation','dark':elevation<=-3,'source_ns':row['receipt_ns'],'solar_elevation_deg':elevation,'gps_service':name,'note':'Approximate solar state, not measured ambient illumination'}
    return None
