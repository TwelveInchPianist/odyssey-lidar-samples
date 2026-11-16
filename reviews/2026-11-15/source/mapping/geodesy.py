"""WGS84/ECEF/ENU conversions in metres, with explicit reference frames."""
import math
import numpy as np

A=6378137.
F=1/298.257223563
E2=F*(2-F)

def geodetic_to_ecef(latitude,longitude,altitude):
    v=np.array([latitude,longitude,altitude],float)
    if not np.isfinite(v).all() or not -90<=v[0]<=90 or not -180<=v[1]<=180:raise ValueError('Invalid WGS84 coordinate')
    lat,lon=np.radians(v[:2]);N=A/math.sqrt(1-E2*math.sin(lat)**2)
    return np.array([(N+altitude)*math.cos(lat)*math.cos(lon),(N+altitude)*math.cos(lat)*math.sin(lon),(N*(1-E2)+altitude)*math.sin(lat)])

def ecef_to_geodetic(xyz):
    p=np.asarray(xyz,float)
    if p.shape!=(3,) or not np.isfinite(p).all() or not 6e6<np.linalg.norm(p)<7e6:raise ValueError('Invalid terrestrial ECEF coordinate')
    x,y,z=p;radius=math.hypot(x,y);lon=math.atan2(y,x)
    if radius<1e-6:return np.array([math.copysign(90,z),0.,abs(z)-A*(1-F)])
    lat=math.atan2(z,radius*(1-E2))
    for _ in range(12):
        N=A/math.sqrt(1-E2*math.sin(lat)**2)
        updated=math.atan2(z+E2*N*math.sin(lat),radius)
        if abs(updated-lat)<1e-13:lat=updated;break
        lat=updated
    N=A/math.sqrt(1-E2*math.sin(lat)**2)
    height=radius/math.cos(lat)-N
    return np.array([math.degrees(lat),math.degrees(lon),height])

def enu_from_ecef_rotation(latitude,longitude):
    lat,lon=map(math.radians,(latitude,longitude));sl,cl=math.sin(lat),math.cos(lat);so,co=math.sin(lon),math.cos(lon)
    return np.array([[-so,co,0],[-sl*co,-sl*so,cl],[cl*co,cl*so,sl]])

def ecef_to_enu(xyz,origin_geodetic):
    origin=np.asarray(origin_geodetic,float)
    return (np.asarray(xyz)-geodetic_to_ecef(*origin))@enu_from_ecef_rotation(*origin[:2]).T

def enu_to_ecef(enu,origin_geodetic):
    origin=np.asarray(origin_geodetic,float)
    return np.asarray(enu)@enu_from_ecef_rotation(*origin[:2])+geodetic_to_ecef(*origin)

def horizontal_distance(a,b):
    # Great-circle distance avoids altitude and longitude wraparound artifacts.
    lat1,lat2=math.radians(a[0]),math.radians(b[0]);dl=lat2-lat1;do=math.radians(b[1]-a[1])
    value=math.sin(dl/2)**2+math.cos(lat1)*math.cos(lat2)*math.sin(do/2)**2
    return 2*A*math.asin(math.sqrt(min(1.,max(0.,value))))
