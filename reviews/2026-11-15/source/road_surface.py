"""Conservative, observed local road surface for grade transitions.

This does not infer unseen terrain. Broad, low-slope patches must be connected
to the fitted road through mutually consistent overlapping patches. Only then
can they replace the single plane for point-height filtering. Sparse objects
are not required to have vertical columns in normal-ground operation.
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

VERSION = 'connected-road-patches-v1'
CELL_M = 1.
STRIDE = 4
HALF = 4


def _connected_patches(planes,seeds):
    """The original overlap graph, evaluated in batches and traversed in C.

    Topology and the three shared-region tests are unchanged. Undirected graph
    components containing a seed are exactly the original flood-fill result.
    """
    nx,ny=seeds.shape
    valid=np.isfinite(planes[:,:,0])
    if not seeds.any():return np.zeros_like(seeds)
    ix,iy=np.indices((nx,ny));ix=ix.ravel();iy=iy.ravel()
    starts=[];ends=[]
    # Positive directions cover each undirected edge once.
    for dx,dy in ((1,0),(0,1),(2,0),(0,2)):
        jx,jy=ix+dx,iy+dy
        mask=(jx<nx)&(jy<ny)
        ax,ay,bx,by=ix[mask],iy[mask],jx[mask],jy[mask]
        mask=valid[ax,ay]&valid[bx,by]
        ax,ay,bx,by=ax[mask],ay[mask],bx[mask],by[mask]
        difference=planes[ax,ay]-planes[bx,by]
        x=(ax+bx)*STRIDE/2;y=(ay+by)*STRIDE/2-42
        edge=np.stack([np.stack([x,y,np.ones(len(x))],axis=1),
                       np.stack([x+(0 if dx else 2),y+(2 if dx else 0),np.ones(len(x))],axis=1),
                       np.stack([x-(0 if dx else 2),y-(2 if dx else 0),np.ones(len(x))],axis=1)],axis=1)
        error=np.max(np.abs((edge@difference[:,:,None])[:,:,0]),axis=1)
        keep=(error<=.15)&(np.linalg.norm(difference[:,:2],axis=1)<=.13)
        starts.extend((ax[keep]*ny+ay[keep]).tolist());ends.extend((bx[keep]*ny+by[keep]).tolist())
    graph=csr_matrix((np.ones(len(starts),np.uint8),(starts,ends)),shape=(nx*ny,nx*ny))
    _,labels=connected_components(graph,directed=False)
    return np.isin(labels,labels[seeds.ravel()]).reshape(nx,ny)


def _fit_batch(A,z,keep):
    Aw=np.where(keep[:,:,None],A,0.);zw=np.where(keep,z,0.)
    u,s,vh=np.linalg.svd(Aw,full_matrices=False)
    cutoff=np.finfo(float).eps*np.maximum(keep.sum(axis=1),3)*s[:,0]
    proj=np.einsum('bni,bn->bi',u,zw)
    scaled=np.divide(proj,s,out=np.zeros_like(proj),where=s>cutoff[:,None])
    fitted=np.einsum('bji,bj->bi',vh,scaled)
    # Preserve the original solver on rank-deficient / poorly-conditioned
    # geometries instead of allowing padded-batch rounding to decide a fit.
    for i in np.flatnonzero(s[:,-1]<=s[:,0]*1e-6):
        fitted[i]=np.linalg.lstsq(A[i,keep[i]],z[i,keep[i]],rcond=None)[0]
    return fitted


def local_heights(points, abc, evidence=None):
    """Return filter heights and the count of adjusted returns.

    A one-metre low envelope bounds fitting cost and prevents dense returns from
    dominating. A patch needs support in both horizontal axes; narrow objects
    and sparse target-only observations cannot establish a road. Step edges,
    walls and disconnected horizontal surfaces do not join the road model.

    Optional ``evidence`` receives per-point ``supported`` and ``heights`` arrays
    from the connected local patches. Support is independent of whether the
    correction exceeds 0.15 m. Default filter heights and the two return values
    remain unchanged; unsupported evidence entries must never establish height.
    """
    base = points[:, 2] - (points[:, :2] @ np.asarray(abc[:2]) + abc[2])
    if evidence is not None:
        evidence['supported']=np.zeros(len(points),bool)
        evidence['heights']=base.copy()
    if len(points) < 100:
        return base, 0
    usable = ((points[:, 0] > 3) & (points[:, 0] < 124) &
              (np.abs(points[:, 1]) < 39) & np.isfinite(points).all(axis=1))
    p = points[usable]
    if len(p) < 100:
        return base, 0
    # Select the lowest observed point in each cell, preserving its actual XY.
    keys = np.floor(p[:, :2] / CELL_M).astype(np.int32)
    cell=keys[:,0]*84+keys[:,1]+42
    lowest=np.full(128*84,np.inf);np.minimum.at(lowest,cell,p[:,2])
    rows=np.arange(len(p));is_lowest=p[:,2]==lowest[cell]
    first=np.full(128*84,len(p),dtype=np.int64)
    np.minimum.at(first,cell[is_lowest],rows[is_lowest])
    q=p[first[first<len(p)]]
    if len(q) < 30:
        return base, 0
    qkeys = np.floor(q[:, :2]).astype(np.int32)
    # Fixed small grids cover the existing detector ROI.
    nx, ny = 32, 21
    index = np.full((128, 84), -1, np.int32)
    index[qkeys[:, 0], qkeys[:, 1] + 42] = np.arange(len(q))
    planes = np.full((nx, ny, 3), np.nan)
    seeds = np.zeros((nx, ny), bool)
    # Batch the same 8x8-cell least-squares fits, preserving every acceptance test.
    grid_x,grid_y=np.meshgrid(np.arange(1,nx),np.arange(1,ny-1),indexing='ij')
    grid_x,grid_y=grid_x.ravel(),grid_y.ravel();cx,cy=grid_x*STRIDE,grid_y*STRIDE-42
    ox,oy=np.meshgrid(np.arange(-HALF,HALF),np.arange(-HALF,HALF),indexing='ij')
    ids=index[cx[:,None]+ox.ravel(),cy[:,None]+42+oy.ravel()]
    seen=ids>=0
    # Empty/under-supported patches cannot pass the existing acceptance rule.
    # Reject them before gathering coordinates and reducing their spans.
    select=np.flatnonzero(seen.sum(axis=1)>=24)
    qq=q[np.maximum(ids[select],0)];seen=seen[select]
    xy=qq[:,:,:2].transpose(0,2,1)
    span=np.max(np.where(seen[:,None,:],xy,-np.inf),axis=2)-np.min(np.where(seen[:,None,:],xy,np.inf),axis=2)
    accepted=span.min(axis=1)>=4.5
    select=select[accepted];qq=qq[accepted];seen=seen[accepted]
    if len(select):
        x,y=cx[select],cy[select]
        A=np.concatenate([qq[:,:,:2]-np.stack([x,y],axis=1)[:,None,:],np.ones((*qq.shape[:2],1))],axis=2)
        z=qq[:,:,2]
        fit=_fit_batch(A,z,seen)
        refining=np.ones(len(qq),bool)
        for _ in range(2):
            residual=z-np.einsum('bni,bi->bn',A,fit)
            med=np.nanmedian(np.where(seen,residual,np.nan),axis=1)
            keep=seen&(np.abs(residual-med[:,None])<.10)
            refining&=keep.sum(axis=1)>=24
            idx=np.flatnonzero(refining)
            if len(idx):fit[idx]=_fit_batch(A[idx],z[idx],keep[idx])
        residual=z-np.einsum('bni,bi->bn',A,fit)
        keep=seen&(np.abs(residual)<.10);n=keep.sum(axis=1)
        xy=qq[:,:,:2].transpose(0,2,1)
        span=np.max(np.where(keep[:,None,:],xy,-np.inf),axis=2)-np.min(np.where(keep[:,None,:],xy,np.inf),axis=2)
        accepted=(n>=24)&(n/seen.sum(axis=1)>=.65)&(span.min(axis=1)>=4.5)&(np.linalg.norm(fit[:,:2],axis=1)<=.25)
        rms=np.sqrt(np.sum(np.where(keep,residual**2,0),axis=1)/np.maximum(n,1));accepted&=rms<=.055
        plane=fit.copy();plane[:,2]-=fit[:,0]*x+fit[:,1]*y
        keepidx=select[accepted];planes[grid_x[keepidx],grid_y[keepidx]]=plane[accepted]
        error=np.abs(z-(qq[:,:,:2]@np.asarray(abc[:2])+abc[2]))
        anchor=(x<40)&((keep&(error<.12)).sum(axis=1)>=16)&(np.linalg.norm(plane[:,:2]-abc[:2],axis=1)<.09)&accepted
        anchoridx=select[anchor];seeds[grid_x[anchoridx],grid_y[anchoridx]]=True
    connected = _connected_patches(planes,seeds)
    if not connected.any():
        return base, 0
    # Average the consistent overlapping patches at each grid cell. Evaluate
    # their planes at each original point, retaining real vertical residuals.
    coefficients = np.zeros((128, 84, 3))
    weights = np.zeros((128, 84))
    for ix, iy in np.argwhere(connected):
        cx, cy = ix*STRIDE, iy*STRIDE
        coefficients[cx-HALF:cx+HALF, cy-HALF:cy+HALF] += planes[ix, iy]
        weights[cx-HALF:cx+HALF, cy-HALF:cy+HALF] += 1
    location = np.flatnonzero(usable)
    weights_at = weights[keys[:, 0], keys[:, 1]+42]
    supported = weights_at > 0
    chosen = coefficients[keys[supported, 0], keys[supported, 1]+42] / weights_at[supported, None]
    actual = p[supported]
    corrected = actual[:, 2] - np.sum(actual[:, :2] * chosen[:, :2], axis=1) - chosen[:, 2]
    selected = location[supported]
    if evidence is not None:
        evidence['supported'][selected]=True
        evidence['heights'][selected]=corrected
    # Leave the ordinary plane unchanged when local differences are below the
    # ground fitter's scale of uncertainty. This is a grade-transition correction,
    # not a new low-object height threshold on every flat-road frame.
    changed = np.abs(corrected-base[selected]) > .15
    base[selected[changed]] = corrected[changed]
    return base, int(changed.sum())
