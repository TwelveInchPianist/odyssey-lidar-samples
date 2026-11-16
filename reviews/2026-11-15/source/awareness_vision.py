"""Camera instance evidence and short temporal confirmation for awareness.

No range is inferred from box size. Camera-only observations can advise about a
road user while lidar is unavailable. Motion of the image is not ground speed.
Pretrained confidence scores are recorded as scores, never calibrated probabilities.
"""
from collections import deque
from dataclasses import dataclass,field
import hashlib,math
from pathlib import Path

MODEL_CLASSES=[0,1,2,3,5,7,15,16,17,18,19,20,21,22,23]
CLASS_LABEL={0:'person',1:'bicycle',2:'vehicle',3:'motorcycle',5:'vehicle',7:'vehicle',
             **{k:'animal' for k in range(15,24)}}
NAME_LABEL={'person':'person','bicycle':'bicycle','motorcycle':'motorcycle',
            **{k:'scooter' for k in ('scooter','motor scooter','motorized scooter','e-scooter')},
            **{k:'vehicle' for k in ('car','truck','bus')},
            **{k:'animal' for k in ('bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','deer')},
            **{k:'furniture' for k in ('tree','bush','pole','traffic sign','building','guardrail','car seat','headrest')}}
CONFIRM_SCORE=.45
MAX_GAP_NS=600_000_000


def preprocess_bgr(image,shape=(832,1280)):
    """One bit-exact bilinear resize contract for ARM and x86 inference."""
    import cv2,numpy as np
    if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3:
        raise ValueError('Expected native BGR uint8 image')
    h,w=image.shape[:2];ratio=min(shape[0]/h,shape[1]/w)
    nw,nh=round(w*ratio),round(h*ratio)
    if (nw,nh)!=(w,h):image=cv2.resize(image,(nw,nh),interpolation=cv2.INTER_LINEAR_EXACT)
    dw,dh=(shape[1]-nw)/2,(shape[0]-nh)/2
    left,right=round(dw-.1),round(dw+.1);top,bottom=round(dh-.1),round(dh+.1)
    image=cv2.copyMakeBorder(image,top,bottom,left,right,cv2.BORDER_CONSTANT,value=(114,114,114))
    tensor=np.ascontiguousarray(image[:,:,::-1].transpose(2,0,1)[None],dtype=np.float32)/255.
    return tensor,ratio,(left,top)


def iou(a,b):
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    area=lambda q:max(0,q[2]-q[0])*max(0,q[3]-q[1])
    return intersection/max(area(a)+area(b)-intersection,1e-9)


def contained_fraction(candidate,reference):
    a,b=candidate['bbox_xyxy'],reference['bbox_xyxy']
    area=max(0,a[2]-a[0])*max(0,a[3]-a[1])
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    return intersection/area if area else 0.



def instances(boxes,width,height):
    valid=[]
    for box in boxes:
        try:
            cls=int(box['class_id']);score=float(box['score']);xy=list(map(float,box['xyxy']))
            name=box.get('class_name')
            label=NAME_LABEL.get(name) if name is not None else CLASS_LABEL.get(cls)
            if (label is None or len(xy)!=4 or not all(math.isfinite(x) for x in xy+[score])
                    or not 0<=score<=1 or not 0<=xy[0]<xy[2]<=width or not 0<=xy[1]<xy[3]<=height):continue
        except (KeyError,ValueError,TypeError):continue
        valid.append({'class_id':cls,'label':label,'score':score,'xyxy':xy,
                      'raw_class':box.get('class_name',str(cls))})
    # Track the actual detector instances. Semantic relationships are composed
    # after tracking; changing bicycle evidence cannot resize or rename a person.
    return valid


@dataclass
class VisualTrack:
    id:int
    first_ns:int
    last_ns:int
    box:list
    label:str
    support:int=1
    scores:deque=field(default_factory=lambda:deque(maxlen=3))


class VisionTracks:
    def __init__(self,camera,*,fx,cx):
        if camera not in ('narrow','wide'):raise ValueError('Road cameras only')
        if not math.isfinite(fx) or fx<=0 or not math.isfinite(cx):raise ValueError('Invalid camera intrinsics')
        self.camera=camera;self.fx=fx;self.cx=cx;self.tracks={};self.next_id=1;self.last_ns=None
        self.rider_relations={}

    def _road_users(self,observations,source_ns):
        rider_labels={'bicycle':'cyclist','motorcycle':'motorcycle','scooter':'scooter'}
        people=[i for i,o in enumerate(observations) if o['label']=='person' and o.get('source_frame_valid') is True]
        bikes=[i for i,o in enumerate(observations) if o['label'] in rider_labels and o.get('source_frame_valid') is True]
        hits={p:[b for b in bikes if iou(observations[p]['bbox_xyxy'],observations[b]['bbox_xyxy'])>.10] for p in people}
        reverse={b:[p for p in people if b in hits[p]] for b in bikes}
        def child_options(indices):
            confirmed=[i for i in indices if observations[i]['valid'] is True]
            if len(confirmed)!=1:return indices
            chosen=confirmed[0]
            # Retain every tentative observation. Only a same-class,
            # overwhelmingly contained child hypothesis is non-vetoing here.
            # It becomes independently ambiguous as soon as it is confirmed.
            return [i for i in indices if i==chosen or observations[i]['valid'] is True
                    or observations[i]['label']!=observations[chosen]['label']
                    or contained_fraction(observations[i],observations[chosen])<.9]
        pp={p:child_options(v) for p,v in hits.items()}
        # A tentative second person can be the true owner of a new bicycle.
        # Never disregard that ownership ambiguity because its score is lower.
        bp=reverse
        pairs=[(p,v[0]) for p,v in pp.items() if len(v)==1 and bp[v[0]]==[p]]
        # A missing or tentative alternative supplies no new pairing evidence,
        # but also does not erase the recent observations that actually happened.
        relations={k:v for k,v in self.rider_relations.items() if 0<=source_ns-v[0]<=MAX_GAP_NS}
        bykey={o['key']:i for i,o in enumerate(observations)}
        for key in list(relations):
            p,b=(bykey.get(k) for k in key)
            if p is None or b is None:continue
            if observations[p].get('source_frame_valid') is not True or observations[b].get('source_frame_valid') is not True:continue
            contradiction=(observations[p]['valid'] is True and observations[b]['valid'] is True and
                           (p not in hits or b not in hits[p]))
            confirmed_ambiguity=(sum(observations[j]['valid'] is True for j in hits.get(p,[]))>1 or
                                 len(reverse.get(b,[]))>1)
            if contradiction or confirmed_ambiguity:relations.pop(key)
        replacements={};represented_bikes=set()
        for p,b in pairs:
            person,bike=observations[p],observations[b];key=(person['key'],bike['key'])
            previous=relations.get(key)
            count=previous[1]+1 if previous and 0<source_ns-previous[0]<=MAX_GAP_NS else 1
            relations[key]=(source_ns,count)
            if count<3 or not person['valid'] or not bike['valid']:continue
            a,c=person['bbox_xyxy'],bike['bbox_xyxy']
            replacements[p]={**person,'label':rider_labels[bike['label']],'raw_class':'person+'+bike['label'],
                'bbox_xyxy':[min(a[0],c[0]),min(a[1],c[1]),max(a[2],c[2]),max(a[3],c[3])],
                'components':[person,bike],'relationship_samples':count,
                'relationship_basis':'persistent_unique_supported_image_pair_not_riding_pose_ground_truth',
                'relationship_tentative_alternatives':[observations[i]['key'] for i in sorted(set(hits[p]+reverse[b]))
                     if i not in (p,b) and observations[i]['valid'] is not True]}
            represented_bikes.add(b)
        self.rider_relations=relations
        return [replacements.get(i,o) for i,o in enumerate(observations) if i not in represented_bikes]

    def update(self,boxes,source_ns,receipt_ns,width,height,*,frame_id=None,valid=True,image_motion=None):
        if type(source_ns) is not int or type(receipt_ns) is not int or not 0<source_ns<=receipt_ns:
            raise ValueError('Camera needs an actual causal source timestamp')
        if self.last_ns is not None and source_ns<=self.last_ns:
            self.tracks.clear();self.rider_relations.clear();self.last_ns=source_ns
            return []  # duplicate/reset frame does not manufacture persistence
        previous_ns=self.last_ns;self.last_ns=source_ns
        self.tracks={k:t for k,t in self.tracks.items() if source_ns-t.last_ns<=MAX_GAP_NS}
        if (isinstance(image_motion,dict) and image_motion.get('valid') is True
                and image_motion.get('source_ns')==source_ns
                and image_motion.get('previous_source_ns')==previous_ns):
            from camera_motion import transform_box
            matrix=image_motion['matrix']
            if (len(matrix)!=2 or any(len(row)!=3 for row in matrix)
                    or not all(math.isfinite(x) for row in matrix for x in row)):
                raise ValueError('Invalid measured image transform')
            for track in self.tracks.values():
                track.box=transform_box(track.box,matrix)
        detections=instances(boxes,width,height);pairs=[]
        for j,d in enumerate(detections):
            for k,t in self.tracks.items():
                # Bicycle/motorcycle/scooter classification can flicker while
                # the same measured two-wheeler remains in view. Preserve its
                # identity, but reconfirm the new class below before speaking it.
                compatible=(t.label==d['label'] or
                            {t.label,d['label']} <= {'bicycle','motorcycle','scooter'})
                overlap=iou(t.box,d['xyxy'])
                if compatible and overlap>=.15:
                    # Match strong observations first; a low-score duplicate
                    # must not steal an established ID from a strong detection.
                    pairs.append((int(d['score']>=CONFIRM_SCORE),overlap,k,j))
        pairs.sort(reverse=True);assigned={};used=set()
        for _,overlap,k,j in pairs:
            if k not in used and j not in assigned:assigned[j]=k;used.add(k)
        output=[]
        for j,d in enumerate(detections):
            if j in assigned:
                t=self.tracks[assigned[j]]
                if t.label!=d['label']:t.scores.clear()
                t.support+=1;t.last_ns=source_ns;t.box=d['xyxy'];t.label=d['label']
            else:
                t=VisualTrack(self.next_id,source_ns,source_ns,d['xyxy'],d['label'])
                self.next_id+=1;self.tracks[t.id]=t
            t.scores.append(d['score'])
            confirmed=len(t.scores)>=3 and min(t.scores)>=CONFIRM_SCORE
            label=d['label'];box=d['xyxy']
            mid=(box[0]+box[2])/2
            output.append({'key':'camera:'+self.camera+':'+str(t.id),'sensor':'camera:'+self.camera,
                'source_ns':source_ns,'receipt_ns':receipt_ns,'valid':bool(valid and confirmed),
                'source_frame_valid':bool(valid),
                'label':label,'class_confirmed':confirmed,'support':t.support,
                'history_s':(source_ns-t.first_ns)/1e9,'bearing_deg':math.degrees(math.atan2(self.cx-mid,self.fx)),
                'bearing_basis':'nominal_camera_direction_only','position_m':None,'range_m':None,
                'frame_id':frame_id,'camera':self.camera,'bbox_xyxy':box,
                'association_bbox_xyxy':d['xyxy'],
                'image_size':[width,height],'raw_class':d['raw_class'],'detector_score':d['score'],
                'score_kind':'pretrained_detector_score_not_calibrated_probability',
                # This identifies this camera instance only. It cannot suppress
                # another object by geographic location or image-box overlap.
                'furniture_positive':confirmed and d['label']=='furniture',
                'furniture_live_evidence':confirmed and d['label']=='furniture',
                'motion_confirmed':False,'ground_speed_mps':None,
                'vru_evidence':confirmed and label in ('person','cyclist','bicycle','motorcycle','scooter','animal')})
        return self._road_users(output,source_ns)


class CameraDetector:
    """Pinned local weights, explicit execution device, native-image coordinates."""
    def __init__(self,weights,expected_sha256,device='mps'):
        weights=Path(weights)
        if not weights.is_file():raise FileNotFoundError('Pinned camera model is missing')
        actual=hashlib.sha256(weights.read_bytes()).hexdigest()
        if actual!=expected_sha256:raise ValueError('Camera model digest mismatch')
        from ultralytics import YOLO
        self.model=YOLO(str(weights));self.device=device;self.sha256=actual
        self.classes=MODEL_CLASSES

    def _predict(self,images):
        import numpy as np,torch
        prepared=[preprocess_bgr(image) for image in images]
        tensor=torch.from_numpy(np.concatenate([row[0] for row in prepared])).to(self.device)
        predictions=self.model.predict(tensor,device=self.device,imgsz=(832,1280),classes=self.classes,
                                       conf=.2,iou=.7,verbose=False,save=False,half=False)
        return predictions,prepared

    def _boxes(self,image,prepared,prediction):
        import numpy as np
        _,ratio,(left,top)=prepared
        h,w=image.shape[:2];boxes=[]
        for k,conf,xy in zip(prediction.boxes.cls.cpu(),prediction.boxes.conf.cpu(),prediction.boxes.xyxy.cpu()):
            box=(xy.numpy()-np.array([left,top,left,top]))/ratio
            box[[0,2]]=np.clip(box[[0,2]],0,w);box[[1,3]]=np.clip(box[[1,3]],0,h)
            boxes.append({'class_id':int(k),'class_name':self.model.names[int(k)],'score':float(conf),'xyxy':box.tolist()})
        return boxes

    def predict(self,images):
        predictions,prepared=self._predict(images)
        result=[]
        for image,prep,p in zip(images,prepared,predictions):result.append(self._boxes(image,prep,p))
        return result


class SemanticCameraDetector(CameraDetector):
    """Fixed-vocabulary pretrained instance model; no fitting or prompt search.

    Its proposals are not independent labels, and a furniture mask never by
    itself vetoes a different camera or lidar object. Runtime adoption requires
    the same encounter and target-device checks as the baseline detector.
    """
    NAMES=['person','bicycle','motorcycle','car','truck','bus','deer','dog',
           'tree','bush','pole','traffic sign','building','guardrail','car seat','headrest']

    def __init__(self,weights,expected_sha256,device='mps'):
        weights=Path(weights)
        if not weights.is_file():raise FileNotFoundError('Pinned semantic model is missing')
        actual=hashlib.sha256(weights.read_bytes()).hexdigest()
        if actual!=expected_sha256:raise ValueError('Semantic model digest mismatch')
        from ultralytics import YOLOE
        self.model=YOLOE(str(weights));self.device=device;self.sha256=actual
        self.model.set_classes(self.NAMES)
        self.classes=None

    def predict_instances(self,images):
        """Same fixed model, retaining instance masks for point association.

        Masks remain in the letterboxed model coordinate system. A mask is a
        hypothesis, not clearance; consumers must preserve overlapping and
        unclaimed lidar returns. No threshold or model weights change here.
        """
        import numpy as np
        predictions,prepared=self._predict(images);output=[]
        for image,prep,p in zip(images,prepared,predictions):
            boxes=self._boxes(image,prep,p)
            masks=(p.masks.data.cpu().numpy().astype(bool) if p.masks is not None
                   else np.empty((0,*prep[0].shape[-2:]),dtype=bool))
            if len(masks)!=len(boxes):raise ValueError('Instance masks and boxes disagree')
            output.append({'boxes':boxes,'masks':masks,'resize_ratio':prep[1],
                'padding_xy':list(prep[2]),'model_image_size':[prep[0].shape[-1],prep[0].shape[-2]],
                'native_image_size':[image.shape[1],image.shape[0]],'semantics':'pretrained_hypotheses'})
        return output
