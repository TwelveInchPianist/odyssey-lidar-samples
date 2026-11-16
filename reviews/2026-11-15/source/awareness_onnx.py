"""Pinned ONNX adapter for offline Windows DirectML qualification."""
import ast,hashlib,time,threading
from pathlib import Path
import numpy as np
from awareness_vision import preprocess_bgr,MODEL_CLASSES
QUALIFIED_MODEL_SHA256='451353a9cb5d793a2db78dd584d8ebef873960adcfc047cb53eb8cfd4f4fb48d'

class OnnxCameraDetector:
 def __init__(self,model,expected_sha256,provider='DmlExecutionProvider',profile=False):
  if expected_sha256!=QUALIFIED_MODEL_SHA256:raise ValueError('Camera model is not the qualified FP32 export')
  import onnxruntime as ort
  p=Path(model);self.sha256=hashlib.sha256(p.read_bytes()).hexdigest()
  if self.sha256!=expected_sha256:raise ValueError('Pinned ONNX digest mismatch')
  if provider not in ort.get_available_providers():raise RuntimeError('Required provider unavailable: '+provider)
  options=ort.SessionOptions();options.enable_mem_pattern=False;options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL;options.intra_op_num_threads=1;options.inter_op_num_threads=1;options.enable_profiling=profile
  self.session=ort.InferenceSession(str(p),sess_options=options,providers=[provider,'CPUExecutionProvider'] if provider!='CPUExecutionProvider' else [provider]);self.provider=provider;self.lock=threading.Lock();self.metrics={}
  if provider not in self.session.get_providers():raise RuntimeError('Requested provider unavailable after initialization')
  self.session.disable_fallback()
  if self.session.get_inputs()[0].type!='tensor(float)':raise ValueError('Only qualified FP32 model input is accepted')
  if self.session.get_inputs()[0].shape!=[1,3,832,1280]:raise ValueError('Unexpected image contract')
  names=self.session.get_modelmeta().custom_metadata_map['names']
  if len(names)>10000:raise ValueError('Oversized class metadata')
  self.names=ast.literal_eval(names)
  if not isinstance(self.names,dict) or len(self.names)!=80:raise ValueError('Expected fixed COCO class metadata')
 def predict(self,images):
  outputs=[]
  with self.lock:
   for image in images:
    start=time.perf_counter();tensor,ratio,(left,top)=preprocess_bgr(image);prep=time.perf_counter()
    raw=self.session.run(None,{'images':tensor})[0];ran=time.perf_counter()
    if raw.shape!=(1,84,21840) or not np.isfinite(raw).all():raise ValueError('Malformed model output')
    rows=raw[0].T;classes=np.argmax(rows[:,4:],axis=1);scores=rows[np.arange(len(rows)),classes+4];keep=(scores>.2)&np.isin(classes,MODEL_CLASSES)
    boxes=rows[keep,:4].copy();classes=classes[keep];scores=scores[keep]
    centers=boxes[:,:2].copy();half=boxes[:,2:]/2;boxes[:,:2]=centers-half;boxes[:,2:]=centers+half
    offset=classes.astype(np.float32)[:,None]*7680;shifted=boxes+offset;order=np.argsort(-scores,kind='stable');chosen=[]
    while len(order) and len(chosen)<300:
     i=int(order[0]);chosen.append(i);rest=order[1:]
     if not len(rest):break
     a=shifted[i];b=shifted[rest];wh=np.maximum(0,np.minimum(a[2:],b[:,2:])-np.maximum(a[:2],b[:,:2]));inter=wh[:,0]*wh[:,1]
     area_a=(a[2]-a[0])*(a[3]-a[1]);area_b=(b[:,2]-b[:,0])*(b[:,3]-b[:,1]);iou=inter/np.maximum(area_a+area_b-inter,1e-9);order=rest[iou<=.7]
    h,w=image.shape[:2];result=[]
    for i in chosen:
     b=(boxes[i]-np.array([left,top,left,top]))/ratio;b[[0,2]]=np.clip(b[[0,2]],0,w);b[[1,3]]=np.clip(b[[1,3]],0,h)
     result.append({'class_id':int(classes[i]),'class_name':self.names[int(classes[i])],'score':float(scores[i]),'xyxy':b.tolist()})
    outputs.append(result);self.metrics={'preprocess_ms':(prep-start)*1000,'inference_ms':(ran-prep)*1000,'postprocess_ms':(time.perf_counter()-ran)*1000,'total_ms':(time.perf_counter()-start)*1000,'input_tensor_sha256':hashlib.sha256(tensor.tobytes()).hexdigest()}
  return outputs
