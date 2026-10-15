"""Disease-only CheXpert absence scores; no segmentation model or labels."""
from pathlib import Path
import importlib.metadata
from baselines.visual import FINDINGS, cli, require, runtime_device, sha, write

LABELS = ['Atelectasis','Consolidation','','Pneumothorax','Edema','','',
          'Effusion','Pneumonia','','Cardiomegaly','','','','Lung Lesion',
          'Fracture','Lung Opacity','Enlarged Cardiomediastinum']
HEADS = {'pleural_effusion':'Effusion','pneumothorax':'Pneumothorax','consolidation':'Consolidation'}


def tensor_state(raw,torch):
    require(isinstance(raw,dict) and raw and all(isinstance(k,str) and torch.is_tensor(v) for k,v in raw.items()),
            'Only a tensor state dictionary is permitted')
    prefixed=[k.startswith('module.') for k in raw]
    require(all(prefixed) or not any(prefixed),'Mixed tensor key prefixes')
    return {k[7:] if all(prefixed) else k:v for k,v in raw.items()}


def prepare_image(path):
    import numpy as np
    from PIL import Image
    with Image.open(path) as image:
        require(image.format=='PNG' and image.mode in ('L','RGB','RGBA'),'Expected 8-bit PNG')
        array=np.asarray(image.convert('L').resize((224,224),Image.Resampling.BILINEAR),dtype=np.float32)
    return ((array/255.0*2.0-1.0)*1024.0)[None,:,:]


class Backend:
    def __init__(self,folder,config,*,device,gpu_uuid,workers):
        import torch
        import torchxrayvision as xrv
        require(importlib.metadata.version('torchxrayvision')=='1.5.4','Pinned torchxrayvision 1.5.4 required')
        require(config['asset_revision']=='torchxrayvision-1.5.4-chexpert-only','CheXpert asset revision changed')
        require(list(xrv.models.model_urls['densenet121-res224-chex']['labels'])==LABELS,'Disease head order changed')
        torch.set_num_threads(workers);self.torch=torch;self.device=device
        self.execution=runtime_device(torch,device,gpu_uuid)
        self.model=xrv.models.DenseNet(weights=None,op_threshs=None,apply_sigmoid=False)
        weights=torch.load(Path(folder)/'chexpert_tensor_state.pt',map_location='cpu',weights_only=True)
        self.model.load_state_dict(tensor_state(weights,torch),strict=True)
        self.model.op_threshs=None;self.model.apply_sigmoid=False
        self.model.eval().to(device)
        self.protocol=dict(weight_format='pure tensor state; weights_only=True',labels=LABELS,
            preprocessing='presented PNG -> grayscale -> bilinear224 -> ((x/255)*2-1)*1024; no source image',
            score='1 - sigmoid(raw disease logit)',disease_heads=HEADS,segmentation_loaded=False,
            torchxrayvision=importlib.metadata.version('torchxrayvision'),parameters=sum(p.numel() for p in self.model.parameters()))

    def prepare(self,row):return self.torch.from_numpy(prepare_image(row['presented_image_path']))

    def infer(self,prepared):
        torch=self.torch
        with torch.inference_mode():
            logits=self.model(torch.stack(prepared).to(self.device))
            require(logits.shape==(len(prepared),18) and torch.isfinite(logits).all().item(),'Invalid CheXpert logits')
            prob=torch.sigmoid(logits).cpu().numpy()
        return [{f:{'score':1-float(row[LABELS.index(HEADS[f])]),
                    'disease_probability':float(row[LABELS.index(HEADS[f])])} for f in FINDINGS} for row in prob]

    def preview(self,tensor,row,path):
        import numpy as np
        from PIL import Image
        path.mkdir();array=tensor.numpy()
        Image.fromarray(np.rint((array[0]/1024+1)*127.5).clip(0,255).astype('uint8')).save(path/'processed.png')
        write(path/'source.json',dict(input_id=row['input_id'],presented_image_sha256=row['presented_image_sha256'],
            tensor_shape=list(array.shape),preview_sha256=sha(path/'processed.png'),actual_tensor=True,visual_review_performed=False))


if __name__=='__main__':raise SystemExit(cli('chexpert'))
