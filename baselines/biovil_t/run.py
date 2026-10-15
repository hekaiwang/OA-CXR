"""Official frozen BioViL-T prompt margin, without temporal prior or fitting."""
import hashlib
import importlib.metadata
from pathlib import Path
from baselines.visual import FINDINGS, cli, require, runtime_device, sha, write

REVISION='692f09e9be1bfe5fdd5f3efdd0e1eca7d2c10b23'
PROMPTS=[phrase.format(finding=f.replace('_',' ')) for f in FINDINGS
         for phrase in ('There is no {finding}.','There is {finding}.')]


class Backend:
    def __init__(self,folder,config,*,device,gpu_uuid,workers):
        import torch
        from transformers import AutoModel,AutoTokenizer
        from health_multimodal.image.model.model import ImageModel
        from health_multimodal.image.model.types import ImageEncoderType
        from health_multimodal.image.data.io import load_image
        from health_multimodal.image.data.transforms import create_chest_xray_transform_for_inference
        require(config['asset_revision']==REVISION,'BioViL-T revision changed')
        require(importlib.metadata.version('hi-ml-multimodal')=='0.2.2','Pinned hi-ml-multimodal 0.2.2 required')
        torch.set_num_threads(workers);self.torch=torch;self.device=device
        self.execution=runtime_device(torch,device,gpu_uuid)
        self.model=ImageModel(img_encoder_type=ImageEncoderType.RESNET50_MULTI_IMAGE,joint_feature_size=128,
            pretrained_model_path=Path(folder)/'biovil_t_image_model_proj_size_128.pt').eval().to(device)
        text=AutoModel.from_pretrained(folder,trust_remote_code=True,local_files_only=True,use_safetensors=True).eval().to(device)
        tokenizer=AutoTokenizer.from_pretrained(folder,local_files_only=True);tokens=tokenizer(PROMPTS,return_tensors='pt',padding=True)
        with torch.inference_mode():
            self.text=text.get_projected_text_embeddings(input_ids=tokens.input_ids.to(device),attention_mask=tokens.attention_mask.to(device))
        require(self.text.shape==(6,128) and torch.isfinite(self.text).all().item(),'Invalid BioViL-T text embeddings')
        self.load_image=load_image;self.transform=create_chest_xray_transform_for_inference(resize=512,center_crop_size=448)
        self.protocol=dict(prompts=PROMPTS,score='negative prompt similarity minus positive prompt similarity',
            preprocessing='official load_image min-max uint8 -> L; resize512 -> center crop448 -> ToTensor -> repeat3',
            crop_limitation='Official transform retains448/512 of each presented side; never uncropped source image',
            dtype='float32',temporal_prior=None,hi_ml_multimodal='0.2.2',
            parameters=sum(p.numel() for p in self.model.parameters())+sum(p.numel() for p in text.parameters()))
        del text

    def prepare(self,row):
        from PIL import Image
        with Image.open(row['presented_image_path']) as im:
            require(im.format=='PNG' and im.mode in ('L','RGB','RGBA'),'Expected 8-bit PNG')
            low,high=im.convert('L').getextrema()
            require(low<high,'Constant image has undefined official min-max normalization')
        tensor=self.transform(self.load_image(Path(row['presented_image_path'])))
        require(tensor.shape==(3,448,448) and self.torch.isfinite(tensor).all().item(),'Invalid BioViL-T tensor')
        return tensor

    def infer(self,prepared):
        torch=self.torch
        with torch.inference_mode():
            emb=self.model(torch.stack(prepared).to(self.device)).projected_global_embedding
            emb=torch.nn.functional.normalize(emb,dim=-1);sims=(emb@self.text.T).cpu().numpy()
        return [{f:dict(score=float(v[2*j]-v[2*j+1]),negative_similarity=float(v[2*j]),positive_similarity=float(v[2*j+1]))
                 for j,f in enumerate(FINDINGS)} for v in sims]

    def preview(self,tensor,row,path):
        import numpy as np
        from PIL import Image
        path.mkdir();array=tensor.numpy();Image.fromarray(np.rint(array[0]*255).clip(0,255).astype('uint8')).save(path/'processed.png')
        write(path/'source.json',dict(input_id=row['input_id'],presented_image_sha256=row['presented_image_sha256'],
            tensor_shape=list(array.shape),tensor_sha256=hashlib.sha256(array.tobytes()).hexdigest(),
            preview_sha256=sha(path/'processed.png'),actual_tensor=True,visual_review_performed=False))


if __name__=='__main__':raise SystemExit(cli('biovil_t'))
