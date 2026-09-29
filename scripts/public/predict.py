"""Predict three relative anatomical-retention scores for a PNG or JPEG."""
from common import *
import argparse
from bundle import device_setup, load_bundle


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--image',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu-uuid',help='Optional idle NVIDIA GPU; CPU is the default')
    a=p.parse_args();device=device_setup(a.gpu_uuid)
    import torch
    from predict_image import preprocess_image,summarize_prediction
    torch.set_num_threads(4)
    adapter,meta=load_bundle(a.weights,device)
    pixels,image_meta=preprocess_image(a.image)
    with torch.inference_mode(),torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=device=='cuda'):
        pred=adapter(torch.from_numpy(pixels).to(device))
    result,_=summarize_prediction(pred)
    result.update(input=image_meta,model_config_sha256=sha256_file(a.weights/'config.json'),
        execution='CUDA BF16 image / FP32 readout' if device=='cuda' else 'CPU FP32',
        clinical_support_model=False)
    out=fresh(a.output);write_json(out/'prediction.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
