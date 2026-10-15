"""Pinned CPU CXR-NIQE distance; not the full AutoQC wrapper pipeline."""
from concurrent.futures import ProcessPoolExecutor
import importlib.metadata
import importlib.util
import multiprocessing
import os
from pathlib import Path
import sys
import types
import warnings
from baselines.visual import FINDINGS, cli, read, require, sha, write

COMMIT='d8a68cfb0d14f48a1db45edf72a48d7ac2178414'
LIVE='auto_qc/classification/live_qa'
_NIQE=None


def load_official(folder):
    import cv2
    import numpy as np
    import scipy.io
    assets=read(Path(__file__).with_name('assets_manifest.json'))
    for name,item in assets['files'].items():
        path=Path(folder)/name
        require(path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],'Pinned AutoQC asset changed')
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    source=Path(folder)/LIVE;p=scipy.io.loadmat(source/'cxr_model_params.mat')
    require(p['mu_prisparam'].shape==(1,36) and p['cov_prisparam'].shape==(36,36)
            and np.isfinite(p['mu_prisparam']).all() and np.isfinite(p['cov_prisparam']).all(),'Invalid NIQE parameters')
    namespace='_pinned_autoqc_'+COMMIT;package=types.ModuleType(namespace);package.__path__=[str(source)]
    sys.modules[namespace]=package
    for leaf in ('utils','niqe'):
        name=namespace+'.'+leaf;path=source/(leaf+'.py')
        module=importlib.util.module_from_spec(importlib.util.spec_from_file_location(name,path));sys.modules[name]=module
        exec(compile(path.read_bytes(),str(path),'exec'),module.__dict__)
    return sys.modules[namespace+'.niqe'].niqe


def initialize(folder):
    global _NIQE
    os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    _NIQE=load_official(folder)


def prepare_image(path):
    import numpy as np
    from PIL import Image
    with Image.open(path) as image:
        require(image.format=='PNG' and image.mode in ('L','RGB','RGBA'),'Expected 8-bit PNG')
        require(image.width==image.height and image.width>=192,'NIQE needs square presented PNG of side >=192')
        gray=np.asarray(image.convert('L'),dtype=np.uint8)
    require(int(gray.max())!=int(gray.min()),'Constant image has undefined NIQE statistics')
    return gray


def vector_scores(vector):
    import numpy as np
    values=np.asarray(vector,dtype=np.float64)
    require(values.shape==(37,) and np.isfinite(values).all() and values[-1]>=0,'Expected 36 NIQE statistics and nonnegative distance')
    return dict(score=-float(values[-1]),cxr_niqe_distance=float(values[-1]),autoqc_vector_mean_diagnostic=float(values.mean()))


def score_gray(gray):
    try:
        require(_NIQE is not None,'Official NIQE unavailable')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');value=vector_scores(_NIQE(gray))
        value.update(warnings=sorted({f'{w.category.__name__}: {w.message}' for w in caught}),
                     effective_niqe_shape=[int(s-s%96) for s in gray.shape])
        return {finding:dict(value) for finding in FINDINGS}
    except (Exception,SystemExit) as exc:return {'_failure':f'{type(exc).__name__}: {exc}'}


class Backend:
    def __init__(self,folder,config,*,device,gpu_uuid,workers):
        require(device=='cpu' and gpu_uuid is None,'AutoQC must remain CPU only')
        require(config['asset_revision']==COMMIT,'AutoQC revision changed')
        initialize(folder)  # fail before any input if installation or official assets are invalid
        self.pool=ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
                                      initializer=initialize,initargs=(str(folder),)) if workers>1 else None
        self.execution=dict(device='cpu',workers=workers,native_threads_per_worker=1)
        self.protocol=dict(score='negative official niqe(gray)[-1]; one score shared across findings',
            preprocessing='presented PNG -> grayscale uint8; no adapter resize; square side>=192',
            official_internal_trim='bottom/right trimmed to multiple96; two scales; 518 input uses 480x480 upper-left',
            full_autoqc_pipeline=False,parameters='official cxr_model_params.mat; no fitting',
            libraries={name:importlib.metadata.version(name) for name in ('numpy','scipy','Pillow')})

    def prepare(self,row):return prepare_image(row['presented_image_path'])
    def infer(self,prepared):return list(self.pool.map(score_gray,prepared)) if self.pool else [score_gray(p) for p in prepared]
    def close(self):
        if self.pool:self.pool.shutdown(wait=True,cancel_futures=True)
    def preview(self,gray,row,path):
        from PIL import Image
        path.mkdir();Image.fromarray(gray).save(path/'presented_gray.png')
        side=gray.shape[0]-gray.shape[0]%96;Image.fromarray(gray[:side,:side]).save(path/'official_niqe_extent.png')
        write(path/'source.json',dict(input_id=row['input_id'],presented_image_sha256=row['presented_image_sha256'],
            image_shape=list(gray.shape),effective_niqe_shape=[side,side],visual_review_performed=False))


if __name__=='__main__':raise SystemExit(cli('autoqc'))
