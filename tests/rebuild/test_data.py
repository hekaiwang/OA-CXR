import hashlib
import io

import numpy as np
import pytest
from PIL import Image
from oa_cxr.rebuild.data import NearIndex, fingerprint, split_union, retention_targets, variants, render_variant


def masks():
    m=np.zeros((2,80,100),dtype=bool);m[0,10:70,10:40]=True;m[1,10:70,60:90]=True
    return m


def test_split_preserves_original_pixels_and_orders_image_x():
    expected=masks();actual=split_union(expected.any(0))
    assert np.array_equal(actual,expected)


def test_invalid_source_annotation_cannot_become_fake_lung():
    with pytest.raises(ValueError):split_union(np.zeros((80,100),dtype=bool))
    with pytest.raises(ValueError):split_union(np.ones((80,100),dtype=bool))


def test_full_removed_and_asymmetric_native_retention():
    y=retention_targets(masks(),[(0,0,100,80),(0,0,1,1),(0,0,50,80)])
    np.testing.assert_array_equal(y[0],[1,1,1]);np.testing.assert_array_equal(y[1],[0,0,0])
    np.testing.assert_array_equal(y[2],[0,0,0])  # min lung, not area average


def test_baseline_uses_thirteen_distinct_views_without_inputting_crop_metadata():
    plan=variants((100,80));assert len(plan)==13 and len(set(b for _,b in plan))==13
    image=Image.fromarray(np.full((80,100),127,dtype=np.uint8));m=masks()
    x,y=render_variant(image,m,plan[1][1])
    assert x.shape==(256,256) and y.shape==(2,256,256) and y.dtype==np.bool_
    changed=render_variant(image,np.zeros_like(m),plan[1][1])[0]
    np.testing.assert_array_equal(x,changed)  # labels cannot alter model image


def test_letterbox_does_not_clip_content():
    image=Image.fromarray(np.full((80,100),255,dtype=np.uint8))
    x,_=render_variant(image,masks(),(0,0,100,80))
    assert x.max()==255 and (x[25:230,:]>0).all()


def test_near_index_hamming_candidates_matches_bruteforce():
    rng=np.random.default_rng(17);index=NearIndex()
    for i in range(60):
        bits=int.from_bytes(rng.bytes(32),'big')
        row={'file_sha256':f'f{i}','decoded_sha256':f'd{i}','gray256_sha256':f'g{i}',
             'dhash256':f'{bits:064x}','thumbnail_hex':bytes([i]*1024).hex()}
        index.add(row)
    for old in index.rows:
        value=int(old['dhash256'],16)
        positions=rng.choice(256,8,replace=False)
        for p in positions:value^=1<<int(p)
        new={**old,'file_sha256':'new','decoded_sha256':'new','gray256_sha256':'new','dhash256':f'{value:064x}'}
        assert index.matches(new)==[index.rows.index(old)]


def test_image_fingerprints_bind_pixels_and_bytes():
    a=io.BytesIO();Image.fromarray(np.arange(100,dtype=np.uint8).reshape(10,10)).save(a,format='PNG')
    f=fingerprint(a.getvalue())
    assert f['file_sha256']==hashlib.sha256(a.getvalue()).hexdigest()
    assert len(f['dhash256'])==64 and len(f['thumbnail_hex'])==2048


def test_integral_area_matches_fractional_pixel_reference():
    from oa_cxr.rebuild.data import _area, _integral_area
    rng=np.random.default_rng(17);mask=rng.random((23,31))>.6
    integral=np.pad(mask.astype(float).cumsum(0).cumsum(1),((1,0),(1,0)))
    for _ in range(100):
        x=sorted(rng.uniform(-5,36,2));y=sorted(rng.uniform(-5,28,2));box=(x[0],y[0],x[1],y[1])
        assert _integral_area(integral,box)==pytest.approx(_area(mask,box),abs=1e-9)
    assert _integral_area(integral,(5,5,4,4))==0
