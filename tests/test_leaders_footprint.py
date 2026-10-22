# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Leader preview geometry shares the computed grid and its edge exclusions."""
import numpy as np
import pytest

from dynamix.devices import register_builtin_devices
from dynamix.model.device import get_device, defaults_for
from tests.test_roi_outputs import _field


@pytest.mark.parametrize('shape',[(128,128),(125,131)])
@pytest.mark.parametrize('edge',['mirror','reflect','periodic','edge_pad','zero_pad'])
def test_preview_matches_grid_and_valid_cell_bounds(clean_registry,shape,edge):
    register_builtin_devices()
    dev=get_device('wavelet_leaders')
    params={**defaults_for(dev),'wavelet':'db2','scale_min':2,'scale_max':8,
            'with_wse':True,'edge':edge,'wse_rule':'fixed','wse_depth':1,'wse_width':2.}
    field=_field(*shape)
    result=dev.compute(field,params)
    for level in range(3):
        fp=dev.footprint(field,params,level,stride=3)
        assert fp['grid']['pad']==result['_pad']
        assert fp['grid']['cell']==2**(level+1)
        from dynamix.core import leaders
        _,pad,work=leaders.grid_layout(shape,3,edge=edge)
        expected=leaders._extend(field.values,work,edge,pad)[::3,::3]
        np.testing.assert_array_equal(fp['padded'],expected)
        valid=result['_valid']['leaders'][level]
        rr,cc=np.nonzero(valid)
        if rr.size:
            y0=rr.min()*2**(level+1)-result['_pad'][0]
            x0=cc.min()*2**(level+1)-result['_pad'][1]
            y1=(rr.max()+1)*2**(level+1)-result['_pad'][0]
            x1=(cc.max()+1)*2**(level+1)-result['_pad'][1]
            assert fp['exclusions'][0]['inner']==(y0,x0,y1-y0,x1-x0)
        assert fp['cell']['three_lambda']==3*2**(level+1)
        assert fp['cell']['atom']==(len(__import__('dynamix.core.dwt',fromlist=['wavelet']).wavelet('db2').dec_lo)-1)*2**(level+1)


def test_region_preview_uses_processing_window_and_has_no_core_exclusion(clean_registry):
    from dynamix.roi.runner import read_processing_window
    register_builtin_devices();dev=get_device('wavelet_leaders')
    params={**defaults_for(dev),'wavelet':'db2','scale_min':2,'scale_max':8,'with_wse':True}
    field=_field(512,512)
    for rect in [(200,180,64,80),(0,0,64,80)]:
        margin=dev.roi_margin(params,field)
        window,_=read_processing_window(field,rect,margin)
        raw=dev.compute(window,params)
        for level in range(3):
            fp=dev.roi_footprint(field,params,rect,level,stride=3)
            assert fp['shown']==margin
            assert fp['grid']['pad']==raw['_pad']
            np.testing.assert_array_equal(fp['padded'],window.values[::3,::3])
            r,c,h,w=rect
            assert fp['outline']==(r-margin,c-margin,h+2*margin,w+2*margin)
            for band in fp['exclusions']:
                y,x,ih,iw=band['inner']
                assert y<=r and x<=c and y+ih>=r+h and x+iw>=c+w
            assert fp['cell']['centre'][0]>=r and fp['cell']['centre'][1]>=c
