import pytest
import numpy as np
import xarray as xr
from stochastic_media.material import LMMaterial
from stochastic_media.sample import LayeredSample
from stochastic_media.simulator import LMSimulator

xs_1 = xr.DataArray([0.1], coords={'energy': [1e6]})
xs_2 = xr.DataArray([0.05], coords={'energy': [1e6]})

def test_material_defaults():
    mat = LMMaterial()
    assert mat.name == ""
    assert isinstance(mat.xs, xr.DataArray)
    assert mat.xs.size == 0 or mat.xs.values.tolist() == [0.0]

def test_layered_sample_volume_fraction():
    mat1 = LMMaterial(name="A", xs=xs_1)
    mat2 = LMMaterial(name="B", xs=xs_2)
    sample = LayeredSample(width=1.0, dx=0.1, material_1=mat1, material_2=mat2, volume_fraction_1=0.3)
    assert sample.volume_fraction_2 == 0.7

def test_invalid_volume_fraction():
    mat1 = LMMaterial(name="A", xs=xs_1)
    mat2 = LMMaterial(name="B", xs=xs_2)
    with pytest.raises(ValueError):
        LayeredSample(width=1.0, dx=0.1, material_1=mat1, material_2=mat2, volume_fraction_1=1.5)



def test_simulate_attenuation_shape():
    mat1 = LMMaterial(name="A", xs=xs_1)
    mat2 = LMMaterial(name="B", xs=xs_2)
    sample = LayeredSample(width=1.0, dx=0.1, material_1=mat1, material_2=mat2, volume_fraction_1=0.5)
    sim = LMSimulator(sample)
    attenuation = sim.simulate_attenuation(size=1000)
    assert attenuation.shape == (1000,1)
    assert np.all(-np.log(attenuation)/sample.width >= 0.04).item() and np.all(-np.log(attenuation)/sample.width <= 0.11).item()

def test_statistics_output():
    mat1 = LMMaterial(name="A", xs=xs_1)
    mat2 = LMMaterial(name="B", xs=xs_2)
    sample = LayeredSample(width=1.0, dx=0.1, material_1=mat1, material_2=mat2, volume_fraction_1=0.5)
    sim = LMSimulator(sample)
    attenuation = sim.simulate_attenuation(size=1000)
    stats = sim.get_statistics(attenuation)
    assert "mean" in stats and "std" in stats
    assert isinstance(stats["mean"], xr.DataArray)
    assert isinstance(stats["std"], xr.DataArray)
