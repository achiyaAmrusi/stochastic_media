import xarray as xr
from dataclasses import dataclass, field

@dataclass
class SLMMaterial:
    name: str = field(default_factory=str)  # str() -> ""
    xs:xr.DataArray  = field(default_factory=lambda: xr.DataArray([]))

