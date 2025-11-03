from dataclasses import dataclass, field
from stochastic_media.material import SLMMaterial

@dataclass
class LayeredSample:
    """
    Represents a two-material stochastic layered sample.

    Each sample is composed of alternating microscopic layers of two materials
    (material_1 and material_2). The volume fractions of each material control
    the probability of encountering each material type in the stochastic model.
    """
    width: float # Total sample thickness (x)
    dx: float # layer width
    material_1: SLMMaterial
    material_2: SLMMaterial
    volume_fraction_1: float = field(default=1.0) # the volume fraction of
    volume_fraction_2: float = field(init=False)  # not set by user

    def __post_init__(self):
        """
        Validate input and compute the second material's volume fraction.
        This is automatically called after initialization.
        """
        if self.volume_fraction_1>1 or self.volume_fraction_1<0:
            raise ValueError("Volume fractions cannot exceed 1.0 and should be positive")
        self.volume_fraction_2 = 1 - self.volume_fraction_1

    @property
    def layer_xs_variance(self) -> float:
        """
        Compute the variance of the microscopic cross section per layer.

        This gives a measure of heterogeneity in the local attenuation
        coefficient (σ) due to random material mixing.

        Mathematically:
            Var(σ) = f1 * f2 * (σ1 - σ2)^2

        where:
            f1, f2 : volume fractions of materials 1 and 2
            σ1, σ2 : macroscopic cross sections of materials 1 and 2
        """
        return (
            self.volume_fraction_1 * self.volume_fraction_2 *
            (self.material_1.xs - self.material_2.xs) ** 2
        ).item()