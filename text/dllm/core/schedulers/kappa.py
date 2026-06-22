from __future__ import annotations

import dataclasses
import math
from typing import Any, ClassVar, Union

import torch

Number = Union[float, torch.Tensor]


@dataclasses.dataclass
class BaseKappaScheduler:

    __registry__: ClassVar[dict[str, type[BaseKappaScheduler]]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        BaseKappaScheduler.__registry__[cls.__name__] = cls
        BaseKappaScheduler.__registry__[cls.__name__.lower()] = cls

    def __call__(self, t: Number) -> Number:
        return self.kappa(t)

    def kappa(self, t: Number) -> Number:
        t_tensor = torch.as_tensor(
            t,
            dtype=torch.float32,
            device=t.device if isinstance(t, torch.Tensor) else None,
        )
        if not torch.all((0.0 <= t_tensor) & (t_tensor <= 1.0)):
            raise ValueError(f"t={t} not in [0,1]")
        out = self._kappa(t_tensor)
        return out.item() if isinstance(t, float) else out

    def kappa_derivative(self, t: Number) -> Number:
        t_tensor = torch.as_tensor(
            t,
            dtype=torch.float32,
            device=t.device if isinstance(t, torch.Tensor) else None,
        )
        if not torch.all((0.0 <= t_tensor) & (t_tensor <= 1.0)):
            raise ValueError(f"t={t} not in [0,1]")
        out = self._kappa_derivative(t_tensor)
        return out.item() if isinstance(t, float) else out

    def weight(self, t: Number) -> Number:
        return self.kappa_derivative(t) / (1 - self.kappa(t) + 1e-6)

    def _kappa(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _kappa_derivative(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError




@dataclasses.dataclass
class CubicKappaScheduler(BaseKappaScheduler):
    a: float = 1.0
    b: float = 1.0

    def _kappa(self, t: torch.Tensor) -> torch.Tensor:
        return (self.a + 1) * (t**3) - (self.a + self.b + 1) * (t**2) + (self.b + 1) * t

    def _kappa_derivative(self, t: torch.Tensor) -> torch.Tensor:
        return 3 * (self.a + 1) * (t**2) - 2 * (self.a + self.b + 1) * t + (self.b + 1)


@dataclasses.dataclass
class LinearKappaScheduler(CubicKappaScheduler):
    a: float = -1.0
    b: float = 0.0


@dataclasses.dataclass
class CosineKappaScheduler(BaseKappaScheduler):
    def _kappa(self, t: torch.Tensor) -> torch.Tensor:
        return 1.0 - torch.cos(0.5 * math.pi * t)

    def _kappa_derivative(self, t: torch.Tensor) -> torch.Tensor:
        return 0.5 * math.pi * torch.sin(0.5 * math.pi * t)




def get_kappa_scheduler_class(name: str) -> type[BaseKappaScheduler]:
    cls = BaseKappaScheduler.__registry__.get(
        name
    ) or BaseKappaScheduler.__registry__.get(name.lower())
    if cls is None:
        available = sorted(k for k in BaseKappaScheduler.__registry__ if k[0].isupper())
        raise ValueError(f"Unknown scheduler '{name}'. Available: {available}")
    return cls


def make_kappa_scheduler(name: str, **kwargs: Any) -> BaseKappaScheduler:
    cls = get_kappa_scheduler_class(name)
    return cls(**kwargs)



if __name__ == "__main__":
    lin_sched = make_kappa_scheduler("LinearKappaScheduler")
    print("Linear κ(0.5):", lin_sched.kappa(0.5))
    print("Linear w(0.5):", lin_sched.weight(0.5))
    print("Linear κ([.25,.5,.75]):", lin_sched.kappa(torch.tensor([0.25, 0.5, 0.75])))
    print("Linear w([.25,.5,.75]):", lin_sched.weight(torch.tensor([0.25, 0.5, 0.75])))
    print("==========================================")
    cos_sched = make_kappa_scheduler("CosineKappaScheduler")
    print("Cosine κ(0.5):", cos_sched.kappa(0.5))
    print("Cosine w(0.5):", cos_sched.weight(0.5))
    print("Cosine κ([.25,.5,.75]):", cos_sched.kappa(torch.tensor([0.25, 0.5, 0.75])))
    print("Cosine w([.25,.5,.75]):", cos_sched.weight(torch.tensor([0.25, 0.5, 0.75])))
