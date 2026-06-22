
import torch
import torchmetrics


class NLLMetric(torchmetrics.aggregation.MeanMetric):

    def __init__(self, **kwargs):
        kwargs.setdefault("sync_on_compute", True)
        super().__init__(**kwargs)


class PPLMetric(NLLMetric):

    def compute(self) -> torch.Tensor:
        mean_nll = super().compute()
        return torch.exp(mean_nll)
