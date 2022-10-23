from torch.optim.lr_scheduler import LambdaLR


class WarmupScheduler(LambdaLR):
    def __init__(self, optimizer, warmup_steps):
        self.warmup_steps = warmup_steps
        self.coeff = 1. / (warmup_steps ** -0.5)
        super(WarmupScheduler, self).__init__(optimizer, self.lr_lambda)

    def lr_lambda(self, step: int) -> float:
        return self.coeff * min((step+1) ** -0.5, (step+1) * (self.warmup_steps ** -1.5))
