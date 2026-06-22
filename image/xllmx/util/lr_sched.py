import math


def adjust_learning_rate(optimizer, it, args):
    if it < args.warmup_iters:
        lr = args.lr * it / args.warmup_iters
    elif it > args.lr_decay_iters:
        lr = args.min_lr
    else:
        decay_ratio = (it - args.warmup_iters) / (args.lr_decay_iters - args.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        lr = args.min_lr + (args.lr - args.min_lr) * coeff

    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def adjust_learning_rate_epoch(optimizer, epoch, args):
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (
            1.0 + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs))
        )
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr
