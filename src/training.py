from __future__ import absolute_import, division, print_function

import math

import torch
from torch.nn import MSELoss
from transformers import get_linear_schedule_with_warmup

from src.common import progress_bar


def is_no_decay_name(name):
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight", "layer_norm.weight", "layer_norm.bias"]
    return any(nd in name for nd in no_decay)


def build_optimizer_and_scheduler(model, train_loader, args):
    grouped = [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and not is_no_decay_name(n)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and is_no_decay_name(n)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(grouped, lr=args.learning_rate)
    steps_per_epoch = max(1, math.ceil(len(train_loader) / max(1, args.gradient_accumulation_step)))
    total_steps = steps_per_epoch * args.n_epochs
    warmup_steps = int(args.warmup_proportion * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(1, total_steps),
    )
    return optimizer, scheduler


def run_batch(model, batch, device, diffusion_loss_weight=0.0, return_aux=False):
    input_ids, attention_mask, visual, acoustic, sequence_lengths, labels = tuple(t.to(device) for t in batch)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        visual=visual,
        acoustic=acoustic,
        sequence_lengths=sequence_lengths,
        return_aux=return_aux or diffusion_loss_weight > 0,
    )
    if isinstance(outputs, dict):
        logits = outputs["logits"]
        diffusion_loss = outputs["visual_diffusion_loss"] + outputs["acoustic_diffusion_loss"]
    else:
        logits = outputs
        diffusion_loss = None

    loss = MSELoss()(logits.view(-1), labels.view(-1))
    if diffusion_loss is not None and diffusion_loss_weight > 0:
        loss = loss + diffusion_loss_weight * diffusion_loss
    return loss, logits, labels


def train_epoch(model, train_loader, optimizer, scheduler, args, device, desc="Train"):
    model.train()
    total_loss = 0.0
    total_count = 0
    optimizer.zero_grad()

    iterator = train_loader
    if args.batch_progress:
        iterator = progress_bar(train_loader, desc=desc, leave=False, position=1)
    for step, batch in enumerate(iterator):
        loss, _, labels = run_batch(
            model,
            batch,
            device,
            diffusion_loss_weight=args.diffusion_loss_weight,
            return_aux=True,
        )
        batch_size = labels.size(0)
        total_loss += loss.detach().cpu().item() * batch_size
        total_count += batch_size

        if args.gradient_accumulation_step > 1:
            loss = loss / args.gradient_accumulation_step
        loss.backward()

        if (step + 1) % args.gradient_accumulation_step == 0:
            if args.max_grad_norm is not None and args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    if len(train_loader) % args.gradient_accumulation_step != 0:
        if args.max_grad_norm is not None and args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return total_loss / max(1, total_count)
