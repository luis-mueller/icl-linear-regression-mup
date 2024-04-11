import os
import mup
import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from mup import MuReadout, set_base_shapes, MuAdamW
from loguru import logger
from contextlib import nullcontext


def rand(size, val_range):
    low, high = val_range
    return (low - high) * torch.rand(*size) + high


def sample(batch_size, num_samples, param_range, sample_range):
    a = rand((batch_size, 1), param_range)
    b = rand((batch_size, 1), param_range)
    x = rand((batch_size, num_samples), sample_range)
    out = torch.empty((batch_size, 2 * num_samples))
    out[:, ::2] = x
    out[:, 1::2] = a * x + b
    return out


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, bias):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.c_attn = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.c_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = num_heads
        self.n_embd = embed_dim
        self.dropout = dropout

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0,
            is_causal=True,
            scale=1 / q.size(-1),
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, bias):
        super().__init__()
        self.attention = CausalSelfAttention(embed_dim, num_heads, dropout, bias)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim, bias=bias),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim, bias=bias),
        )
        self.ln_1 = nn.LayerNorm(embed_dim, bias=bias)
        self.ln_2 = nn.LayerNorm(embed_dim, bias=bias)

    def forward(self, x):
        x = x + self.attention(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x


class Transformer(nn.Sequential):
    def __init__(
        self,
        num_layers,
        block_size,
        embed_dim,
        num_heads,
        dropout,
        use_mup=True,
        bias=False,
    ):
        super().__init__()
        self.token_encoder = nn.Linear(1, embed_dim, bias=bias)
        self.pos_encoder = nn.Embedding(block_size, embed_dim)
        self.layers = nn.ModuleList(
            [
                TransformerLayer(embed_dim, num_heads, dropout, bias)
                for _ in range(num_layers)
            ]
        )
        if use_mup:
            self.head = MuReadout(embed_dim, 1, bias=bias)
        else:
            self.head = nn.Linear(embed_dim, 1, bias=bias)
        self.block_size = block_size

    def init_parameters(self, use_mup=True):
        if use_mup:
            self.apply(self._init_mup_weights)
        else:
            self.apply(self._init_weights)

    def forward(self, tokens):
        pos = torch.arange(0, self.block_size, dtype=torch.long, device=tokens.device)
        x = self.token_encoder(tokens.unsqueeze(-1)) + self.pos_encoder(pos)
        for layer in self.layers:
            x = layer(x)
        return self.head(x[:, ::2]).squeeze()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _init_mup_weights(self, module):
        if isinstance(module, nn.Linear):
            mup.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            mup.init.normal_(module.weight, mean=0.0, std=0.02)


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg):
    torch.manual_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Accelerator 🚀: {device}")

    dtype = (
        "bfloat16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "float16"
    )
    logger.info(f"Data type: {dtype}")

    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]
    ctx = (
        nullcontext()
        if device == "cpu"
        else torch.amp.autocast(device_type=device, dtype=ptdtype)
    )

    transformer_block_size = 2 * cfg.block_size - 1

    if cfg.use_mup:
        logger.info("Using mu parameterization")
        base_model = Transformer(
            cfg.num_layers,
            transformer_block_size,
            cfg.base_embed_dim,
            cfg.base_num_heads,
            cfg.dropout,
        )
        delta_model = Transformer(
            cfg.num_layers,
            transformer_block_size,
            cfg.delta_embed_dim,
            cfg.delta_num_heads,
            cfg.dropout,
        )
    else:
        logger.info("Using standard parameterization")

    model = Transformer(
        cfg.num_layers,
        transformer_block_size,
        cfg.embed_dim,
        cfg.num_heads,
        cfg.dropout,
        use_mup=cfg.use_mup,
    )

    if cfg.use_mup:
        set_base_shapes(model, base_model, delta=delta_model)

    model.init_parameters(cfg.use_mup)
    model = model.to(device)

    if cfg.use_mup:
        optimizer = MuAdamW(
            model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )

    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

    warmup_iters = int(0.025 * cfg.num_steps)

    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: min(1.0, step / warmup_iters)
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.num_steps - warmup_iters
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        [warmup_scheduler, cosine_scheduler],
        [warmup_iters],
    )
    logger.info(f"Cosine schedule with {warmup_iters} warm-up iters.")

    def get_loss():
        samples = sample(
            cfg.batch_size, cfg.block_size, cfg.param_range, cfg.sample_range
        ).to(device)
        tokens, targets = samples[:, :-1], samples[:, 1::2]
        with ctx:
            preds = model(tokens)
        return F.l1_loss(preds, targets)

    logger.info("Starting training 🍿")
    best_loss = None
    for iter in range(cfg.num_steps):
        train_loss = get_loss()
        scaler.scale(train_loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        if best_loss is None or train_loss < best_loss:
            best_loss = float(train_loss.detach().item())

        if (iter + 1) % cfg.log_every == 0:
            lr = optimizer.param_groups[0]["lr"]
            logger.info(f"Steps: {iter} | LR: {lr:.7f} | Best loss: {best_loss:.4f}")
    logger.info("Training complete ✨")

    results_file_exists = os.path.exists(cfg.results_file)
    results = pd.DataFrame(
        [
            dict(
                use_mup=cfg.use_mup,
                learning_rate=cfg.learning_rate,
                embed_dim=cfg.embed_dim,
                depth=cfg.num_layers,
                num_steps=cfg.num_steps,
                loss=best_loss,
            )
        ]
    )
    results.to_csv(
        cfg.results_file,
        index=False,
        header=not results_file_exists,
        mode="a" if results_file_exists else "w",
    )


if __name__ == "__main__":
    main()
