"""
对抗判别器（Adversarial Discriminator）
区分天然蛋白质与生成蛋白质。

输入：ProteinMPNN 编码器输出的每残基嵌入 h_V（聚合后）
输出：标量分数（logit），越高表示越"天然"

作为 GRPO 奖励函数的一个分量，提供对抗性监督信号。
兼容 PyTorch >= 1.12
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProteinDiscriminator(nn.Module):
    """
    图级判别器。
    使用注意力池化将每残基嵌入聚合为蛋白质级表示，
    再通过分类头输出真/假分数。

    训练目标：
      - 天然蛋白（来自 PDB）→ 标签 1
      - 生成蛋白（来自策略网络）→ 标签 0
    """

    def __init__(self, in_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        """
        Args:
            in_dim:     输入嵌入维度（与 ProteinMPNN hidden_dim 一致）
            hidden_dim: 判别器隐藏层维度
            dropout:    Dropout 概率
        """
        super().__init__()

        # 注意力池化：学习哪些残基对判别最重要
        # 输出 [B, N, 1] 的注意力权重
        self.attn = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # 分类头：蛋白质级嵌入 -> 真/假 logit
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)   # 输出单个 logit
        )

    def forward(self, h_V: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_V:  [B, N, D]  每残基嵌入（来自 MPNN 编码器）
            mask: [B, N]     二值掩码

        Returns:
            score: [B]  判别器 logit（未经 sigmoid，正值倾向天然）
        """
        # 数值稳定性检查：输入是否包含 NaN/Inf
        if torch.isnan(h_V).any() or torch.isinf(h_V).any():
            # 直接返回有限零值，避免 NaN * 0 仍为 NaN
            return torch.zeros(h_V.shape[0], device=h_V.device, dtype=h_V.dtype)
        
        # 计算注意力权重
        attn_logits = self.attn(h_V).squeeze(-1)              # [B, N]
        
        # 检查注意力 logits
        if torch.isnan(attn_logits).any() or torch.isinf(attn_logits).any():
            return torch.zeros(h_V.shape[0], device=h_V.device, dtype=h_V.dtype)
        
        # padding 位置设为极小值，softmax 后权重接近 0
        # 使用 -1e4 而非 -1e9，避免 fp16 溢出
        attn_logits = attn_logits.masked_fill(mask == 0, -1e4)
        
        # 数值稳定的 softmax：先减去最大值
        attn_logits_max = attn_logits.max(dim=-1, keepdim=True)[0]
        attn_logits_stable = attn_logits - attn_logits_max
        attn_weights = torch.softmax(attn_logits_stable, dim=-1)     # [B, N]
        
        # 检查 softmax 结果
        if torch.isnan(attn_weights).any() or torch.isinf(attn_weights).any():
            return torch.zeros(h_V.shape[0], device=h_V.device, dtype=h_V.dtype)

        # 注意力加权池化：得到蛋白质级表示
        h_graph = (attn_weights.unsqueeze(-1) * h_V).sum(dim=1)  # [B, D]
        
        # 检查池化结果
        if torch.isnan(h_graph).any() or torch.isinf(h_graph).any():
            return torch.zeros(h_V.shape[0], device=h_V.device, dtype=h_V.dtype)

        # 分类
        score = self.classifier(h_graph).squeeze(-1)          # [B]
        
        # 最终检查
        if torch.isnan(score).any() or torch.isinf(score).any():
            return torch.zeros(h_V.shape[0], device=h_V.device, dtype=h_V.dtype)
            
        return score

    def loss(self, real_h_V: torch.Tensor, real_mask: torch.Tensor,
             fake_h_V: torch.Tensor, fake_mask: torch.Tensor) -> torch.Tensor:
        """
        判别器训练损失（二元交叉熵）。
        天然蛋白标签 = 1，生成蛋白标签 = 0。

        Args:
            real_h_V:   [B_real, N, D]  天然蛋白的残基嵌入
            real_mask:  [B_real, N]     天然蛋白掩码
            fake_h_V:   [B_fake, N, D]  生成蛋白的残基嵌入（detach 后传入）
            fake_mask:  [B_fake, N]     生成蛋白掩码

        Returns:
            loss: 标量，判别器损失
        """
        real_score = self.forward(real_h_V, real_mask)        # [B_real]
        fake_score = self.forward(fake_h_V, fake_mask)        # [B_fake]

        # 天然蛋白目标为 1
        real_loss = F.binary_cross_entropy_with_logits(
            real_score, torch.ones_like(real_score)
        )
        # 生成蛋白目标为 0
        fake_loss = F.binary_cross_entropy_with_logits(
            fake_score, torch.zeros_like(fake_score)
        )
        # 两项取平均，保持梯度量级稳定
        return (real_loss + fake_loss) * 0.5
