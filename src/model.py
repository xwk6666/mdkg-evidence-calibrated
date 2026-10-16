import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel
from torchvision import models

try:
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file as load_safetensors_file
except ImportError:  # The default BERT path still works without these optional helpers.
    hf_hub_download = None
    load_safetensors_file = None


class InteractiveAttention(nn.Module):
    """Text-guided spatial attention over visual feature maps."""

    def __init__(self, visual_dim: int = 2048, semantic_dim: int = 512, hidden_dim: int = 512):
        super().__init__()
        self.v_proj = nn.Linear(visual_dim, hidden_dim)
        self.s_proj = nn.Linear(semantic_dim, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, visual_feat_map: torch.Tensor, semantic_feat: torch.Tensor):
        if visual_feat_map.dim() != 4:
            raise ValueError(f"visual_feat_map must have shape [B, C, H, W], got {tuple(visual_feat_map.shape)}")
        if semantic_feat.dim() != 2:
            raise ValueError(f"semantic_feat must have shape [B, D], got {tuple(semantic_feat.shape)}")

        batch_size, channels, height, width = visual_feat_map.size()

        v_flat = visual_feat_map.view(batch_size, channels, -1).permute(0, 2, 1)
        key = self.v_proj(v_flat)
        value = v_flat

        query = self.s_proj(semantic_feat).unsqueeze(1)
        scores = torch.bmm(query, key.transpose(1, 2)) / (self.hidden_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)

        attended_visual_feat = torch.bmm(attn_weights, value).squeeze(1)
        attn_map = attn_weights.view(batch_size, 1, height, width)
        return attended_visual_feat, attn_map


class EntityEncoder(nn.Module):
    """Encode an entity set with shared BERT features plus trainable type embeddings."""

    def __init__(self, hidden_dim: int = 768, output_dim: int = 512, num_entity_types: int = 9):
        super().__init__()
        self.type_embedding = nn.Embedding(num_entity_types, hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        pooled_entity_text: torch.Tensor,
        entity_type_ids: torch.Tensor,
        entity_mask: torch.Tensor,
    ) -> torch.Tensor:
        type_feat = self.type_embedding(entity_type_ids)
        entity_feat = self.fusion(torch.cat([pooled_entity_text, type_feat], dim=-1))
        entity_feat = F.normalize(entity_feat, p=2, dim=-1)

        mask = entity_mask.unsqueeze(-1)
        masked_feat = entity_feat * mask
        denom = mask.sum(dim=1).clamp_min(1.0)
        pooled_feat = masked_feat.sum(dim=1) / denom
        return F.normalize(pooled_feat, p=2, dim=-1)


def load_text_encoder_from_hub(model_name: str, config):
    """Load a text encoder safely under torch<2.6.

    Recent transformers versions refuse to load legacy PyTorch `.bin` weights
    with torch<2.6. For this project we prefer safetensors-backed encoders.
    The liiinn/crisis-bert checkpoint stores DistilBERT weights with a `bert.`
    prefix, so it needs a tiny compatibility shim.
    """

    if model_name == "liiinn/crisis-bert":
        if hf_hub_download is None or load_safetensors_file is None:
            raise RuntimeError("liiinn/crisis-bert requires huggingface_hub and safetensors.")
        model = AutoModel.from_config(config)
        weights_path = hf_hub_download(model_name, "model.safetensors")
        raw_state = load_safetensors_file(weights_path)
        state = {
            key.removeprefix("bert."): value
            for key, value in raw_state.items()
            if key.startswith("bert.")
        }
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(
            f"Loaded safetensors text encoder: {model_name} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )
        return model

    try:
        return AutoModel.from_pretrained(model_name, config=config, use_safetensors=True)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load text encoder with safetensors. "
            f"Model `{model_name}` may only provide legacy pytorch_model.bin weights. "
            "Your current kg_project environment uses torch<2.6, and transformers blocks "
            "legacy torch.load for security reasons. Use `liiinn/crisis-bert` for the "
            "disaster-domain experiment, use `bert-base-uncased`, or upgrade torch to >=2.6 "
            "before using models such as `vinai/bertweet-base`."
        ) from exc


class AblationDisasterModel(nn.Module):
    def __init__(
        self,
        use_attention: bool = True,
        use_hard_negative: bool = True,
        use_entity_branch: bool = False,
        text_encoder_name: str = "bert-base-uncased",
    ):
        super().__init__()
        self.use_attention = use_attention
        self.use_hard_negative = use_hard_negative
        self.use_entity_branch = use_entity_branch
        self.text_encoder_name = text_encoder_name

        print(
            f"Model init: attention=[{self.use_attention}], "
            f"hard_negative=[{self.use_hard_negative}], entity_branch=[{self.use_entity_branch}], "
            f"text_encoder=[{self.text_encoder_name}]"
        )

        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.vision_encoder = nn.Sequential(*list(resnet.children())[:-2])
        text_config = AutoConfig.from_pretrained(self.text_encoder_name)
        text_hidden_size = int(getattr(text_config, "hidden_size", 768))
        self.text_encoder = load_text_encoder_from_hub(self.text_encoder_name, text_config)

        self.v_proj = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Linear(512, 512),
        )
        self.t_proj = nn.Sequential(
            nn.Linear(text_hidden_size, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Linear(512, 512),
        )

        if self.use_attention:
            self.interactive_attention = InteractiveAttention(
                visual_dim=2048,
                semantic_dim=512,
                hidden_dim=512,
            )
        else:
            self.interactive_attention = None

        if self.use_entity_branch:
            self.entity_encoder = EntityEncoder(hidden_dim=text_hidden_size, output_dim=512, num_entity_types=9)
        else:
            self.entity_encoder = None

        self.last_attn_map = None

    @staticmethod
    def pool_text_output(text_outputs) -> torch.Tensor:
        """Return one vector per text for BERT-like and DistilBERT-like models."""

        pooled = getattr(text_outputs, "pooler_output", None)
        if pooled is not None:
            return pooled
        if not hasattr(text_outputs, "last_hidden_state"):
            raise AttributeError("Text encoder output has neither pooler_output nor last_hidden_state.")
        return text_outputs.last_hidden_state[:, 0]

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        visual_feat_map = self.vision_encoder(images)

        if self.use_attention and self.interactive_attention is not None:
            raise RuntimeError("encode_image() cannot be used when attention requires text semantics; use forward() instead.")

        pooled = F.adaptive_avg_pool2d(visual_feat_map, (1, 1)).flatten(1)
        v_feat = self.v_proj(pooled)
        return F.normalize(v_feat, p=2, dim=-1)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_text = self.pool_text_output(text_outputs)
        t_feat = self.t_proj(pooled_text)
        return F.normalize(t_feat, p=2, dim=-1)

    def encode_entities(
        self,
        entity_input_ids: torch.Tensor,
        entity_attention_mask: torch.Tensor,
        entity_type_ids: torch.Tensor,
        entity_mask: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_entity_branch or self.entity_encoder is None:
            raise RuntimeError("Entity branch is disabled for this model.")

        batch_size, max_entities, seq_len = entity_input_ids.shape
        flat_input_ids = entity_input_ids.view(batch_size * max_entities, seq_len)
        flat_attention_mask = entity_attention_mask.view(batch_size * max_entities, seq_len)

        entity_outputs = self.text_encoder(input_ids=flat_input_ids, attention_mask=flat_attention_mask)
        pooled_entities = self.pool_text_output(entity_outputs)
        pooled_entities = pooled_entities.view(batch_size, max_entities, -1)

        entity_feat = self.entity_encoder(
            pooled_entity_text=pooled_entities,
            entity_type_ids=entity_type_ids,
            entity_mask=entity_mask,
        )
        return entity_feat

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        entity_input_ids: torch.Tensor = None,
        entity_attention_mask: torch.Tensor = None,
        entity_type_ids: torch.Tensor = None,
        entity_mask: torch.Tensor = None,
        return_attention: bool = False,
        return_entity: bool = False,
    ):
        visual_feat_map = self.vision_encoder(images)

        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_text = self.pool_text_output(text_outputs)
        t_feat = self.t_proj(pooled_text)

        if self.use_attention and self.interactive_attention is not None:
            attended_visual_feat, attn_map = self.interactive_attention(visual_feat_map, t_feat)
            self.last_attn_map = attn_map
            v_feat = self.v_proj(attended_visual_feat)
        else:
            pooled_visual = F.adaptive_avg_pool2d(visual_feat_map, (1, 1)).flatten(1)
            self.last_attn_map = None
            v_feat = self.v_proj(pooled_visual)
            attn_map = None

        v_feat = F.normalize(v_feat, p=2, dim=-1)
        t_feat = F.normalize(t_feat, p=2, dim=-1)

        entity_feat = None
        if return_entity:
            if entity_input_ids is None or entity_attention_mask is None or entity_type_ids is None or entity_mask is None:
                raise ValueError("Entity tensors must be provided when return_entity=True.")
            entity_feat = self.encode_entities(
                entity_input_ids=entity_input_ids,
                entity_attention_mask=entity_attention_mask,
                entity_type_ids=entity_type_ids,
                entity_mask=entity_mask,
            )

        if return_attention:
            if return_entity:
                return v_feat, t_feat, entity_feat, attn_map
            return v_feat, t_feat, attn_map
        if return_entity:
            return v_feat, t_feat, entity_feat
        return v_feat, t_feat
