
import torch

def count_parameters(model, trainable_only=True):
    params = (p for p in model.parameters() if (not trainable_only or p.requires_grad))
    return sum(p.numel() for p in params)

def load_and_describe(model_ctor, checkpoint_path, device="cpu", show_structure=True):
    """
    model_ctor: callable returning an *uninitialized* model instance with the same
                hyperparameters that were used when training the checkpoint.
    checkpoint_path: path to the .pth file you want to inspect.
    """
    model = model_ctor().to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    if show_structure:
        print(model)
    total = count_parameters(model, trainable_only=False)
    trainable = count_parameters(model, trainable_only=True)
    print(f"\nTrainable parameters: {trainable:,}")
    print(f"Total parameters:     {total:,}")
    return model

# --- Example usages -------------------------------------------------
# 1) ResNet2D checkpoint
from Lin_Theo_m_height_problem import ResNet2DWithParams  # adjust import to where the class lives

def build_resnet():
    return ResNet2DWithParams(k_max=6, nk_max=6, n_params=3, base_ch=32, num_blocks=3)

load_and_describe(build_resnet, "best_resnet2d_no_norm.pth")

# 2) Sparse MoE checkpoint
from Lin_Theo_m_height_problem import MoE_ResNet_SparseGate  # adjust import as needed

def build_sparse_moe():
    return MoE_ResNet_SparseGate(
        num_experts=6, k_max=6, nk_max=6, n_params=3,
        base_ch=64, num_blocks=5, top_k=2, gating_hidden_dim=16
    )

load_and_describe(build_sparse_moe, "best_moe_resnet_65_5_sparsegate_e6_k2_finetuned.pth")