from gemma_scratch.model import Gemma3Model
from gemma_scratch.config import GEMMA3_CONFIG_CUSTOM, create_gemma_config
import torch


def count_parameters(module):
    """Counts the total number of trainable parameters in a given PyTorch module."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def print_parameter_breakdown(model):
    """
    Analyzes the model's parameters and prints a detailed breakdown by component.
    """
    params = {
        "Embedding Layer": 0,
        "Output Layer": 0,
        "Layer Normalization (Total)": 0,
        "Multi-head Attention (Total)": 0,
        "Multi-layer Perceptron (Total)": 0,
    }

    # 1. Embedding Layer
    params["Embedding Layer"] = count_parameters(model.tok_emb)

    # 2. Output Layer (check for weight tying)
    is_weight_tied = model.tok_emb.weight.data.is_set_to(model.out_head.weight.data)
    if not is_weight_tied:
        params["Output Layer"] = count_parameters(model.out_head)

    # 3. Final Layer Normalization (outside the blocks)
    params["Layer Normalization (Total)"] += count_parameters(model.final_norm)

    # 4. Parameters within each Transformer Block
    if model.blocks:
        first_block = model.blocks[0]
        num_blocks = len(model.blocks)

        # Parameters for GroupedQueryAttention (linear layers only)
        single_att_linear_params = (
            count_parameters(first_block.att.W_query)
            + count_parameters(first_block.att.W_key)
            + count_parameters(first_block.att.W_value)
            + count_parameters(first_block.att.out_proj)
        )

        # Parameters for q_norm and k_norm within a single Attention module
        single_att_q_norm_params = count_parameters(first_block.att.q_norm)
        single_att_k_norm_params = count_parameters(first_block.att.k_norm)

        # Parameters for FeedForward module in a single block
        single_mlp_params = count_parameters(first_block.ff)

        # Parameters for the 4 RMSNorms directly within a single TransformerBlock
        single_block_direct_ln_params = (
            count_parameters(first_block.input_layernorm)
            + count_parameters(first_block.post_attention_layernorm)
            + count_parameters(first_block.pre_feedforward_layernorm)
            + count_parameters(first_block.post_feedforward_layernorm)
        )

        # Accumulate totals across all blocks
        params["Multi-head Attention (Total)"] += single_att_linear_params * num_blocks
        params["Multi-layer Perceptron (Total)"] += single_mlp_params * num_blocks

        params["Layer Normalization (Total)"] += (
            single_block_direct_ln_params * num_blocks
        )  # 4 RMSNorms per block
        params["Layer Normalization (Total)"] += (
            single_att_q_norm_params + single_att_k_norm_params
        ) * num_blocks  # 2 RMSNorms per attention module in each block

        # Calculate single block total for display
        params["Transformer Block (Single)"] = (
            single_att_linear_params
            + single_att_q_norm_params
            + single_att_k_norm_params
            + single_mlp_params
            + single_block_direct_ln_params
        )

    # --- Print the Results ---
    print("--- Model Parameter Breakdown ---")
    total_calculated = 0

    print_order = [
        "Embedding Layer",
        "Output Layer",
        "Multi-head Attention (Total)",
        "Multi-layer Perceptron (Total)",
        "Layer Normalization (Total)",
        "Transformer Block (Single)",
    ]

    for name in print_order:
        if name in params:
            count = params[name]
            if "Single" not in name:
                total_calculated += count

            count_str = f"{count:12,}"
            if name == "Output Layer" and is_weight_tied:
                count_str = " (Tied with Embedding)"

            print(f"{name:<32}: {count_str}")

    print("-" * 46)

    total_model_params = count_parameters(model)
    print(f"{'Sum of Components':<32}: {total_calculated:12,}")
    print(f"{'Full Transformer Model (Actual)':<32}: {total_model_params:12,}")

    if total_model_params != total_calculated:
        print("\nWarning: Discrepancy detected!\n")
        print(
            f"The calculated sum ({total_calculated:,}) does not match the model's total ({total_model_params:,})."
        )
        print(
            "This might be due to minor discrepancies in how sub-modules are reported or floating-point arithmetic."
        )
    else:
        print("\nVerification successful: Sum of components matches the total.")


if __name__ == "__main__":
    gemma_config = create_gemma_config(
        # Architecture Dimensions
        vocab_size=50_257,
        context_length=32_768 // 8,
        emb_dim=640 // 2,
        n_heads=4,
        hidden_dim=2048 // 2,
        head_dim=256 // 2,
        n_kv_groups=1,
        # Layer & Attention Configuration
        n_layers=9,
        sliding_layers_per_block=2,
        full_layers_per_block=1,
        sliding_window=512,
        # RoPE & Normalization
        rope_local_base=10_000.0,
        rope_base=1_000_000.0,
        qk_norm=True,
        query_pre_attn_scalar=256,
        # System
        dtype=torch.bfloat16,
    )

    gemma_model = Gemma3Model(gemma_config)
    gemma_model.out_head.weight = (
        gemma_model.tok_emb.weight
    )  # Ensure weight tying for accurate count
    print_parameter_breakdown(gemma_model)
