from gemma_scratch.model import Gemma3Model
from gemma_scratch.config import GEMMA3_CONFIG_CUSTOM


def count_parameters(module):
    """Counts the total number of trainable parameters in a given PyTorch module."""
    if module is None:
        return 0
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def get_module_param_shape(module, param_name):
    """
    Safely gets the shape of a parameter within a module.
    Handles cases where the parameter might not exist or is None.
    """
    if module is None:
        return None
    param = getattr(module, param_name, None)
    if param is not None and hasattr(param, "shape"):
        return param.shape
    return None


def generate_report(model):
    """
    Generates a detailed parameter report including matrix sizes.
    """
    report_lines = []

    # --- Header ---
    report_lines.append("Parameters of each component:")
    report_lines.append(f"{'-' * 97}")
    report_lines.append(f"{'Component':<50} {'Parameters':>15} {'Matrix Size':>30}")

    # --- Helper function for formatting ---
    def add_line(name, count_or_note, matrix_size=None, separator_char="-", level=0):
        is_note = isinstance(count_or_note, str)
        count_str = f"{count_or_note:,}" if not is_note else count_or_note
        matrix_size_str = str(matrix_size) if matrix_size is not None else ""

        if level == 0:
            report_lines.append(f"{separator_char * 97}")
            report_lines.append(f"{name:<50} {count_str:>15} {matrix_size_str:>30}")
        elif level == 1:
            report_lines.append(f"  {name:<48} {count_str:>15} {matrix_size_str:>30}")

    # --- Embedding Layer ---
    embedding_params = count_parameters(model.tok_emb)
    embedding_size = get_module_param_shape(model.tok_emb, "weight")
    add_line("embedding", embedding_params, embedding_size)

    # --- Transformer Blocks ---
    total_block_params = 0
    for i, block in enumerate(model.blocks):
        # Attention
        q_proj_params = count_parameters(block.att.W_query)
        q_proj_size = get_module_param_shape(block.att.W_query, "weight")
        k_proj_params = count_parameters(block.att.W_key)
        k_proj_size = get_module_param_shape(block.att.W_key, "weight")
        v_proj_params = count_parameters(block.att.W_value)
        v_proj_size = get_module_param_shape(block.att.W_value, "weight")
        out_proj_params = count_parameters(block.att.out_proj)
        out_proj_size = get_module_param_shape(block.att.out_proj, "weight")

        attention_params = (
            q_proj_params + k_proj_params + v_proj_params + out_proj_params
        )

        # Layer Norms (Attention)
        input_layernorm_params = count_parameters(block.input_layernorm)
        input_layernorm_size = get_module_param_shape(block.input_layernorm, "scale")
        post_attention_layernorm_params = count_parameters(
            block.post_attention_layernorm
        )
        post_attention_layernorm_size = get_module_param_shape(
            block.post_attention_layernorm, "scale"
        )
        q_norm_params = count_parameters(block.att.q_norm)
        q_norm_size = get_module_param_shape(block.att.q_norm, "scale")
        k_norm_params = count_parameters(block.att.k_norm)
        k_norm_size = get_module_param_shape(block.att.k_norm, "scale")

        ln_attention_params = (
            input_layernorm_params
            + post_attention_layernorm_params
            + q_norm_params
            + k_norm_params
        )

        # MLP
        fc1_params = count_parameters(block.ff.fc1)
        fc1_size = get_module_param_shape(block.ff.fc1, "weight")
        fc2_params = count_parameters(block.ff.fc2)
        fc2_size = get_module_param_shape(block.ff.fc2, "weight")
        fc3_params = count_parameters(block.ff.fc3)
        fc3_size = get_module_param_shape(block.ff.fc3, "weight")

        mlp_params = fc1_params + fc2_params + fc3_params

        # Layer Norms (MLP)
        pre_feedforward_layernorm_params = count_parameters(
            block.pre_feedforward_layernorm
        )
        pre_feedforward_layernorm_size = get_module_param_shape(
            block.pre_feedforward_layernorm, "scale"
        )
        post_feedforward_layernorm_params = count_parameters(
            block.post_feedforward_layernorm
        )
        post_feedforward_layernorm_size = get_module_param_shape(
            block.post_feedforward_layernorm, "scale"
        )

        ln_mlp_params = (
            pre_feedforward_layernorm_params + post_feedforward_layernorm_params
        )

        block_total = (
            attention_params + ln_attention_params + mlp_params + ln_mlp_params
        )
        total_block_params += block_total

        add_line(f"transformer_block_{i}", block_total)
        report_lines.append("-" * 97)
        add_line(f"attention_q_proj_{i}", q_proj_params, q_proj_size, level=1)
        add_line(f"attention_k_proj_{i}", k_proj_params, k_proj_size, level=1)
        add_line(f"attention_v_proj_{i}", v_proj_params, v_proj_size, level=1)
        add_line(f"attention_out_proj_{i}", out_proj_params, out_proj_size, level=1)
        report_lines.append("-" * 97)
        add_line(
            f"input_layernorm_{i}",
            input_layernorm_params,
            input_layernorm_size,
            level=1,
        )
        add_line(
            f"post_attention_layernorm_{i}",
            post_attention_layernorm_params,
            post_attention_layernorm_size,
            level=1,
        )
        add_line(f"q_norm_{i}", q_norm_params, q_norm_size, level=1)
        add_line(f"k_norm_{i}", k_norm_params, k_norm_size, level=1)
        report_lines.append("-" * 97)
        add_line(f"mlp_fc1_{i}", fc1_params, fc1_size, level=1)
        add_line(f"mlp_fc2_{i}", fc2_params, fc2_size, level=1)
        add_line(f"mlp_fc3_{i}", fc3_params, fc3_size, level=1)
        report_lines.append("-" * 97)
        add_line(
            f"pre_feedforward_layernorm_{i}",
            pre_feedforward_layernorm_params,
            pre_feedforward_layernorm_size,
            level=1,
        )
        add_line(
            f"post_feedforward_layernorm_{i}",
            post_feedforward_layernorm_params,
            post_feedforward_layernorm_size,
            level=1,
        )

    # --- Final Normalization ---
    final_norm_params = count_parameters(model.final_norm)
    final_norm_size = get_module_param_shape(model.final_norm, "scale")
    add_line("final_norm", final_norm_params, final_norm_size)

    # --- Output Layer ---
    add_line("output_layer", "(Tied to embedding)", "(Tied to embedding)")

    # --- Total ---
    final_total = embedding_params + total_block_params + final_norm_params
    report_lines.append(f"{'-' * 97}")
    report_lines.append(f"{'Total':<50} {final_total:15,} {'':>30}")

    # --- Verification Footer ---
    actual_model_params = count_parameters(model)
    if final_total == actual_model_params:
        report_lines.append(
            "\nVerification successful: Calculated total matches trainable model parameters."
        )
    else:
        report_lines.append(
            f"\nVerification Warning: Discrepancy found. Calculated: {final_total:,}, Actual: {actual_model_params:,}"
        )

    return "\n".join(report_lines)


if __name__ == "__main__":
    gemma_model = Gemma3Model(GEMMA3_CONFIG_CUSTOM)
    gemma_model.out_head.weight = gemma_model.tok_emb.weight

    report_content = generate_report(gemma_model)

    with open("model_summary.txt", "w") as f:
        f.write(report_content)

    print("Successfully generated 'parameter_report.txt'")
    print("\n--- Parameter Report Content ---")
    print(report_content)
