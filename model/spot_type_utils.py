import torch


def expand_spot_type_sequence(spot_type, batch_size, token_count, device):
    """Normalize spot-type inputs to shape (B, token_count)."""
    if spot_type is None:
        return torch.zeros((batch_size, token_count), dtype=torch.long, device=device)

    spot_type = spot_type.to(device=device, dtype=torch.long)
    if spot_type.ndim == 1:
        spot_type = spot_type.unsqueeze(1)

    if spot_type.ndim != 2:
        raise ValueError(f'Expected spot_type to have 1 or 2 dims, got {spot_type.shape}.')

    if spot_type.size(1) == token_count:
        return spot_type

    if spot_type.size(1) == 1:
        return spot_type.expand(-1, token_count)

    raise ValueError(
        f'Expected {token_count} token type IDs, got shape {tuple(spot_type.shape)}.'
    )


def gather_token_bank(token_bank, token_type_ids):
    """Index a `(num_types, dim)` token bank with optional padding IDs."""
    valid_mask = token_type_ids >= 0
    safe_ids = token_type_ids.clamp_min(0)
    gathered = token_bank[safe_ids]
    return gathered * valid_mask.unsqueeze(-1).type_as(gathered)
