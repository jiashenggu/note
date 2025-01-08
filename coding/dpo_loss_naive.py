import torch
import torch.nn.functional as F


def dpo_loss(
    chosen_logps: torch.FloatTensor,
    rejected_logps: torch.FloatTensor,
    ref_chosen_logps: torch.FloatTensor,
    ref_rejected_logps: torch.FloatTensor,
    beta: float = 0.5,
    device: str = "cpu",
) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """
    Compute the DPO loss for a batch of policy and reference model log probabilities.

    Args:
        chosen_logps (`torch.FloatTensor`):
            Log probabilities of the model for the chosen responses. Shape: `(batch_size,)`.
        rejected_logps (`torch.FloatTensor`):
            Log probabilities of the model for the rejected responses. Shape: `(batch_size,)`.
        ref_chosen_logps (`torch.FloatTensor`):
            Log probabilities of the reference model for the chosen responses. Shape: `(batch_size,)`.
        ref_rejected_logps (`torch.FloatTensor`):
            Log probabilities of the reference model for the rejected responses. Shape: `(batch_size,)`.

    Returns:
        A tuple of three tensors: `(losses, chosen_rewards, rejected_rewards)`.
        The losses tensor contains the DPO loss for each example in the batch.
        The `chosen_rewards` and `rejected_rewards` tensors contain the rewards for the chosen and rejected
        responses, respectively.
    """

    logratios = chosen_logps - rejected_logps

    ref_logratios = ref_chosen_logps - ref_rejected_logps

    logratios = logratios.to(device)
    ref_logratios = ref_logratios.to(device)
    logits = logratios - ref_logratios

    # The beta is a temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5.
    # We ignore the reference model as beta -> 0. The label_smoothing parameter encodes our uncertainty about the
    # labels and calculates a conservative DPO loss.
    losses = -F.logsigmoid(beta * logits)

    chosen_rewards = (
        beta * (chosen_logps.to(device) - ref_chosen_logps.to(device)).detach()
    )
    rejected_rewards = (
        beta * (rejected_logps.to(device) - ref_rejected_logps.to(device)).detach()
    )

    return losses, chosen_rewards, rejected_rewards


# write a test case for the above function
def test_dpo_loss():
    """ Test the DPO loss function """
    chosen_logps = torch.tensor([0.9, 0.8, 0.7])
    rejected_logps = torch.tensor([0.1, 0.2, 0.3])
    ref_chosen_logps = torch.tensor([0.8, 0.7, 0.6])
    ref_rejected_logps = torch.tensor([0.2, 0.3, 0.4])
    losses, chosen_rewards, rejected_rewards = dpo_loss(
        chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps
    )
    assert losses.shape == (3,)
    assert chosen_rewards.shape == (3,)
    assert rejected_rewards.shape == (3,)


test_dpo_loss()
