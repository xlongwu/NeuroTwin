"""Artificial single-harmonic counterexample, not a real-data forecast score.

Set q_spectral=U, with U/V the cosine/sine coordinates initialized from BOLD.
Under a fixed gate and zero forcing, its derivative must use both coordinates.
This checks the isolated readout; the full backbone/forcing can compensate.
"""
import json
from pathlib import Path
import torch
from models.tfm_dynamics import SpectralOscillator

theta = torch.arange(512, dtype=torch.float32) * (2 * torch.pi / 512)
u, v = theta.cos().view(-1, 1, 1), theta.sin().view(-1, 1, 1)
osc = SpectralOscillator(8, torch.tensor([2 * torch.pi * 4 / 64]))
gamma = torch.nn.functional.softplus(osc.damping_raw).detach()
ideal = (-gamma * u - osc.omega * v).squeeze(-1)
with torch.no_grad():
    osc.gate.weight.zero_()
    osc.gate.bias.zero_()
    osc.drive.weight.zero_()
    osc.drive.bias.zero_()
    # Closed-form best U-only scalar with gate=0.5, not numerical training.
    coefficient = (u.squeeze(-1) * ideal).sum() / u.square().sum()
    osc.readout.weight.fill_(2 * coefficient)
    actual = osc.rate(torch.zeros(512, 1, 8), u)
    both_coordinates = (-gamma * u - osc.omega * v).squeeze(-1)
    assert torch.allclose(both_coordinates, ideal)
    assert not torch.allclose(actual, ideal)
record = {'scope': 'artificial phase sweep of a damped single sinusoid; isolated fixed-gate readout only',
          'not_real_data_prediction': True,
          'omega_radians_per_TR': float(osc.omega[0]), 'damping_per_TR': float(gamma[0]),
          'optimal_U_coefficient_after_gate': float(coefficient),
          'U_only_derivative_MSE': float((actual - ideal).square().mean()),
          'U_V_derivative_MSE': float((both_coordinates - ideal).square().mean()),
          'unexplained_derivative_energy_fraction': float((actual - ideal).square().mean() / ideal.square().mean()),
          'boundary': 'Full model has latent drive/gate/forcing inputs; this establishes a readout limitation, not its isolated causal contribution to E2 failure.'}
out = Path(__file__).resolve().parents[1] / '读出语义检查.json'
out.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(record, ensure_ascii=False, indent=2))
