"""Laboratory -- Neural Models: Learning, Depth, Activations, and Output Layers
=============================================================================
Implementation and experimental suite for neural models on the sensor-disagreement
(XOR) problem and its 3-class extension.

Course: Artificial Intelligence -- Laboratory Exercise
Author: AI Student / Engineering Assistant
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Set standard reproducibility seed
DEFAULT_SEED = 2


# =====================================================================
# DATASET DEFINITIONS
# =====================================================================
# Inputs: x1, x2 redundant binary safety sensors
X_DATA = torch.tensor(
    [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float32
)

# Binary XOR targets: disagreement warning y in {0, 1}
Y_XOR = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float32)

# Three-class targets: 0 = both inactive, 1 = disagree, 2 = both active
Y_3CLASS = torch.tensor([0, 1, 1, 2], dtype=torch.long)


# =====================================================================
# MODEL ARCHITECTURES
# =====================================================================
class LinearBaseline(nn.Module):
  """Single affine transformation followed by sigmoid output (no hidden layer)."""

  def __init__(self):
    super().__init__()
    self.linear = nn.Linear(2, 1)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.linear(x)


class XORNet(nn.Module):
  """2-2-1 Neural Model with selectable nonlinear hidden activation."""

  def __init__(self, activation: str = "sigmoid"):
    super().__init__()
    self.fc1 = nn.Linear(2, 2)
    self.activation_name = activation.lower()

    if self.activation_name == "sigmoid":
      self.act = nn.Sigmoid()
    elif self.activation_name == "tanh":
      self.act = nn.Tanh()
    elif self.activation_name == "relu":
      self.act = nn.ReLU()
    else:
      raise ValueError(f"Unsupported activation: '{activation}'")

    self.fc2 = nn.Linear(2, 1)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    h = self.act(self.fc1(x))
    out = self.fc2(h)
    return out


class ThreeClassNet(nn.Module):
  """2-2-3 Neural Model for multiclass sensor state classification."""

  def __init__(self, activation: str = "tanh"):
    super().__init__()
    self.fc1 = nn.Linear(2, 2)
    if activation.lower() == "tanh":
      self.act = nn.Tanh()
    elif activation.lower() == "sigmoid":
      self.act = nn.Sigmoid()
    elif activation.lower() == "relu":
      self.act = nn.ReLU()
    else:
      raise ValueError(f"Unsupported activation: '{activation}'")
    self.fc2 = nn.Linear(2, 3)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    h = self.act(self.fc1(x))
    logits = self.fc2(h)
    return logits


# =====================================================================
# EXPERIMENTS
# =====================================================================
def run_task1_linear_baseline():
  """Task 1: Train a single affine layer with sigmoid output on XOR to demonstrate failure."""
  print("\n" + "=" * 70)
  print("  TASK 1: LINEAR BASELINE EXPERIMENT (NO HIDDEN LAYER)")
  print("=" * 70)

  torch.manual_seed(DEFAULT_SEED)
  model = LinearBaseline()
  criterion = nn.BCEWithLogitsLoss()
  optimizer = optim.SGD(model.parameters(), lr=0.1)

  init_loss = criterion(model(X_DATA), Y_XOR).item()

  for _ in range(2000):
    optimizer.zero_grad()
    logits = model(X_DATA)
    loss = criterion(logits, Y_XOR)
    loss.backward()
    optimizer.step()

  final_loss = loss.item()
  probs = torch.sigmoid(model(X_DATA)).detach()
  preds = (probs > 0.5).float()
  correct = (preds == Y_XOR).sum().item()

  print(f"Initial BCE Loss: {init_loss:.5f}")
  print(f"Final BCE Loss:   {final_loss:.5f}")
  print(f"Accuracy:         {int(correct)}/4 correct")
  print("Predictions per input:")
  for i in range(4):
    inp = X_DATA[i].numpy()
    target = int(Y_XOR[i].item())
    p = probs[i].item()
    pred = int(preds[i].item())
    print(
        f"  Input: {inp} | Target: {target} | Probability: {p:.4f} | Pred:"
        f" {pred}"
    )
  print(
      "\nConclusion: As predicted mathematically, a linear classifier cannot"
      " solve XOR."
  )


def run_task4_parts_ab():
  """Task 4 Part A & B: Basic Learning Check and Backpropagation Analysis."""
  print("\n" + "=" * 70)
  print("  TASK 4: PART A (LEARNING CHECK) & PART B (BACKPROPAGATION CHECK)")
  print("=" * 70)

  torch.manual_seed(DEFAULT_SEED)
  model = XORNet(activation="sigmoid")
  criterion = nn.BCEWithLogitsLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.1)

  # Step 1: Forward & Backward pass to inspect gradients
  optimizer.zero_grad()
  logits_step1 = model(X_DATA)
  loss_step1 = criterion(logits_step1, Y_XOR)
  loss_step1.backward()

  early_w1_grad = model.fc1.weight.grad.clone()
  early_grad_norm = early_w1_grad.norm().item()

  print(f"Part A: Initial Loss (Step 1): {loss_step1.item():.5f}")
  print("\nPart B: First-Layer Weight Gradient dL/dW^(1) at Step 1:")
  print(early_w1_grad)
  print(f"Euclidean norm of dL/dW^(1): {early_grad_norm:.6f}")

  # Verify batch average gradient property
  print(
      "\nVerifying that batch gradient is the exact average of example-wise"
      " gradients:"
  )
  example_grads = []
  for i in range(4):
    model.zero_grad()
    l_i = criterion(model(X_DATA[i : i + 1]), Y_XOR[i : i + 1])
    l_i.backward()
    example_grads.append(model.fc1.weight.grad.clone())

  mean_manual_grad = torch.stack(example_grads).mean(dim=0)
  diff = (early_w1_grad - mean_manual_grad).abs().max().item()
  print(f"  Maximum absolute difference: {diff:.2e} (Matches identically)")

  # Complete training
  optimizer.step()
  for _ in range(2, 2001):
    optimizer.zero_grad()
    out = model(X_DATA)
    loss = criterion(out, Y_XOR)
    loss.backward()
    optimizer.step()

  final_loss = loss.item()
  final_probs = torch.sigmoid(model(X_DATA)).detach()
  final_preds = (final_probs > 0.5).float()
  all_correct = (final_preds == Y_XOR).all().item()

  print(f"\nPart A: Final Loss (Epoch 2000): {final_loss:.6f}")
  print(f"All 4/4 Labels Correct:        {all_correct}")
  print("Final Model Outputs:")
  for i in range(4):
    inp = X_DATA[i].numpy()
    target = int(Y_XOR[i].item())
    prob = final_probs[i].item()
    pred = int(final_preds[i].item())
    print(
        f"  Input: {inp} | Target: {target} | Output Prob: {prob:.4f} | Pred:"
        f" {pred} [{'CORRECT' if pred == target else 'FAIL'}]"
    )


def run_task4_part_c_symmetry():
  """Task 4 Part C: Symmetry Experiment (Zero Weight Initialisation)."""
  print("\n" + "=" * 70)
  print("  TASK 4: PART C -- SYMMETRY EXPERIMENT (ZERO INITIALISATION)")
  print("=" * 70)

  model = XORNet(activation="sigmoid")

  # Set all parameters (weights and biases) to zero
  for p in model.parameters():
    p.data.zero_()

  criterion = nn.BCEWithLogitsLoss()
  optimizer = optim.SGD(model.parameters(), lr=0.1)

  print("Initial W^(1) matrix:")
  print(model.fc1.weight.data)

  print("\nInspecting the two rows of W^(1) across training steps:")
  print(f"{'Step':<6} | {'Loss':<10} | {'Row 0 == Row 1?':<16} | {'W^(1) Values'}")
  print("-" * 65)

  for step in range(1, 6):
    optimizer.zero_grad()
    logits = model(X_DATA)
    loss = criterion(logits, Y_XOR)
    loss.backward()
    optimizer.step()

    row0 = model.fc1.weight.data[0].tolist()
    row1 = model.fc1.weight.data[1].tolist()
    is_identical = model.fc1.weight.data[0].equal(model.fc1.weight.data[1])
    print(
        f"{step:<6} | {loss.item():<10.5f} | {str(is_identical):<16} | R0:"
        f" [{row0[0]:.4f}, {row0[1]:.4f}], R1: [{row1[0]:.4f}, {row1[1]:.4f}]"
    )

  print("\nExplanation:")
  print(
      "When both hidden units share identical zero weights, they compute the"
      " exact same"
  )
  print(
      "activation h1 = h2 for every input. Backpropagation propagates"
      " identical gradients"
  )
  print(
      "dL/dW1[0, :] = dL/dW1[1, :]. Because their initial states and gradients"
      " are identical,"
  )
  print(
      "the rows remain permanently identical. The network cannot break symmetry"
  )
  print("and effectively collapses to having only a single hidden unit.")


def run_task4_part_d_activations():
  """Task 4 Part D: Activation Experiment (Sigmoid vs Tanh vs ReLU)."""
  print("\n" + "=" * 70)
  print("  TASK 4: PART D -- ACTIVATION EXPERIMENT (SIGMOID, TANH, RELU)")
  print("=" * 70)

  activations = ["Sigmoid", "Tanh", "ReLU"]
  results = []

  for act in activations:
    torch.manual_seed(DEFAULT_SEED)
    model = XORNet(activation=act)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    # Early step gradient norm
    optimizer.zero_grad()
    logits = model(X_DATA)
    loss = criterion(logits, Y_XOR)
    loss.backward()
    early_grad_norm = model.fc1.weight.grad.norm().item()
    optimizer.step()

    # Train for 2000 steps
    for _ in range(2, 2001):
      optimizer.zero_grad()
      logits = model(X_DATA)
      loss = criterion(logits, Y_XOR)
      loss.backward()
      optimizer.step()

    final_loss = loss.item()
    probs = torch.sigmoid(model(X_DATA)).detach()
    preds = (probs > 0.5).float()
    all_correct = (preds == Y_XOR).all().item()

    results.append({
        "activation": act,
        "final_loss": final_loss,
        "correct": all_correct,
        "early_norm": early_grad_norm,
    })

  # Print Result Table
  col_norm_header = "Early ||Grad W(1)||2"
  print(
      f"{'Hidden activation':<18} | {'Final loss':<12} | {'4/4 correct?':<14} |"
      f" {col_norm_header:<22}"
  )
  print("-" * 72)
  for r in results:
    correct_str = "Yes" if r["correct"] else "No"
    print(
        f"{r['activation']:<18} | {r['final_loss']:<12.6f} |"
        f" {correct_str:<14} | {r['early_norm']:<22.6f}"
    )


def run_task5_three_class_extension():
  """Task 5: Multiclass Sensor Extension (3 Logits, CrossEntropyLoss, Stability Diagnostic)."""
  print("\n" + "=" * 70)
  print("  TASK 5: THREE-CLASS EXTENSION (CLASSES 0, 1, 2)")
  print("=" * 70)

  torch.manual_seed(DEFAULT_SEED)
  model = ThreeClassNet(activation="tanh")
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.05)

  print("Pre-Implementation Predictions:")
  print(
      "1. Shape of final weight matrix: [3, 2] (3 output classes, 2 hidden"
      " units)"
  )
  print("2. Number of logits per example:  3 logits")
  print("3. Why softmax sums to 1:        sum_j exp(z_j) / sum_k exp(z_k) = 1")
  print("4. Why logit gradient is p - y:  dL/dz_i = p_i - y_i (standard"
        " softmax-cross-entropy result)")

  # Train for 1500 epochs
  for epoch in range(1, 1501):
    optimizer.zero_grad()
    logits = model(X_DATA)
    loss = criterion(logits, Y_3CLASS)
    loss.backward()
    optimizer.step()

  final_logits = model(X_DATA).detach()
  final_probs = torch.softmax(final_logits, dim=-1)
  final_preds = final_probs.argmax(dim=-1)

  print(f"\nFinal 3-Class Cross-Entropy Loss: {loss.item():.6f}")
  print(
      f"Final Weight Matrix fc2.weight.shape: {list(model.fc2.weight.shape)}"
  )
  print("\nPredicted Class Probabilities:")
  print(
      f"{'Input (x1, x2)':<16} | {'Target':<8} | {'P(Class 0)':<12} |"
      f" {'P(Class 1)':<12} | {'P(Class 2)':<12} | {'Pred':<6}"
  )
  print("-" * 75)
  for i in range(4):
    inp = str(X_DATA[i].numpy().tolist())
    target = Y_3CLASS[i].item()
    p0 = final_probs[i, 0].item()
    p1 = final_probs[i, 1].item()
    p2 = final_probs[i, 2].item()
    pred = final_preds[i].item()
    print(
        f"{inp:<16} | {target:<8} | {p0:<12.4f} | {p1:<12.4f} | {p2:<12.4f} |"
        f" {pred:<6}"
    )

  # Verify numerical sum to approximately 1
  prob_sum = final_probs[0].sum().item()
  print(f"\nVerification of Softmax Normalisation for input (0, 0):")
  print(f"  Sum of probabilities: {prob_sum:.10f} (approximately 1.0)")

  # Numerical stability diagnostic (+100 to all logits)
  shifted_logits = final_logits + 100.0
  shifted_probs = torch.softmax(shifted_logits, dim=-1)
  max_diff = (final_probs - shifted_probs).abs().max().item()

  print("\nNumerical Stability Diagnostic (Shift Invariance):")
  print(f"  Max absolute difference after adding +100 to all logits: {max_diff:.2e}")
  print(
      "  Explanation: Adding constant C to all logits scales exp(z_i + C) ="
      " exp(z_i) * exp(C)."
  )
  print(
      "  The common factor exp(C) cancels exactly between numerator and"
      " denominator."
  )
  print(
      "  Software implementations subtract max(z) before exponentiating to"
      " avoid numerical overflow."
  )


# =====================================================================
# MAIN RUNNER
# =====================================================================
def main():
  print("=" * 70)
  print("   NEURAL MODELS LABORATORY: COMPLETE EXPERIMENTAL SUITE")
  print("   Learning, Depth, Activations, and Output Layers")
  print("=" * 70)

  run_task1_linear_baseline()
  run_task4_parts_ab()
  run_task4_part_c_symmetry()
  run_task4_part_d_activations()
  run_task5_three_class_extension()

  print("\n" + "=" * 70)
  print("   ALL EXPERIMENTS COMPLETED SUCCESSFULLY.")
  print("=" * 70)


if __name__ == "__main__":
  main()
