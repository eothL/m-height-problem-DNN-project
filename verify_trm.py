import torch
from trm_model import TRMWithParams
from adversarial_loss import AdversarialViolationLoss
import time

def test_components():
    print("--- Testing TRM and Adversarial Loss ---")
    
    # Virtual params
    B = 4
    n, k, m = 10, 4, 3
    
    # Mock data
    P = torch.randn(B, k, n-k)
    # Pad to (6, 6) as expected by defaults
    P_padded = torch.zeros(B, 6, 6)
    P_padded[:, :k, :(n-k)] = P
    
    params = torch.tensor([[n, k, m]] * B)
    h_true = torch.tensor([1.5] * B) # Dummy targets
    
    # 1. Test TRM
    print("\n[Test] TRM Initialization and Forward Pass")
    try:
        model = TRMWithParams(hidden_dim=32, num_steps=3)
        y_pred = model(P_padded, params)
        print(f"TRM Output shape: {y_pred.shape}") # Should be (B, Steps, 1)
        assert y_pred.shape == (B, 3, 1)
        print("Success: TRM Forward Pass")
    except Exception as e:
        print(f"Failed: {e}")
        return

    # 2. Test Adversarial Loss
    print("\n[Test] Adversarial Violation Loss")
    try:
        # P_padded needs to be actual random normal for the check to "work" smoothly, 
        # or we just check it runs without error.
        criterion = AdversarialViolationLoss(lambda_violation=1.0, num_samples=5, adv_steps=2)
        
        # Test with TRM output
        loss, mse, viol = criterion(y_pred, h_true, P_padded=P_padded, params=params)
        print(f"Loss: {loss.item():.4f} (MSE: {mse:.4f}, Violation: {viol:.4f})")
        
        # Backward check
        loss.backward()
        print("Success: Backward Pass")
    except Exception as e:
        print(f"Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_components()
