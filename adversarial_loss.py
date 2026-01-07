import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class AdversarialViolationLoss(nn.Module):
    def __init__(self, lambda_violation=0.5, num_samples=10, adv_steps=5, adv_lr=0.1, eps=1e-9):
        super().__init__()
        self.lambda_violation = lambda_violation
        self.num_samples = num_samples # Initial random samples
        self.adv_steps = adv_steps     # Number of gradient ascent steps
        self.adv_lr = adv_lr           # Step size for gradient ascent
        self.eps = eps
        
        # Base loss
        self.mse_loss = nn.MSELoss() # We'll compute LogMSE manually to match existing style

    def forward(self, y_pred, y_true, P_padded=None, params=None, calculate_violation=True):
        """
        y_pred: (Batch, 1) or (Batch, Steps, 1)
        y_true: (Batch, 1)
        """
        # Handle TRM output (multiple steps) - take the last step for the main loss
        # But for deep supervision, we might want to average loss across steps.
        # For now, let's just take the last step for the main "prediction" to keep interface consistent.
        if y_pred.dim() == 3:
            # If TRM, y_pred is (B, Steps, 1). 
            # We enforce violation loss on the LAST step.
            # We can also add a mean MSE over all steps for stability.
            y_pred_final = y_pred[:, -1, :] 
            
            # Deep supervision MSE:
            # Expand y_true to (B, Steps, 1)
            steps = y_pred.size(1)
            # y_true is (B,) or (B,1). Ensure (B,1,1) then expand.
            if y_true.dim() == 1:
                y_true_exp = y_true.view(-1, 1, 1) # (B, 1, 1)
            else:
                y_true_exp = y_true.view(-1, 1, 1) # Force to (B, 1, 1) assuming B is dim 0

            loss_logmse = self.log_mse_loss(y_pred.view(-1, 1), y_true_exp.expand(-1, steps, -1).reshape(-1, 1))
        else:
            y_pred_final = y_pred
            loss_logmse = self.log_mse_loss(y_pred_final, y_true)

        if not calculate_violation or P_padded is None or params is None or self.lambda_violation == 0:
            return loss_logmse, loss_logmse.item(), 0.0

        # --- Adversarial Violation Check ---
        # 1. Initial random sampling (same as before)
        # 2. Gradient ascent refinement
        
        B = P_padded.shape[0]
        device = y_pred_final.device
        total_penalty = torch.tensor(0.0, device=device)

        # Process per (n, k, m) group to batch matrix operations efficiently
        unique_params_combos, inverse_indices = torch.unique(params, dim=0, return_inverse=True)

        for i in range(unique_params_combos.size(0)):
            n, k, m = int(unique_params_combos[i, 0].item()), int(unique_params_combos[i, 1].item()), int(unique_params_combos[i, 2].item())
            
            # Extract group data
            mask = inverse_indices == i
            group_P_padded = P_padded[mask]
            group_y_pred = y_pred_final[mask]
            group_batch_size = group_P_padded.size(0)

            # Validity checks
            if m + 1 > n or k <= 0 or n - k < 0:
                continue

            # Construct G = [I | P]
            P_actual = group_P_padded[:, :k, :(n-k)]
            I = torch.eye(k, device=device).unsqueeze(0).expand(group_batch_size, -1, -1)
            G_group = torch.cat([I, P_actual], dim=2) # (B_group, k, n)

            # --- Adversarial Loop ---
            # Initialize random codewords x
            # x shape: (B_group, num_samples, k)
            x = torch.randn(group_batch_size, self.num_samples, k, device=device, requires_grad=True)
            
            # We want to maximize hm(xG). 
            # Since hm is not smooth, we focus on maximizing the ratio of specific target indices 
            # found in the initial pass, OR we just maximize the 'soft' m-height.
            
            # Optimizer for x
            # We simply do manual gradient updates since it's an inner loop
            
            for step in range(self.adv_steps):
                # Calculate codewords C = xG
                # C shape: (B_group, num_samples, n)
                C = torch.matmul(x, G_group)
                
                # Calculate magnitudes
                magnitudes = torch.abs(C)
                
                # To differentiate sort, we can't easily.
                # However, we can use the "straight-through" idea or just gradient of the values 
                # at the indices selected by the current sort.
                # In PyTorch, gather/topk is differentiable w.r.t values.
                
                # Flatten to find topk across n dimension
                magnitudes_flat = magnitudes.view(-1, n)
                
                # Get top k values (we need index 0 and index m)
                # This is differentiable w.r.t magnitudes_flat
                if m + 1 > n: break # Should not happen due to check above
                    
                top_vals, _ = torch.topk(magnitudes_flat, k=m+1, dim=1, largest=True)
                
                c_max = top_vals[:, 0]
                c_m = top_vals[:, m]
                
                # Calculate m-height for each sample
                hm_samples_flat = c_max / (c_m + self.eps)
                
                # We want to MAXIMIZE hm_samples_flat
                # Loss for ascent = -mean(hm_samples_flat)
                loss_adv = -torch.mean(hm_samples_flat)
                
                # Compute gradients
                grad_x = torch.autograd.grad(loss_adv, x, create_graph=False)[0]
                
                # Update x (Gradient Ascent)
                with torch.no_grad():
                    x = x + self.adv_lr * grad_x
                    # Optional: normalize x to prevent explosion? 
                    # hm is scale invariant, so magnitude of x doesn't matter much 
                    # but keeping it reasonable helps numerical stability.
                    x = F.normalize(x, dim=2)
                
                # Enable grad again for next step (manual update detaches)
                x.requires_grad = True

            # --- Final Check with Optimized x ---
            with torch.no_grad():
                C_final = torch.matmul(x, G_group)
                magnitudes_final = torch.abs(C_final)
                magnitudes_flat_final = magnitudes_final.view(-1, n)
                top_vals_final, _ = torch.topk(magnitudes_flat_final, k=m+1, dim=1, largest=True)
                c_max_final = top_vals_final[:, 0]
                c_m_final = top_vals_final[:, m]
                hm_samples_final_flat = c_max_final / (c_m_final + self.eps)
                
                # Reshape back to (B_group, num_samples)
                hm_samples_final = hm_samples_final_flat.view(group_batch_size, self.num_samples)
                
                # Find the max violation found for each batch item
                max_hm_found, _ = torch.max(hm_samples_final, dim=1)
                
                # Penalty: if found > pred, penalize
                # pred is single value, expand
                penalty = F.relu(max_hm_found - group_y_pred.squeeze())
                total_penalty += torch.sum(penalty)

        loss_violation = total_penalty / B
        total_loss = loss_logmse + self.lambda_violation * loss_violation
        
        return total_loss, loss_logmse.item(), loss_violation.item()

    def log_mse_loss(self, y_pred, y_true):
        y_pred_safe = torch.clamp(y_pred, min=self.eps)
        y_true_safe = torch.clamp(y_true, min=self.eps)
        log2_pred = torch.log2(y_pred_safe)
        log2_true = torch.log2(y_true_safe)
        return torch.mean((log2_true - log2_pred) ** 2)
