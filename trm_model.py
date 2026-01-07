import torch
import torch.nn as nn
import torch.nn.functional as F

class TRMBlock(nn.Module):
    """
    The recursive core of the TRM. It takes the current solution (y), 
    the latent state (z), and the problem encoding (x) to update z and y.
    """
    def __init__(self, hidden_dim, problem_dim):
        super().__init__()
        # Input to the block: y (1) + z (hidden_dim) + x (problem_dim)
        input_dim = 1 + hidden_dim + problem_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim + 1) # Outputs new z (hidden_dim) and delta_y (1)
        )

    def forward(self, y_curr, z_curr, x_encoded):
        # Concatenate inputs along the feature dimension
        combined = torch.cat([y_curr, z_curr, x_encoded], dim=1)
        
        out = self.net(combined)
        
        # Split output into new z and y update
        z_new = out[:, :-1]
        y_delta = out[:, -1:]
        
        y_new = y_curr + y_delta
        return y_new, z_new

class TRMWithParams(nn.Module):
    # Based on the user's specific problem: m-height prediction
    # Inputs: P (matrix), parameters (n, k, m)
    def __init__(self, k_max=6, nk_max=6, n_params=3, hidden_dim=64, num_steps=5, enforce_lower_bound=True):
        super().__init__()
        self.num_steps = num_steps
        self.enforce_lower_bound = enforce_lower_bound
        
        # 1. Problem Encoder (Embeds P and params into a fixed-size vector 'x')
        flat_dim = k_max * nk_max
        self.p_encoder = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.param_encoder = nn.Linear(n_params, hidden_dim)
        
        # Combined problem encoding size
        self.problem_dim = hidden_dim * 2
        
        # 2. Initial State Generator
        # Predicts initial y_0 and z_0 from the problem encoding
        self.init_net = nn.Sequential(
            nn.Linear(self.problem_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim + 1) # z_0 + y_0
        )
        
        # 3. Recursive Block
        self.trm_block = TRMBlock(hidden_dim, self.problem_dim)

    def forward(self, P, params):
        batch_size = P.size(0)
        
        # Flatten P
        P_flat = P.view(batch_size, -1)
        
        # Encode inputs
        x_p = self.p_encoder(P_flat)
        x_params = self.param_encoder(params.float())
        x_encoded = torch.cat([x_p, x_params], dim=1) # The "Problem" context
        
        # Initialize state
        init_out = self.init_net(x_encoded)
        z_curr = init_out[:, :-1]
        y_curr = init_out[:, -1:]
        
        outputs = []
        
        # Recursive steps
        for _ in range(self.num_steps):
            y_curr, z_curr = self.trm_block(y_curr, z_curr, x_encoded)
            
            # Apply lower bound if requested (usually applied at the very end, 
            # but for TRM we apply it to the "current solution" at each step 
            # so the recurrence sees the valid value)
            if self.enforce_lower_bound:
                 # Applying softplus to y to ensure it's > 1.0 (since m-height >= 1)
                 # We treat the raw network output as 'z' in "1 + softplus(z)" formula
                 # But here y_curr IS the prediction.
                 # To keep it simple and consistent with standard ResNet lower_bound logic:
                 # Let's interpret y_curr as the raw logit and transform it for the output list.
                 # BUT, the recurrence needs to know the actual value? 
                 # Usually TRM works on the value itself. 
                 # Let's keep y_curr as the 'score' and transform it only for final output/loss
                 # to avoid vanishing gradients through softplus in the recurrence.
                 val = 1.0 + F.softplus(y_curr)
            else:
                 val = y_curr
            
            outputs.append(val)
            
        # Return list of outputs for deep supervision (or just the last one)
        # We return the stacked outputs: (Batch, Num_Steps, 1)
        return torch.stack(outputs, dim=1)

