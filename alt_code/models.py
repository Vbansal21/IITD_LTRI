# models.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Helper: Optional normalization layer ---
def get_norm(norm_type, dim):
    if norm_type is None:
        return nn.Identity()
    elif norm_type.lower() == 'layernorm':
        return nn.LayerNorm(dim)
    elif norm_type.lower() == 'batchnorm':
        return nn.BatchNorm1d(dim)
    else:
        raise ValueError(f'Unknown norm_type: {norm_type}')

# --- Helper: Optional activation ---
def get_activation(act_type):
    if act_type is None:
        return nn.Identity()
    act_type = act_type.lower()
    if act_type == 'relu':
        return nn.ReLU()
    if act_type == 'leakyrelu':
        return nn.LeakyReLU(0.1)
    if act_type == 'gelu':
        return nn.GELU()
    if act_type == 'swish':
        return nn.SiLU()
    if act_type == 'mish':
        return nn.Mish()
    raise ValueError(f'Unknown activation: {act_type}')

# --- Optional Dropout ---
def get_dropout(p):
    if p and p > 0:
        return nn.Dropout(p)
    return nn.Identity()

# --- Optional MLP block, for plugging anywhere ---
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim, activation='leakyrelu', norm_type=None, dropout_p=0.0):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_layers
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(get_norm(norm_type, dims[i+1]))
            layers.append(get_activation(activation))
            layers.append(get_dropout(dropout_p))
        layers.append(nn.Linear(dims[-1], output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# --- Tensor Basis Neural Network for Q' direction (with full modularity) ---
class TBNN_Q_direction(nn.Module):
    """
    Tensor Basis Neural Network: Predicts direction of Q'
    Optionally, you can add hidden_layers, activation, normalization, dropout, custom tensor basis, etc.
    """
    def __init__(self,
                 input_dim=5, # number of invariants
                 hidden_layers=[50,100,100,100,50],
                 output_dim=10, # number of tensor bases
                 activation='leakyrelu',
                 norm_type=None,
                 dropout_p=0.1,
                 custom_mlp=None, # override with your own MLP block
                 tensor_basis_fn=None, # allow swapping for new tensor basis
                 enforce_traceless=True,
                 enforce_normalized=True,
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.enforce_traceless = enforce_traceless
        self.enforce_normalized = enforce_normalized

        # Allow custom MLP (for Transformer, Mixer, etc.)
        if custom_mlp is not None:
            self.network = custom_mlp
        else:
            self.network = MLP(input_dim, hidden_layers, output_dim, activation, norm_type, dropout_p)

        # Optional tensor basis
        if tensor_basis_fn is not None:
            self.compute_tensor_bases = tensor_basis_fn

    def forward(self, s, w):
        # Compute 5 invariants (default, can be replaced via tensor_basis_fn)
        s_sq = torch.einsum('bik,bkj->bij', s, s)
        w_sq = torch.einsum('bik,bkj->bij', w, w)

        lambda_1 = torch.einsum('bii->b', s_sq)
        lambda_2 = torch.einsum('bii->b', w_sq)
        s_cubed = torch.einsum('bik,bkj->bij', s_sq, s)
        lambda_3 = torch.einsum('bii->b', s_cubed)
        w_sq_s = torch.einsum('bik,bkj->bij', w_sq, s)
        lambda_4 = torch.einsum('bii->b', w_sq_s)
        w_sq_s_sq = torch.einsum('bik,bkj->bij', w_sq, s_sq)
        lambda_5 = torch.einsum('bii->b', w_sq_s_sq)

        invariants = torch.stack([lambda_1, lambda_2, lambda_3, lambda_4, lambda_5], dim=1)

        # 10 tensor bases (use modular function)
        T = self.compute_tensor_bases(s, w, s_sq, w_sq)

        # Coefficients for each basis
        g = self.network(invariants) # (batch, 10)

        # Linear combination to form Q_hat'
        Q_hat_prime_pred = torch.einsum('bn,bnij->bij', g, T)

        # Tracelessness
        if self.enforce_traceless:
            I3 = torch.eye(3, device=Q_hat_prime_pred.device)
            trace = torch.einsum('bii->b', Q_hat_prime_pred)
            Q_hat_prime_pred -= trace[:, None, None] * I3 / 3.0

        # Self-normalize
        if self.enforce_normalized:
            norm_Q_pred = torch.sqrt(torch.sum(Q_hat_prime_pred**2, dim=(1,2), keepdim=True) + 1e-8)
            Q_hat_prime_pred = Q_hat_prime_pred / norm_Q_pred

        return Q_hat_prime_pred

    def compute_tensor_bases(self, s, w, s_sq=None, w_sq=None):
        """
        Compute the default 10 tensor bases as in Ling et al. (2016) / AIBM paper.
        Overwrite or swap via tensor_basis_fn argument in constructor.
        """
        I = torch.eye(3, device=s.device).unsqueeze(0).expand(s.shape[0], -1, -1)
        s_sq = s_sq if s_sq is not None else torch.einsum('bik,bkj->bij', s, s)
        w_sq = w_sq if w_sq is not None else torch.einsum('bik,bkj->bij', w, w)

        # T1
        T1 = s
        # T2
        T2 = torch.einsum('bik,bkj->bij', s, w) - torch.einsum('bik,bkj->bij', w, s)
        # T3
        T3 = s_sq - torch.einsum('bii->b', s_sq).view(-1,1,1)/3 * I
        # T4
        T4 = w_sq - torch.einsum('bii->b', w_sq).view(-1,1,1)/3 * I
        # T5
        T5 = torch.einsum('bik,bkj->bij', w, s_sq) - torch.einsum('bik,bkj->bij', s_sq, w)

        # --- Corrected T7/T8/T10, bugfixes included ---
        ws = torch.einsum('bik,bkj->bij', w, s)
        sw = torch.einsum('bik,bkj->bij', s, w)
        wsw2 = torch.einsum('bik,bkj->bij', ws, w_sq)
        w2sw = torch.einsum('bik,bkj->bij', w_sq, sw)
        T7 = wsw2 - w2sw

        sws2 = torch.einsum('bik,bkj->bij', sw, s_sq)
        s2ws = torch.einsum('bik,bkj->bij', s_sq, w)
        T8 = sws2 - s2ws

        w2s = torch.einsum('bik,bkj->bij', w_sq, s)
        sw2 = torch.einsum('bik,bkj->bij', s, w_sq)
        T6 = w2s + sw2 - 2./3. * torch.einsum('bii->b', sw2).view(-1, 1, 1) * I

        w2s2 = torch.einsum('bik,bkj->bij', w_sq, s_sq)
        s2w2 = torch.einsum('bik,bkj->bij', s_sq, w_sq)
        T9 = w2s2 + s2w2 - 2./3. * torch.einsum('bii->b', s2w2).view(-1,1,1) * I

        ws2 = torch.einsum('bik,bkj->bij', w, s_sq)
        s2w = torch.einsum('bik,bkj->bij', s_sq, w)
        ws2w2 = torch.einsum('bik,bkj->bij', ws2, w_sq)
        w2s2w = torch.einsum('bik,bkj->bij', w_sq, s2w)
        T10 = ws2w2 - w2s2w

        # Stack all
        return torch.stack([T1,T2,T3,T4,T5,T6,T7,T8,T9,T10], dim=1)

# --- FCNN for psi magnitude, fully modular ---
class FCNN_psi_magnitude(nn.Module):
    """
    Fully Connected Neural Network for psi magnitude.
    Easily upgradeable to deeper, wider, or to transformer/mixer, etc.
    """
    def __init__(self,
                 input_dim=2, # q, r invariants
                 hidden_layers=[50,80,50],
                 activation='leakyrelu',
                 norm_type=None,
                 dropout_p=0.1,
                 custom_mlp=None,
                 output_activation='relu',
                 enforce_positive=True,
                 ):
        super().__init__()
        self.enforce_positive = enforce_positive
        self.output_activation = output_activation

        if custom_mlp is not None:
            self.network = custom_mlp
        else:
            self.network = MLP(input_dim, hidden_layers, 1, activation, norm_type, dropout_p)

    def forward(self, q, r):
        x = torch.stack([q, r], dim=1)  # (batch, 2)
        out = self.network(x).squeeze(-1)
        if self.enforce_positive:
            if self.output_activation == 'relu':
                out = F.relu(out)
            elif self.output_activation == 'softplus':
                out = F.softplus(out)
        return out

# --- Optional: Example for plugging in a Transformer, Mixer, or ResNet block as custom_mlp ---
# (left for you to fill as needed, or request a ready-made snippet)

# --- Utility: Model factory for playing with combinations quickly ---
def make_model(model_type='tbnn', **kwargs):
    """
    Factory for model creation. Use 'tbnn', 'fcnn_psi', etc.
    Example:
        model = make_model('tbnn', hidden_layers=[128,128,128])
    """
    if model_type.lower() == 'tbnn':
        return TBNN_Q_direction(**kwargs)
    elif model_type.lower() == 'fcnn_psi':
        return FCNN_psi_magnitude(**kwargs)
    else:
        raise NotImplementedError(f'Model type {model_type} not recognized.')

# --- For BYOL-style teacher-student distillation, wrap models externally with your distillation logic ---

# --- End of file ---
