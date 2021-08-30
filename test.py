import torch
from common_nets import Mlp
from model_nets import HDNet
from utils import generate_coords
from torchdiffeq import odeint_adjoint as odeint


def test(h_layer_dims, model_path='models/hd.pth', num_batch=1000, log_interval=5):
    # HDnet calculate the Hamiltonian dynamics network given the Hamiltonian network Hnet
    Hnet = Mlp(input_dim=32, output_dim = 1, layer_dims=h_layer_dims) 
    HDnet = HDNet(Hnet=Hnet)
    HDnet.load_state_dict(torch.load(model_path))
    qp_one = generate_coords(batch_size=1)
    qp_zero = odeint(HDnet, qp_one, torch.tensor([1.0], requires_grad=True))

test(h_layer_dims=[64, 16, 32, 8, 16, 2])