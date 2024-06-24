import torch
from torch.autograd import grad
import tensorly as tl
from tensorly import tenalg
from neuralop.models import GINO

tenalg.set_backend("einsum")

def run_gino_model(gno_transform_type, gno_coord_dim, batch_size):
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Fixed variables
    in_channels = 3
    out_channels = 2
    projection_channels = 16
    lifting_channels = 16
    fno_n_modes = (8, 8, 8)
    
    # Data parameters
    n_in = 100
    n_out = 100
    latent_density = 8
    
    # Initialize model
    model = GINO(
        in_channels=in_channels,
        out_channels=out_channels,
        gno_radius=0.3,  # make this large to ensure neighborhoods fit
        projection_channels=projection_channels,
        gno_coord_dim=gno_coord_dim,
        in_gno_mlp_hidden_layers=[16, 16],
        out_gno_mlp_hidden_layers=[16, 16],
        in_gno_transform_type=gno_transform_type,
        out_gno_transform_type=gno_transform_type,
        fno_n_modes=fno_n_modes[:gno_coord_dim],
        fno_lifting_channels=lifting_channels,
        fno_projection_channels=projection_channels,
        fno_norm="ada_in",
    ).to(device)
    
    # Create latent queries
    latent_geom = torch.stack(torch.meshgrid([torch.linspace(0, 1, latent_density)] * gno_coord_dim, indexing='ij'))
    latent_geom = latent_geom.permute(*list(range(1, gno_coord_dim+1)), 0).to(device)
    
    # Create input geometry and output queries
    input_geom = torch.randn(n_in, gno_coord_dim, device=device)
    output_queries = torch.randn(n_out, gno_coord_dim, device=device)
    
    # Create data and features
    x = torch.randn(batch_size, n_in, in_channels, device=device)
    x.requires_grad_(True)
    
    ada_in = torch.randn(1, device=device)
    
    # Forward pass
    out = model(x=x,
                input_geom=input_geom,
                latent_queries=latent_geom,
                output_queries=output_queries,
                ada_in=ada_in)
    
    # Compute loss and backward pass
    loss = out.sum()
    loss.backward()
    
    # Check for unused parameters
    n_unused_params = sum(1 for param in model.parameters() if param.grad is None)
    
    print(f"Output shape: {out.shape}")
    print(f"Loss: {loss.item()}")
    print(f"Number of unused parameters: {n_unused_params}")
    
    return out, loss

# Example usage
for gno_transform_type in ["linear", "nonlinear_kernelonly", "nonlinear"]:
    for gno_coord_dim in [2, 3]:
        for batch_size in [1, 4]:
            print(f"\nRunning with parameters:")
            print(f"gno_transform_type: {gno_transform_type}")
            print(f"gno_coord_dim: {gno_coord_dim}")
            print(f"batch_size: {batch_size}")
            
            out, loss = run_gino_model(gno_transform_type, gno_coord_dim, batch_size)
            
            print("--------------------")