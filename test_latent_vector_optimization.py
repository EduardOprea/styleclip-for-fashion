import click
import sys
import os 

sys.path.append(os.path.join(os.getcwd(), "models/stylegan2adapytorch"))

@click.command()
@click.pass_context
@click.option('--checkpoint', help='Path to network pikle', required=True)
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--description', help='Description of generated image', required=True)
def main(ctx: click.Context,
    checkpoint: str,
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    description: str):

  experiment_type = 'edit' #@param ['edit', 'free_generation']

  description = 'A shirt with floral print' #@param {type:"string"}

  latent_path = None #@param {type:"string"}

  optimization_steps = 1 #@param {type:"number"}

  l2_lambda = 0.008 #@param {type:"number"}

  id_lambda = 0.005 #@param {type:"number"}

  stylespace = False #@param {type:"boolean"}

  create_video = True #@param {type:"boolean"}

  use_seed = True #@param {type:"boolean"}

  seed = 1 #@param {type: "number"}


  #@title Additional Arguments
  args = {
      "description": description,
      "ckpt": checkpoint,
      "stylegan_size": 512,
      "lr_rampup": 0.05,
      "lr": 0.1,
      "step": optimization_steps,
      "mode": experiment_type,
      "l2_lambda": l2_lambda,
      "id_lambda": id_lambda,
      'work_in_stylespace': stylespace,
      "latent_path": latent_path,
      "truncation": 0.7,
      "save_intermediate_image_every": 1 if create_video else 20,
      "results_dir": outdir,
      "ir_se50_weights": "",
      "noise_mode": noise_mode,
      "truncation_psi": truncation_psi
  }


  if use_seed:
    import torch
    torch.manual_seed(seed)
  from optimization.run_optimization_ada_pytorch import main
  from argparse import Namespace
  result = main(Namespace(**args))

  #@title Visualize Result
  from torchvision.utils import make_grid
  from torchvision.transforms import ToPILImage
  result_image = ToPILImage()(make_grid(result.detach().cpu(), normalize=True, scale_each=True, range=(-1, 1), padding=0))
  h, w = result_image.size
  result_image.resize((h // 2, w // 2))
if __name__ == "__main__":
  main()
