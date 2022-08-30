def main():
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
      "ckpt": "pretrained/fashiongan_hm.pkl",
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
      "results_dir": "results",
      "ir_se50_weights": "",
      "noise_mode": "const",
      "truncation_psi": 1.0
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
