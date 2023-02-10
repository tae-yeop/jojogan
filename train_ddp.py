import os
from copy import deepcopy
from argparse import ArgumentParser
import torch
from PIL import Image
from torch import optim
from torch.nn import functional as F
from torchvision import transforms, utils
from tqdm import tqdm
from util import *
import random

from e4e_projection import projection as e4e_projection
from model import Discriminator, Generator


def set_random_seeds(random_seed=0):

    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def style_inversion(style_path):
  style_imgs_list = os.listdir(style_path)

  for name in style_imgs_list:
    # Style Image 원본 위치
    style_path = os.path.join('style_images', name)
    name = strip_path_extension(name)
    # Crop align후 저장 위치
    style_aligned_path = os.path.join('style_images_aligned', f'{name}.png')
    # style reference code 저장 위치
    style_code_path = os.path.join('inversion_codes', f'{name}.pt')

    
    # Style Image를 얻고 이를 Crop and align
    style_aligned = align_face(style_path)
    # PIL 저장
    style_aligned.save(style_aligned_path)
    # e4e GAN Inversion한뒤 .pt로 저장함.
    latent = e4e_projection(style_aligned, style_code_path)
    
def load_style_info(style_path, device):
  style_imgs_list = os.listdir(style_path)
  targets = []
  latents = []

  transform = transforms.Compose(
      [
          transforms.Resize((1024, 1024)),
          transforms.ToTensor(),
          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
      ])

  
  for name in style_imgs_list:
    # Load targets
    name = strip_path_extension(name)
    style_aligned_path = os.path.join('style_images_aligned', f'{name}.png')
    style_aligned = Image.open(style_aligned_path).convert('RGB')

    # Load latent 
    style_code_path = os.path.join('inversion_codes', f'{name}.pt')
    latent = torch.load(style_code_path)['latent']

    targets.append(transform(style_aligned).to(device))
    latents.append(latent.to(device))

  targets = torch.stack(targets, 0)
  latents = torch.stack(latents, 0)

  return targets, latents


def load_models():
  # Model 
  device = torch.device('cuda:{}'.format(local_rank))
  map_location = {"cuda:0": "cuda:{}".format(local_rank)}
  ckpt = torch.load('models/stylegan2-ffhq-config-f.pt', map_location=lambda storage, loc: storage)

  # 원래 FFHQ에 학습된 generator
  original_generator = Generator(1024, 512, 8, 2).to(device)
  ddp_original_generator = torch.nn.parallel.DistributedDataParallel(original_generator, device_ids=[local_rank], output_device=local_rank)
  ddp_original_generator.load_state_dict(ckpt["g_ema"], strict=False)
  
  # Fine tuned되어질 Generator
  ddp_generator = deepcopy(ddp_original_generator)

  # Pretrained Discriminator
  discriminator = Discriminator(1024, 2).eval().to(device)
  ddp_discriminator = torch.nn.parallel.DistributedDataParallel(discriminator, device_ids=[local_rank], output_device=local_rank)
  discriminator.load_state_dict(ckpt["d"], strict=False)

  return generator, discriminator


def test(args, file_name):
  arg = 
  
  
  with torch.no_grad():
    original_m
    
if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('--exp', type=str)
  parser.add_argument('--seed', type=int, default=3000)
  
  # Style mixing, Latent manipulation
  parser.add_argument('--id_swap', type=int, default=7)
  parser.add_argument('--latent_dim', type=int, default=512)
  parser.add_argument('--alpha', type=float, default=1.0)
  # Imageset Path
  parser.add_argument('--style_path', type=str, default='./style_images')
  parser.add_argument('--con_path', type=str, default='./test_input')

  # Finetuning Iteration
  parser.add_argument('--iter', type=int, default=1000)


  num_epochs_default = 10000
  batch_size_default = 256 # 1024
  learning_rate_default = 0.1
  random_seed_default = 0
  model_dir_default = "saved_models"
  model_filename_default = "resnet_distributed.pth"

  # DDP arguments
  parser.add_argument("--local_rank", type=int, help="Local rank. Necessary for using the torch.distributed.launch utility.")
  parser.add_argument("--batch_size", type=int, help="Training batch size for one process.", default=batch_size_default)
  # finetuning한 곳 save할 곳
  parser.add_argument("--model_dir", type=str, help="Directory for saving models.", default=model_dir_default)
  
  parser.add_argument('--precision', type=str, default='float32') #float32, float16, bf16
  
  arg = parser.parse_args()

  local_rank = arg.local_rank
    
  # shape_predictor 있는지 체크
  # 없으면 다운
  if not os.path.exists('./models/dlibshape_predictor_68_face_landmarks.dat'):
    os.system('wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2')
    os.system('bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2')
    os.system('mv shape_predictor_68_face_landmarks.dat models/dlibshape_predictor_68_face_landmarks.dat')
    
  # encoder pretrained weight 있는지 체크
  # 없으면 다운
  if not os.path.exists('./models/e4e_ffhq_encode.pt'):
    os.systme('gdown --id 1o6ijA3PkcewZvwJJ73dJ0fxhndn0nnh7 -O /home/aiteam/tykim/JoJoGAN/models/e4e_ffhq_encode.pt')
    
  # gan pretrained weight 있는지 체크
  # 없으면 다운
  if not os.path.exists('./models/stylegan2-ffhq-config-f.pt'):
    os.systme('gdown --id 1Yr7KuD959btpmcKGAUsbAk5rPjX2MytK -O /home/aiteam/tykim/JoJoGAN/models/stylegan2-ffhq-config-f.pt')

  set_random_seeds(arg.seed)
  
  
  # DDP 설정 전에 미리 
  
  # [1] gpu 설정, Initialize the distributed learning processes
  # torch.distributed.init_process_group(backend='nccl')
  
  # Load models
  generator, discriminator = load_models()
  
  # Get style latents, targets
  targets, latents = load_style_info(style_path='./style_images', device)

  # PIL을 Tensor로 바꿀 transform
  transform = transforms.Compose(
      [
          transforms.Resize((1024, 1024)),
          transforms.ToTensor(),
          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
      ]
  )


  g_optim = optim.Adam(generator.parameters(), lr=2e-3, betas=(0, 0.99))
  id_swap = list(range(arg.id_swap, generator.n_latent))
  num_iter = arg.iter
  alpha = arg.alpha
  
  fp16_scaler = torch.cuda.amp.GradScaler(enabled=True)  
  for idx in tqdm(range(num_iter)):
    # print("Local Rank: {}, Epoch: {}, Training ...".format(local_rank, idx))
    running_loss = 0.0
    
    with torch.cuda.amp.autocast():
      mean_w = generator.get_latent(torch.randn([latents.size(0), latent_dim])).unsqueeze(1).repeat(1, generator.n_latent, 1)
      in_latent[:, id_swap] = alpha * latents[:, id_swap] + (1-alpha) * mean_w[:, id_swap]
      img = generator(in_latent, input_is_latent=True)
      
      with torch.no_grad():
        real_feat = discriminator(targets)
      fake_feat = discriminator(img)
      
      loss = sum([F.l1_loss(a, b) for a, b in zip(fake_feat, real_feat)])/len(fake_feat)
      
    # mixed precision training
    # backward + optimizer step
    fp16_scaler.scale(loss).backward()
    fp16_scaler.step(g_optim)
    fp16_scaler.update()
    
    # print statistics
    running_loss += loss.item()
    
    print(f'[Epoch {idx + 1}/{num_iter}] loss: {running_loss / latents.size(0):.3f}')
    
    
    