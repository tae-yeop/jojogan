import os
import tempfile
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import optim
from torch.nn import functional as F
from torchvision import transforms, utils
from tqdm import tqdm
from util import *

from e4e_projection_pl import projection as e4e_projection
from model import Discriminator, Generator
from e4e.models.psp_pl import pSp


from pytorch_lightning import LightningModule, Trainer, seed_everything
import pytorch_lightning as pl
from pytorch_lightning.callbacks import *

from argparse import ArgumentParser 
from argparse import Namespace


from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
  def __init__(self, tfms, style_path):
    """
    style_path : style images가 있는 폴더의 path
    """
    super().__init__()
    style_imgs_list = os.listdir(style_path)
    # con_imgs_list = listdir(con_path)

    targets = []
    latents = []

    # for name in style_imgs_list:
    #   style_img_path = os.path.join('style_images', name)
    #   # Style Image를 얻고 이를 Crop and align
    #   # 이미 align한게 있다면 그냥 불러오고 아니면 align_face를 수행한다
    #   name = strip_path_extension(name)
    #   style_aligned_path = os.path.join('style_images_aligned', f'{name}.png')

    #   if os.path.exists(style_aligned_path):
    #     style_aligned = Image.open(style_aligned_path).convert('RGB')
    #   else:
    #     # Alignment를 하고 PIL 이미지를 리턴
    #     style_aligned = align_face(style_img_path)
    #     # png로 저장한다.
    #     style_aligned.save(style_aligned_path)

    #   # w를 찾아낸 뒤 pt로 저장한다.
    #   style_code_path = os.path.join('inversion_codes', f'{name}.pt')
    #   if not os.path.exists(style_code_path):
    #     latent = e4e_projection(style_aligned, style_code_path)
    #   else:
    #     latent = torch.load(style_code_path)['latent']

    #   targets.append(tfms(style_aligned))
    #   latents.append(latent)

    # targets = torch.stack(targets, 0)
    # latents = torch.stack(latents, 0) # shape : [the num of styles, 18, 512]
    

    self.sample = {"style_targets" : targets ,
                   "style_latents": latents}
    
  def __len__(self): return 6
  def __getitem__(self, idx): return self.sample


class DataModule(pl.LightningDataModule):
  def __init__(self, style_path):
    
    super().__init__()
    self.style_path = style_path
    # self.con_path = con_path
    self.transform = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

  def prepare_data(self):
    pass
  
  def setup(self, stage=None):
    if stage == 'fit' or stage == 'None':
      self.train = CustomDataset(self.transform, self.style_path)

  def train_dataloader(self):
    return DataLoader(self.train, batch_size=1, shuffle=False) #num_workers=8, pin_memory=True)



class JJGAN(pl.LightningModule):
  def __init__(self, args):
    super().__init__()
    self.latent_dim = args.latent_dim
    self.alpha = args.alpha
    
    # 원래 FFHQ에 학습된 generator
    self.original_generator = Generator(1024, self.latent_dim, 8, 2)
    ckpt = torch.load('models/stylegan2-ffhq-config-f.pt', map_location=lambda storage, loc: storage)
    self.original_generator.load_state_dict(ckpt["g_ema"], strict=False)
    
    print('original_generator device : ', next(self.original_generator.parameters()).device)
    # Fine tuned되어질 Generator
    self.generator = deepcopy(self.original_generator)
    
    # Discriminator
    self.discriminator = Discriminator(1024, 2).eval()
    self.discriminator.load_state_dict(ckpt["d"], strict=False)

    self.psp_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

  def projection(self, img, name):
    img = self.psp_transform(img).unsqueeze(0).to(self.device)
    print('img device:', img.device)
    images, w_plus = self.psp(img, randomize_noise=False, return_latents=True)
    print('psp w_plus : ', w_plus.device)
    result_file = {}
    result_file['latent'] = w_plus[0]
    torch.save(result_file, name)
    return w_plus[0]

  def configure_optimizers(self):
    optimizer = optim.Adam(self.generator.parameters(), lr=2e-3, betas=(0, 0.99))
    return optimizer
  
  def on_train_start(self):
    
    # pSp
    ckpt = torch.load('models/e4e_ffhq_encode.pt', map_location=lambda storage, loc: storage)
    opts = ckpt['opts']
    opts['checkpoint_path'] = 'models/e4e_ffhq_encode.pt'
    opts= Namespace(**opts)
    self.psp = pSp(opts).eval().to(self.device)

    print('psp device', next(self.psp.parameters()).device)
    
    # 10000개의 w를 만들고 평균을 낸다. [1,512]
    mean_latent = self.original_generator.mean_latent(10000)

    # 몇 번째 레이어부터 어디까지 값을 바꿀지 설정
    self.id_swap = list(range(7, self.generator.n_latent))

    # Style References를 tensor targets으로 만들기 위한 pipeline
    self.transform = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    self.targets, self.latents = self.set_style_latents(args.style_path)
    
  def training_step(self, batch, batch_idx):
    # W space를 활용
    mean_w = self.generator.get_latent(torch.randn([self.latents.size(0), self.latent_dim], device=self.device)).unsqueeze(1).repeat(1, self.generator.n_latent, 1)
    # mean_w = self.generator.get_latent(torch.randn([self.latents.size(0), self.latent_dim], device=self.device)).unsqueeze(1).repeat(1, self.generator.n_latent, 1)
    print('mean_w :', mean_w.device)
    
    # Style refs에 대한 w를 복사해둠
    in_latent = self.latents.clone()
    print('self.latents : ', self.latents.device)
    print('in_latent :', in_latent.device)
    in_latent[:, self.id_swap] = self.alpha * self.latents[:, self.id_swap] + (1 - self.alpha) * mean_w[:, self.id_swap]
    
    

    img = self.generator(in_latent, input_is_latent=True)

    with torch.no_grad():
      real_feat = self.discriminator(self.targets.to(self.device)) 
    fake_feat = self.discriminator(img)

    loss = sum([F.l1_loss(a, b) for a, b in zip(fake_feat, real_feat)])/len(fake_feat)
    return loss

  def set_style_latents(self, style_path):
    style_imgs_list = os.listdir(style_path)

    targets = []
    latents = []
    
    for name in style_imgs_list:
      # Style Image를 얻고 이를 Crop and align
      style_path = os.path.join('style_images', name)
      assert os.path.exists(style_path), f"{style_path} does not exist!"
      # Alignment를 하고 png로 저장한다.
      name = strip_path_extension(name)
      style_aligned_path = os.path.join('style_images_aligned', f'{name}.png')
      # 이미 align한게 있다면 그냥 불러온다. 
      if not os.path.exists(style_aligned_path):
        style_aligned = align_face(style_path)
        style_aligned.save(style_aligned_path)
      else:
        style_aligned = Image.open(style_aligned_path).convert('RGB')

      # GAN Invert를 해서 w를 찾아낸다.
      style_code_path = os.path.join('inversion_codes', f'{name}.pt')
      if not os.path.exists(style_code_path):
        latent = self.projection(style_aligned, style_code_path)
      else:
        latent = torch.load(style_code_path)['latent'].to(self.device)

      print('projection latent : ', latent.device)
      # PIL -> Tensor
      # 원래는 여러개가 들어감 [N, 18, 512] 이렇게 되야 하는데 
      targets.append(self.transform(style_aligned))
      latents.append(latent)
      
    targets = torch.stack(targets, 0)
    latents = torch.stack(latents, 0)
    return targets, latents
    
  def inference(self, my_w):
    """
    Returns inversion, styled_img
    """
    with torch.no_grad():
      # Inversion
      inversion = self.style_generator(my_w, input_is_latent=True)
      # Domain styled imgs
      my_sample = self.generator(my_w, input_is_latent=True)

    return inversion, my_sample


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('--exp', type=str)
  parser.add_argument('--seed', type=int, default=3000)
  
  parser.add_argument('--iter', type=int, default=1000)
  parser.add_argument('--gpus', type=int, default=1)
  parser.add_argument('--strategy', type=str, default=None)
  parser.add_argument('--nodes', type=int, default=1)
  parser.add_argument('--num_workers', type=int, default=int(os.cpu_count()/2))
  parser.add_argument('--precision', type=str, default='float32') #float32, float16, bf16
  
  parser.add_argument('--id_swap', type=int, default=7)
  parser.add_argument('--latent_dim', type=int, default=512)
  parser.add_argument('--alpha', type=float, default=1.0)

  parser.add_argument('--style_path', type=str, default='./style_images')
  parser.add_argument('--con_path', type=str, default='./test_input')
  
  
  args = parser.parse_args()
  
  os.makedirs('inversion_codes', exist_ok=True)
  os.makedirs('style_images', exist_ok=True)
  os.makedirs('style_images_aligned', exist_ok=True)
  os.makedirs('models', exist_ok=True)
  
  # shape_predictor 있는지 체크
  # 없으면 다운
  if not os.path.exists('./models/dlibshape_predictor_68_face_landmarks.dat'):
    os.system('wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2')
    os.system('bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2')
    os.system('mv shape_predictor_68_face_landmarks.dat models/dlibshape_predictor_68_face_landmarks.dat')
    
  # pretrained weight 있는지 체크
  # 없으면 다운
  if not os.path.exists('./models/e4e_ffhq_encode.pt'):
    os.systme('gdown --id 1o6ijA3PkcewZvwJJ73dJ0fxhndn0nnh7 -O /home/aiteam/tykim/JoJoGAN/models/e4e_ffhq_encode.pt')
    
  # pretrained weight 있는지 체크
  # 없으면 다운
  if not os.path.exists('./models/stylegan2-ffhq-config-f.pt'):
    os.systme('gdown --id 1Yr7KuD959btpmcKGAUsbAk5rPjX2MytK -O /home/aiteam/tykim/JoJoGAN/models/stylegan2-ffhq-config-f.pt')
    
  
  torch.manual_seed(args.seed)
  torch.backends.cudnn.benchmark = True

  # PL setup
  dm = DataModule(style_path=args.style_path)
  dm.prepare_data()
  dm.setup("fit")
  # jjGan = JJGAN(args, dm.train[0]['style_latents'], dm.train[0]['style_targets'])
  jjGan = JJGAN(args)
  
  tpgb_cb = TQDMProgressBar(refresh_rate=10)

  if args.precision == 'float32':
    args.precision = 32
  elif args.precision == 'float16':
    args.precision = 16

  
  trainer = Trainer(max_epochs=args.iter, gpus=args.gpus, strategy=args.strategy,
                    num_nodes=args.nodes, precision=args.precision, callbacks=[tpgb_cb],
                    accelerator="gpu")

  # print(args.gpus)
  jjGan.datamodule = dm
  trainer.fit(jjGan, dm)