import torch


if __name__ == '__main__': 
  torch.distributed.init_process_group(backend='nccl', init_method='env://')
  # init method default : env://
  print(torch.distributed.get_rank())