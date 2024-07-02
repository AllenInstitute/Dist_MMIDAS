import pprint
import os
import argparse
import functools
import torch
import torch.nn as nn 
import torch.overrides as overrides
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms 
from torch.optim.lr_scheduler import StepLR
import torch.distributed as dist 
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
  CPUOffload,
  BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
  always_wrap_policy,
  size_based_auto_wrap_policy,
  enable_wrap,
  wrap,
)
from datetime import timedelta
from tqdm import trange
import wandb
import mmidas
import string
import random
import threading
import time
import matplotlib.pyplot as plt
import gc

def subset_dataset(dataset, percent):
  return torch.utils.data.Subset(dataset, list(range(int(len(dataset) * percent))))

def reset_cuda_for_next_run(world_size):
  torch.cuda.empty_cache()
  for rank in range(world_size):
    torch.cuda.reset_peak_memory_stats(rank)
    torch.cuda.reset_accumulated_memory_stats(rank)
  torch.cuda.synchronize()
  torch.cuda.ipc_collect()

def record_memory_history(fname, folder='memory-snapshots'):
  if not os.path.exists(folder):
    os.makedirs(folder)
  fname = f"{folder}/{fname}"
  torch.cuda.memory._dump_snapshot(fname)
  dprint(f"> saved memory snapshot to {fname}")

def avg(xs):
  return sum(xs) / len(xs)

def bold(s):
  match s:
    case int(x):
      return f"\033[1mrank {s}\033[0m"
    case _:
      return f"\033[1m{s}\033[0m"

# %%
def rank_vs_device(rank, color='yellow'):
  dprint(f"rank: {rank} vs current device: {torch.cuda.current_device()}", color=color, bold=True)

def count_params(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)

class MemoryLogger:
  def __init__(self, world_size, interval=0.005, run=None, rank=None):
    # self.memory_allocated = [[] for _ in range(world_size)]
    self.memory_allocated = []
    self.max_memory_allocated = []
    self.running = False
    self.interval = interval
    self.world_size = world_size
    self.run = run
    if not (type(rank) == int):
      self.rank = torch.cuda.current_device()
    else:
      self.rank = rank
    self.synced = False

  def log_memory(self):
    torch.cuda.set_device(self.rank)
    start = time.time()
    while self.running:
      self.memory_allocated.append(MB(torch.cuda.memory_allocated(self.rank)))
      self.max_memory_allocated.append(MB(torch.cuda.max_memory_allocated(self.rank)))
      if self.run is not None:
        self.run.log({f'rank {self.rank} memalloc': self.memory_allocated[-1],
                      f'rank {self.rank} max memalloc': self.max_memory_allocated[-1]})
      time.sleep(self.interval)
    mem_avg = avg(self.memory_allocated)
    dprint(f"{bold(self.rank)}: average memory allocated: {mem_avg}MB")
    if self.run is not None:
      self.run.log({f'rank {self.rank} logger avg memalloc': mem_avg})
    
  def start(self):
    dprint(f"{bold(self.rank)}: starting memory logger")
    self.running = True
    self.thread = threading.Thread(target=self.log_memory)
    self.thread.start()

  def stop(self):
    dprint(f"{bold(self.rank)}: stopping memory logger")
    self.running = False
    self.thread.join()

  def synchronize(self):
    self.synced = True

  def get(self):
    assert not self.running
    return self.memory_allocated
  
  def get_synchronized(self):
    self.stop()
    return self.memory_allocated

def get_plotter(s):
  match s:
    case 'line':
      return plt.plot
    case 'scatter':
      return plt.scatter
    case _:
      raise ValueError(f"invalid plot type: {s}")

def truncate(lst, n):
  ret = []
  for i in range(min(n, len(lst))):
    ret.append(lst[(len(lst) // n) * i])
  return ret
  

def MB(bytes):
  return bytes / 1024**2

def random_string(n):
  return ''.join(random.choices(string.ascii_lowercase + string.digits, k=n))

class DeepestNet(nn.Module):
  def __init__(self):
    super(DeepestNet, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, 3, 1)
    self.conv2 = nn.Conv2d(32, 64, 3, 1)
    self.dropout1 = nn.Dropout(0.25)
    self.dropout2 = nn.Dropout(0.5)
    self.fc1 = nn.Linear(9216, 9000)
    self.fc1a = nn.Linear(9000, 2000)
    self.fc1b = nn.Linear(2000, 2000)
    self.fc1c = nn.Linear(2000, 2000)
    self.fc1d = nn.Linear(2000, 2000)
    self.fc1e = nn.Linear(2000, 2000)
    self.fc1f = nn.Linear(2000, 2000)
    self.fc1g = nn.Linear(2000, 2000)
    self.fc1h = nn.Linear(2000, 2000)
    self.fc1i = nn.Linear(2000, 2000)
    self.fc1j = nn.Linear(2000, 2000)
    self.fc1k = nn.Linear(2000, 2000)
    self.fc1l = nn.Linear(2000, 2000)
    self.fc1m = nn.Linear(2000, 2000)
    self.fc1n = nn.Linear(2000, 2000)
    self.fc1o = nn.Linear(2000, 2000)
    self.fc1p = nn.Linear(2000, 2000)
    self.fc1q = nn.Linear(2000, 2000)
    self.fc1r = nn.Linear(2000, 2000)
    self.fc1s = nn.Linear(2000, 2000)
    self.fc1t = nn.Linear(2000, 2000)
    self.fc1u = nn.Linear(2000, 2000)
    self.fc1v = nn.Linear(2000, 2000)
    self.fc1w = nn.Linear(2000, 2000)
    self.fc1x = nn.Linear(2000, 2000)
    self.fc1y = nn.Linear(2000, 2000)
    self.fc3 = nn.Linear(2000, 2000)
    self.fc4 = nn.Linear(2000, 2000)
    self.fc5 = nn.Linear(2000, 2000)
    self.fc6 = nn.Linear(2000, 2000)
    self.fc7 = nn.Linear(2000, 2000)
    self.fc8 = nn.Linear(2000, 2000)
    self.fc9 = nn.Linear(2000, 2000)
    self.fc10 = nn.Linear(2000, 2000)
    self.fc11 = nn.Linear(2000, 2000)
    self.fc12 = nn.Linear(2000, 2000)
    self.fc13 = nn.Linear(2000, 2000)
    self.fc14 = nn.Linear(2000, 2000)
    self.fc15 = nn.Linear(2000, 2000)
    self.fc16 = nn.Linear(2000, 2000)
    self.fc17 = nn.Linear(2000, 2000)
    self.fc18 = nn.Linear(2000, 2000)
    self.fc19 = nn.Linear(2000, 2000)
    self.fc20 = nn.Linear(2000, 2000)
    self.fc21 = nn.Linear(2000, 2000)
    self.fc22 = nn.Linear(2000, 2000)
    self.fc23 = nn.Linear(2000, 2000)
    self.fc24 = nn.Linear(2000, 2000)
    self.fc25 = nn.Linear(2000, 2000)
    self.fc26 = nn.Linear(2000, 2000)
    self.fc27 = nn.Linear(2000, 2000)
    self.fc28 = nn.Linear(2000, 2000)
    self.fc29 = nn.Linear(2000, 2000)
    self.fc30 = nn.Linear(2000, 2000)
    self.fc31 = nn.Linear(2000, 2000)
    self.fc32 = nn.Linear(2000, 2000)
    self.fc33 = nn.Linear(2000, 2000)
    self.fc34 = nn.Linear(2000, 2000)
    self.fc35 = nn.Linear(2000, 2000)
    self.fc36 = nn.Linear(2000, 2000)
    self.fc37 = nn.Linear(2000, 2000)
    self.fc38 = nn.Linear(2000, 2000)
    self.fc39 = nn.Linear(2000, 2000)
    self.fc40 = nn.Linear(2000, 2000)
    self.fc41 = nn.Linear(2000, 2000)
    self.fc42 = nn.Linear(2000, 2000)
    self.fc43 = nn.Linear(2000, 2000)
    self.fc44 = nn.Linear(2000, 2000)
    self.fc45 = nn.Linear(2000, 2000)
    self.fc46 = nn.Linear(2000, 2000)
    self.fc47 = nn.Linear(2000, 2000)
    self.fc48 = nn.Linear(2000, 2000)
    self.fc49 = nn.Linear(2000, 2000)
    self.fc50 = nn.Linear(2000, 2000)
    self.fc51 = nn.Linear(2000, 2000)
    self.fc52 = nn.Linear(2000, 2000)
    self.fc53 = nn.Linear(2000, 2000)
    self.fc54 = nn.Linear(2000, 2000)
    self.fc55 = nn.Linear(2000, 2000)
    self.fc56 = nn.Linear(2000, 2000)
    self.fc2a = nn.Linear(2000, 128)
    self.fc2 = nn.Linear(128, 10)

  def forward(self, x):
    x = self.conv1(x)
    x = F.relu(x)
    x = self.conv2(x)
    x = F.relu(x)
    x = F.max_pool2d(x, 2)
    x = self.dropout1(x)
    x = torch.flatten(x, 1)
    x = self.fc1(x)
    x = F.relu(x)
    x = self.dropout2(x)
    x = self.fc1a(x)
    x = self.fc1b(x)
    x = self.fc1c(x)
    x = self.fc1d(x)
    x = self.fc1e(x)
    x = self.fc1f(x)
    x = self.fc1g(x)
    x = self.fc1h(x)
    x = self.fc1i(x)
    x = self.fc1j(x)
    x = self.fc1k(x)
    x = self.fc1l(x)
    x = self.fc1m(x)
    x = self.fc1n(x)
    x = self.fc1o(x)
    x = self.fc1p(x)
    x = self.fc1q(x)
    x = self.fc1r(x)
    x = self.fc1s(x)
    x = self.fc1t(x)
    x = self.fc1u(x)
    x = self.fc1v(x)
    x = self.fc1w(x)
    x = self.fc1x(x)
    x = self.fc1y(x)
    x = self.fc3(x)
    x = self.fc4(x)
    x = self.fc5(x)
    x = self.fc6(x)
    x = self.fc7(x)
    x = self.fc8(x)
    x = self.fc9(x)
    x = self.fc10(x)
    x = self.fc11(x)
    x = self.fc12(x)
    x = self.fc13(x)
    x = self.fc14(x)
    x = self.fc15(x)
    x = self.fc16(x)
    x = self.fc17(x)
    x = self.fc18(x)
    x = self.fc19(x)
    x = self.fc20(x)
    x = self.fc21(x)
    x = self.fc22(x)
    x = self.fc23(x)
    x = self.fc24(x)
    x = self.fc25(x)
    x = self.fc26(x)
    x = self.fc27(x)
    x = self.fc28(x)
    x = self.fc29(x)
    x = self.fc30(x)
    x = self.fc31(x)
    x = self.fc32(x)
    x = self.fc33(x)
    x = self.fc34(x)
    x = self.fc35(x)
    x = self.fc36(x)
    x = self.fc37(x)
    x = self.fc38(x)
    x = self.fc39(x)
    x = self.fc40(x)
    x = self.fc41(x)
    x = self.fc42(x)
    x = self.fc43(x)
    x = self.fc44(x)
    x = self.fc45(x)
    x = self.fc46(x)
    x = self.fc47(x)
    x = self.fc48(x)
    x = self.fc49(x)
    x = self.fc50(x)
    x = self.fc51(x)
    x = self.fc52(x)
    x = self.fc53(x)
    x = self.fc54(x)
    x = self.fc55(x)
    x = self.fc56(x)
    x = self.fc2a(x)
    x = self.fc2(x)
    output = F.log_softmax(x, dim=1)
    return output
  

class DeepNet(nn.Module):
  def __init__(self):
    super(DeepNet, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, 3, 1)
    self.conv2 = nn.Conv2d(32, 64, 3, 1)
    self.dropout1 = nn.Dropout(0.25)
    self.dropout2 = nn.Dropout(0.5)
    self.fc1 = nn.Linear(9216, 9000)
    self.fc1a = nn.Linear(9000, 1000)
    self.fc1b = nn.Linear(1000, 1000)
    self.fc1c = nn.Linear(1000, 1000)
    self.fc1d = nn.Linear(1000, 128)
    self.fc2 = nn.Linear(128, 10)

  def forward(self, x):
    x = self.conv1(x)
    x = F.relu(x)
    x = self.conv2(x)
    x = F.relu(x)
    x = F.max_pool2d(x, 2)
    x = self.dropout1(x)
    x = torch.flatten(x, 1)
    x = self.fc1(x)
    x = F.relu(x)
    x = self.dropout2(x)
    x = self.fc1a(x)
    x = self.fc1b(x)
    x = self.fc1c(x)
    x = self.fc1d(x)
    x = self.fc2(x)
    output = F.log_softmax(x, dim=1)
    return output

# %%
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, 3, 1)
    self.conv2 = nn.Conv2d(32, 64, 3, 1)
    self.dropout1 = nn.Dropout(0.25)
    self.dropout2 = nn.Dropout(0.5)
    self.fc1 = nn.Linear(9216, 128)
    self.fc2 = nn.Linear(128, 10)

  def forward(self, x):
    x = self.conv1(x)
    x = F.relu(x)
    x = self.conv2(x)
    x = F.relu(x)
    x = F.max_pool2d(x, 2)
    x = self.dropout1(x)
    x = torch.flatten(x, 1)
    x = self.fc1(x)
    x = F.relu(x)
    x = self.dropout2(x)
    x = self.fc2(x)
    output = F.log_softmax(x, dim=1)
    return output

def plot(data, plot_t, xlabel, ylabel, title, legend, fname, folder=''):
  plotter = get_plotter(plot_t)
  plotter(range(len(data)), data)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.title(title)
  plt.legend([legend])
  if not os.path.exists(folder) and not (folder == ''): 
    os.makedirs(folder)
  plt.savefig(f"{folder}/{fname}")
  dprint(f"> saved plot to {folder}/{fname}")
  plt.close()


def trivial(rank, world_size, args):
  parallel = not args.no_parallel

  setup(rank, world_size, parallel=parallel, is_multinode=args.multinode)
  dprint(f"> starting trivial test on rank {rank}")
  if parallel:
    local_rank = rank if not args.multinode else int(os.environ['SLURM_LOCALID'])
    torch.cuda.set_device(local_rank)
  for _ in range(args.repeat):  
    t = torch.tensor([1, 2, 3], device=rank)
    dprint(f"rank {rank} - before reduce: {t}")
    if parallel:
      dist.all_reduce(t, op=dist.ReduceOp.SUM)
      dprint(f"rank {rank} - after reduce: {t}")
  cleanup(parallel=parallel)

# [] TODO can make this slightly more efficient
def _dprint():
  color_code = {
    'black': '30',
    'red': '31',
    'green': '32',
    'yellow': '33',
    'blue': '34',
    'magenta': '35',
    'cyan': '36',
    'white': '37',
  }
  def dprint(*args, color=None, bold=False, italics=False, **kwargs):
    if __debug__:
      start_code = "\033["
      end_code = "\033[0m"
      codes = []
      if bold:
        codes.append('1')
      if italics:
        codes.append('3')
      if color is not None:
        codes.append(color_code[color])
      if len(codes) > 0:
        start_code += ';'.join(codes) + 'm'
        print(start_code, end='')
      print(*args, **kwargs)
      print(end_code, end='')
  return dprint

dprint = _dprint()

def print_gpus():
  dprint(f"gpus: {torch.cuda.device_count()}")
  for i in range(torch.cuda.device_count()):
    dprint(f"\tgpu {i}: {torch.cuda.get_device_name(i)}")

def eparams(model):
  return list(model.parameters())

def setup(rank, world_size, backend='nccl', is_multinode=False, timeout=120, parallel=True):
  if not parallel:
    return
  if is_multinode:
    os.environ['MASTER_ADDR'] = os.environ['SLURM_SUBMIT_HOST']
    os.environ['MASTER_PORT'] = os.getenv('MASTER_PORT', '12355')
  else:
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
  dprint(f"rank {rank} - master addr: {os.environ['MASTER_ADDR']}")
  dprint(f"rank {rank} - master port: {os.environ['MASTER_PORT']}")
  assert not ("A100" in torch.cuda.get_device_name(rank) and backend == 'nccl'), "a100 doesn't support nccl"
  dist.init_process_group(backend=backend, rank=rank, world_size=world_size, timeout=timedelta(seconds=timeout))

def cleanup(parallel=True):
  if not parallel:
    return
  dist.destroy_process_group()

def rank2dev(rank):
  if type(rank) == int:
    return f"cuda:{rank}"
  return rank

def make_data_loaders(task, parallel, batch_size, test_batch_size, percent, **kwargs):
  match task, parallel:
    case 'mnist', True:
      dprint(f"> making dataloader: mnist (parallel: {parallel})")
      rank = kwargs['rank']
      world_size = kwargs['world_size']

      transform = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize((0.1307,), (0.3081,))
      ])
      dataset1 = datasets.MNIST('../data', train=True, download=False, transform=transform)
      dataset2 = datasets.MNIST('../data', train=False, transform=transform)
      dataset1 = subset_dataset(dataset1, percent)
      dataset2 = subset_dataset(dataset2, percent)
      sampler1 = DistributedSampler(dataset1, rank=rank, num_replicas=world_size, shuffle=True)
      sampler2 = DistributedSampler(dataset2, rank=rank, num_replicas=world_size)
      train_kwargs = {'batch_size': batch_size, 'sampler': sampler1}
      test_kwargs = {'batch_size': test_batch_size, 'sampler': sampler2}
      cuda_kwargs = {'num_workers': 2, 'pin_memory': True, 'shuffle': False, 'drop_last': True} # changed from tutorial
      train_kwargs.update(cuda_kwargs)
      test_kwargs.update(cuda_kwargs)
      train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
      test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
      return train_loader, test_loader, sampler1
    case 'mnist', False:
      dprint(f"> making dataloader: mnist (parallel {parallel})")
      transform = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize((0.1307,), (0.3081,))
      ])
      dataset1 = datasets.MNIST('../data', train=True, download=False, transform=transform)
      dataset2 = datasets.MNIST('../data', train=False, transform=transform)
      dataset1 = subset_dataset(dataset1, percent)
      dataset2 = subset_dataset(dataset2, percent)
      train_loader = torch.utils.data.DataLoader(dataset1, batch_size=batch_size, shuffle=True, drop_last=True)
      test_loader = torch.utils.data.DataLoader(dataset2, batch_size=test_batch_size, shuffle=False, drop_last=True)
      return train_loader, test_loader, None
    case 'mmidas', True:
      raise NotImplementedError # TODO
    case 'mmidas', False:

      raise NotImplementedError # TODO
    
# %%
# def mmidas_data

# %% 
def mmidas_dataloaders(batch_size):
  ...
  
# %%
def make_model(name, parallel, rank=None, **config):
  model = None
  dest = None
  dprint(f"> rank {rank} - making model: {name}")
  match name:
    case 'net':
      model = Net()
    case 'deep':
      model = DeepNet()
    case 'deepest':
      model = DeepestNet()
    case 'mmidas':
      raise NotImplementedError # TODO
    case _:
      raise ValueError(f"invalid model: {name}")
  return model


# %%
def transform_model(model, is_fsdp, is_jit, rank, wrap, min_params=1000, offload=None):
  if is_fsdp:
    if rank == 0:
      dprint(f"> transforming: fsdp")
    my_auto_wrap_policy = None
    match wrap:
      case 'size_based':
        my_auto_wrap_policy = functools.partial(
          size_based_auto_wrap_policy, min_num_params=min_params
        )
        dprint(f"\trank {rank} - > applying sized based wrap policy (min params: {min_params})")
      case 'always':
        my_auto_wrap_policy = always_wrap_policy
        dprint(f"\trank {rank} - > applying always wrap policy")
      case 'none':
        my_auto_wrap_policy = None
      case _:
        raise ValueError(f"invalid wrap policy: {wrap}")
    cpu_offload = CPUOffload(offload) if offload else None
    if offload: dprint(f"\trank {rank} - > offloading to cpu: {offload}")
    # maybe pass in device_id
    assert rank == torch.cuda.current_device()
    model = FSDP(model, auto_wrap_policy=my_auto_wrap_policy, cpu_offload=cpu_offload, use_orig_params=is_jit, device_id=rank)
  if is_jit:
    dprint(f"> transforming: jit")
    model = torch.compile(model)
  return model


def train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=None, parallel=True, run=None, print_loss=True, is_reduce=True):
  model.train()
  ddp_loss = torch.zeros(2).to(rank)
  mem = []
  if sampler:
    sampler.set_epoch(epoch)
  for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(rank), target.to(rank) # TODO
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target, reduction='sum')
    _mem_alloc = MB(torch.cuda.memory_allocated(rank))
    mem.append(_mem_alloc)
    if run is not None:
      run.log({f'cuda {rank} memory allocated': _mem_alloc})
    loss.backward()
    optimizer.step()
    ddp_loss[0] += loss.item()
    ddp_loss[1] += len(data)
  if parallel and is_reduce:
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
  if print_loss:
    dprint('Train Epoch: {} \tLoss: {:.6f}, \t Rank {} memory allocated: {}'.format(epoch, ddp_loss[0] / ddp_loss[1], rank, mem[-1]))
  if run is not None:
    assert torch.cuda.max_memory_allocated() == torch.cuda.max_memory_allocated(rank)
    run.log({'train_loss': ddp_loss[0] / ddp_loss[1],
             f'rank {rank} max memalloc': torch.cuda.max_memory_allocated(rank) / 1024**2,})
  return mem

def test(model, rank, world_size, test_loader, parallel=True, print_loss=True, is_reduce=True):
    model.eval()
    correct = 0
    ddp_loss = torch.zeros(3).to(rank)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(rank), target.to(rank)
            output = model(data)
            ddp_loss[0] += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            ddp_loss[1] += pred.eq(target.view_as(pred)).sum().item()
            ddp_loss[2] += len(data)
    if parallel and is_reduce:
      dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

    if print_loss:
      test_loss = ddp_loss[0] / ddp_loss[2]
      dprint('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
          test_loss, int(ddp_loss[1]), int(ddp_loss[2]),
          100. * ddp_loss[1] / ddp_loss[2]))

def model_run(rank, world_size, args, run, mlogger):
  parallel=  not args.no_parallel
  is_fsdp = not args.no_fsdp and parallel
  print_train_loss = 'train' not in args.no_loss
  print_test_loss = 'test' not in args.no_loss
  plot_memory = 'memory' in args.plot
  master_rank = rank == 0 or not parallel

  if parallel:
    assert torch.cuda.is_available()

  _name = f"({torch.cuda.get_device_name(rank)})" if rank == 'cuda' or isinstance(rank, int) else ''
  dprint(f"> training on {rank2dev(rank)} {_name} (host: {os.uname().nodename})")

  if parallel:
    torch.cuda.set_device(rank)

  train_loader, test_loader, sampler = make_data_loaders(task=args.task, parallel=(parallel and not args.no_sampler),
                                              batch_size=args.batch_size,
                                              test_batch_size=args.test_batch_size, rank=rank, world_size=world_size, percent=args.percent)
  model = make_model(args.model, parallel, rank=rank)
  if not is_fsdp:
    model = model.to(rank)
  model = transform_model(model, (is_fsdp and parallel), args.jit, rank, args.wrap, args.min_params, args.cpu_offload)

  optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
  scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

  if 'model' in args.plot and run is not None:
    run.watch(model, log='all', log_freq=100, idx=(rank if type(rank) == int else None), log_graph=True)
  
  rets = {
    'model': model,
    'epoch_times': [],
    'mems': [],
  }
  pbar = trange(args.epochs, colour='red') if master_rank else range(args.epochs)
  for epoch in pbar:
    start = time.time()
    rets['mems'] += train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=sampler, parallel=parallel, run=run, print_loss=master_rank and print_train_loss)
    test(model=model, rank=rank, world_size=world_size, test_loader=test_loader, parallel=parallel, print_loss=master_rank and print_test_loss)
    scheduler.step()
    rets['epoch_times'].append(time.time() - start)
    if run is not None:
      run.log({'seconds per epoch': rets['epoch_times'][-1]})
  return rets

def fsdp_main(rank, world_size, args):
  parallel = not args.no_parallel
  is_fsdp = not args.no_fsdp and parallel
  print_train_loss = 'train' not in args.no_loss
  print_test_loss = 'test' not in args.no_loss
  plot_memory = 'memory' in args.plot
  master_rank = rank == 0 or not parallel # TODO

  if parallel:
    assert torch.cuda.is_available()

  setup(rank, world_size, args.backend, args.multinode, args.timeout, parallel=parallel)
  if parallel:
    torch.cuda.set_device(rank)

  _name = f"({torch.cuda.get_device_name(rank)})" if rank == 'cuda' or isinstance(rank, int) else ''
  dprint(f"> training on {rank2dev(rank)} {_name} (host: {os.uname().nodename})")


  # if args.time_cuda:
  init_start_event = torch.cuda.Event(enable_timing=True)
  init_end_event = torch.cuda.Event(enable_timing=True)

  run = None
  if args.wandb:
    wandb.require('service')
    run = wandb.init(project='dist-mmidas',
                    group=f"{args.task}-{args.id}",
                    config=args)
    
  epoch_times = []
  mems = []
  mlogger = None
  if plot_memory:
    mlogger = MemoryLogger(world_size, interval=args.interval, run=run, rank=rank)
  init_start_event.record()
  model = None
  for i in range(args.runs):
    rets = model_run(rank, world_size, args, run, mlogger)
    epoch_times += rets['epoch_times']
    mems += rets['mems']
    if i == args.runs - 1:
      model = rets['model']
    else:
      del rets['model']
      del rets
      reset_cuda_for_next_run(world_size)
      gc.collect()

    # if args.record_memory_history:
    #   torch.cuda.memory._record_memory_history()

    # if args.log_after_epoch < 0 and mlogger is not None:
    #   assert not mlogger.running
    #   mlogger.start()

    #   if args.log_after_epoch == epoch and mlogger is not None and i == 0:
    #     assert not mlogger.running
    #     mlogger.start()

  init_end_event.record()
  if mlogger is not None:
    mlogger.stop()

  mems_tensor = torch.zeros(world_size, device=rank)
  if parallel:
    mems_tensor[rank] = avg(mems)
    dist.all_reduce(mems_tensor, op=dist.ReduceOp.SUM)
  if master_rank:
    dprint(" -- summary -- ", bold=True)

  if parallel:
    if run is not None:
      run.log({'avg seconds per epoch': avg(epoch_times),
              f'cuda {rank} average memory allocated': mems_tensor[rank],
              'avg memory allocated across all gpus': mems_tensor.mean()})
    if master_rank:
      dprint(f"avg seconds per epoch: {avg(epoch_times)}")
      dprint(f"cuda {rank} average memory allocated: {mems_tensor[rank]}")
      dprint(f"avg memory allocated across all gpus: {mems_tensor.mean()}")
  else:
    if run is not None: 
      run.log({'avg seconds per epoch': avg(epoch_times),
              f'cuda {torch.cuda.current_device()} average memory allocated': avg(mems),
              'avg memory allocated across all gpus': avg(mems)})
    if master_rank:
      dprint(f"avg seconds per epoch: {avg(epoch_times)}")
      dprint(f"cuda {torch.cuda.current_device()} average memory allocated: {avg(mems)}")
      dprint(f"avg memory allocated across all gpus: {avg(mems)}")
      
  if args.record_memory_history:
    record_memory_history(f"{time.time()}-{args.id}-{rank}.pickle")

  if rank == 0 or rank == 'cuda':
    torch.cuda.synchronize()
    dprint(f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec")
  if master_rank:
    dprint(f"{model}")
    dprint(f"number of parameters: {count_params(model)}")


  # save(save_model=args.save_model, parallel=parallel, rank=rank, model=args.)
  if args.save_model:
     if parallel:
      dist.barrier() # TODO: might give bugs
     states = model.state_dict()
     if master_rank:
         torch.save(states, "mnist_cnn.pt")
  cleanup(parallel=parallel)

# TODO: [] lots of stuff missing from this guy
def mnist_main(rank, world_size, args):
  dprint(f"> rank {rank} (example) - starting...")

  parallel = not args.no_parallel

  setup(rank, world_size)

  transform=transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.1307,), (0.3081,))
  ])

  dataset1 = datasets.MNIST('../data', train=True, download=True,
                      transform=transform)
  dataset2 = datasets.MNIST('../data', train=False,
                      transform=transform)

  sampler1 = DistributedSampler(dataset1, rank=rank, num_replicas=world_size, shuffle=True)
  sampler2 = DistributedSampler(dataset2, rank=rank, num_replicas=world_size)

  train_kwargs = {'batch_size': args.batch_size, 'sampler': sampler1}
  test_kwargs = {'batch_size': args.test_batch_size, 'sampler': sampler2}
  cuda_kwargs = {'num_workers': 2,
                  'pin_memory': True,
                  'shuffle': False}
  train_kwargs.update(cuda_kwargs)
  test_kwargs.update(cuda_kwargs)

  train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
  test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

  my_auto_wrap_policy = functools.partial(
      size_based_auto_wrap_policy, min_num_params=args.min_params
  )
  torch.cuda.set_device(rank)
  init_start_event = torch.cuda.Event(enable_timing=True)
  init_end_event = torch.cuda.Event(enable_timing=True)
  
  model = Net().to(rank)


  model = FSDP(model, auto_wrap_policy=my_auto_wrap_policy, cpu_offload=CPUOffload(args.cpu_offload) if args.cpu_offload else None)

  optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

  # ***** this is mnist_main ******
  if args.wandb:
    wandb.require('service')
    run = None
    run = wandb.init(project='dist-mmidas',
                    group=f"{args.task}-{args.id}")
    run.watch(model, log='all', log_freq=100, idx = (rank if type(rank) == int else None), log_graph=True)

  scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
  init_start_event.record()
  for epoch in trange(1, args.epochs + 1):
      assert parallel
      train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=sampler1, parallel=parallel, run=run)
      test(model, rank, world_size, test_loader, parallel=parallel)
      scheduler.step()

  init_end_event.record()

  if rank == 0:
      print(f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec")
      print(f"{model}")

  if args.save_model:
      # use a barrier to make sure training is done on all ranks
      dist.barrier()
      states = model.state_dict()
      if rank == 0:
          torch.save(states, "mnist_cnn.pt")

  cleanup()

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                        help='input batch size for testing (default: 128)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--seed', type=int, default=546, metavar='S',
                        help='random seed (default: 546)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--nccl-debug', action='store_true', default=False, help='enable NCCL debugging')
    parser.add_argument('--backend', type=str, default='nccl', help='distributed backend (default: nccl)')
    parser.add_argument('--multinode', action='store_true', default=False, help='enable multinode training') # TODO
    parser.add_argument('--no-parallel', action='store_true', default=False, help='disable parallelism')
    parser.add_argument('--min-params', type=int, default=20000, help='minimum number of parameters to wrap')
    parser.add_argument('--no-wrap', action='store_true', default=False, help='disable wrapping')
    parser.add_argument('--cpu-offload', action='store_true', default=False, help='enable CPU offload')
    parser.add_argument('--timeout', type=int, default=120, help='timeout for distributed ops (default: 120)')
    parser.add_argument('--trivial', action='store_true', default=False, help='run trivial test')
    parser.add_argument('--implementation', type=str, default='torch', help='implementation, torch or jax (default: torch)') # TODO
    parser.add_argument('--mixed', action='store_true', default=False, help='use mixed precision for faster training') # TODO
    parser.add_argument('--jit', action='store_true', default=False, help='jit compile the model for faster training') # TODO
    parser.add_argument('--wandb', action='store_true', default=False, help='log to wandb') # TODO
    parser.add_argument('--device', type=str, default='cuda', help='device to use (default: cuda)')
    parser.add_argument('--task', type=str, default='mmidas', help='name of the model to train (default: mmidas-smartdq). Options: mmidas-smartseq, mmidas-10x, mnist, trivial') # TODO
    parser.add_argument('--no_fsdp', action='store_true', default=False, help='disable fsdp')
    parser.add_argument('--time-cuda', action='store_true', default=False, help='time cuda ops') # TODO
    parser.add_argument('--world-size', type=int, default=-1, help='world size for distributed training (default: -1)')
    parser.add_argument('--example', action='store_true', default=False, help='run mnist example code')
    parser.add_argument('--repeat', type=int, default=1, help='number of times to repeat trivial test')
    parser.add_argument('--no-sampler', action='store_true', default=False, help='disable distributed sampler')
    parser.add_argument('--backward_prefetch', action='store_true', default=False, help='enable backward prefetching') # TODO
    parser.add_argument('--wrap', type=str, default='size_based', choices=['size_based', 'always', 'none'], help='fsdp wrap policy (default: size_based). Options: size_based, always, none') # TODO
    parser.add_argument('--default', action='store_true', default=False, help='use default settings') # TODO
    parser.add_argument('--plot', nargs='+', default=['time', 'loss'], help='plot memory usage (default: memory)') # TODO
    parser.add_argument('--plot-style', type=str, choices=['line', 'scatter'], default='line', help='plot type (default: line)')
    parser.add_argument('--plot-backend', type=str, default='matplotlib', help='plot backend (default: wandb). Options: matplotlib, plotly, seaborn') # TODO
    parser.add_argument('--model', type=str, default='net', help='model to train (default: net). Options: mmidas, net') # TODO
    parser.add_argument('--parallel', type=str, default='fsdp', help='parallel training method (default: fsdp). Options: fsdp, ddp, none') # TODO
    parser.add_argument('--no-loss', nargs='+', default=[], help='losses to disable (default: [])')
    parser.add_argument('--no-reduce', action='store_true', default=False, help='disable reduce ops') # TODO
    parser.add_argument('--record-memory-history', action='store_true', default=False, help='record memory history')
    parser.add_argument('--tensor-parallel', action='store_true', default=False, help='use tensor parallelism') # TODO
    parser.add_argument('--interval', type=float, default=1, help='memory logging interval (default: 0.005)')
    parser.add_argument('--config', type=str, default='default', help='run config (default: default)') # TODO
    parser.add_argument('--log-after-epoch', type=int, default=-1, help='log after epoch (default: -1)') # TODO
    parser.add_argument('--truncation', type=int, default=7, help='truncate memory history (default: 7)')
    parser.add_argument('--runs', type=int, default=1, help='number of runs (default: 1)') # TODO
    parser.add_argument('--percent', type=float, default=1.0, help='percent of data to use (default: 1.0)') # TODO
    parser.add_argument('--id', type=str, default='', help='experiment id (default: random)') # TODO

    # mmidas-smartseq args
    # parser.add_argument('--categories', type=int, default=120, help="(maximum) number of cell types (default: 120)")
    # parser.add_argument('--state_dim', type=int, default=2, help="state variable dimension (default: 2)")
    # parser.add_argument('--arms', type=int, default=2, help="number of mixVAE arms for each modality (default: 2)")
    # parser.add_argument('--temp', type=float, default=1.0, help="gumbel-softmax temperature (default: 1.0)")
    # parser.add_argument('--tau', type=float, default=.005, help="softmax temperature (default: .005)")
    # parser.add_argument('--beta', type=float, default=.01, help="KL regularization parameter (default: .01)")
    # parser.add_argument('--lam', type=float, default=1.0, help="coupling factor (default: 1.0)")
    # parser.add_argument('--lam_pc', type=float, default=1.0, help="coupling factor for ref arm (default: 1.0)")
    # parser.add_argument('--latent_dim', type=int, default=10, help="latent dimension (default: 10)")
    # parser.add_argument('--epochs2', type=int, default=10000, help="Number of epochs to train (default: 10000)")
    # parser.add_argument('--epochs_p', type=int, default=10000, help="Number of epochs to train pruning algorithm (default: 10000)")
    # parser.add_argument('--min_con', type=float, default=.99, help="minimum consensus (default: .99)")
    # parser.add_argument('--max_prun_it', type=int, default=50, help="maximum number of pruning iterations (default: 50)")
    # parser.add_argument('--ref_pc', action='store_true', default=False, help="use a reference prior component")
    # parser.add_argument('--fc_dim', type=int, default=100, help="number of nodes at the hidden layers (default: 100)")
    # parser.add_argument('--batch_size2', type=int, default=5000, help="batch size (default: 5000)")
    # parser.add_argument('--no_variational', action='store_true', default=False, help="enable variational mode")
    # parser.add_argument('--augmentation', action='store_true', default=False, help="enable VAE-GAN augmentation")
    # parser.add_argument('--lr2', type=float, default=.001, help="learning rate (default: .001)")
    # parser.add_argument('--p_drop', type=float, default=0.5, help="input probability of dropout (default: 0.5)")
    # parser.add_argument('--s_drop', type=float, default=0.2, help="state probability of dropout (default: 0.2)")
    # parser.add_argument('--pretrained', action='store_true', default=False, help="use pretrained model")
    # parser.add_argument('--n_pr', type=int, default=0, help="number of pruned categories in case of using a pretrained model (default: 0)")
    # parser.add_argument('--loss_mode', type=str, default='MSE', help="loss mode, MSE or ZINB (default: MSE)")
    # parser.add_argument('--runs', type=int, default=1, help="number of the experiment (default: 1)")
    # parser.add_argument('--hard', action='store_true', default=False, help="hard encoding")
    args = parser.parse_args()

    args.no_parallel = args.parallel == 'none' or args.world_size == 1
    if args.id == '':
      args.id = wandb.util.random_string(4) if args.wandb else random_string(4)

    dprint(f'args:', color='red', bold=True)
    for arg in sorted(vars(args)):
      if arg == 'id':
        dprint(f"{arg}: {getattr(args, arg)}", color='magenta')
      else:
        dprint(f"{arg}: {getattr(args, arg)}")
    dprint()

    # TODO: add other args checking
    assert not (args.no_parallel and args.multinode), "cannot disable parallelism and enable multinode training"
    assert not (not args.no_parallel and (args.device == 'cpu' or args.device == 'mps')), "cannot disable parallelism and use cpu or mps"

    torch.manual_seed(args.seed)

    if args.nccl_debug:
      os.environ['NCCL_DEBUG'] = 'INFO'
    
    dprint('available devices:', color='green', bold=True)
    dprint(f"cpu: {torch.cpu.is_available()}")
    dprint(f"cuda: {torch.cuda.is_available()}")
    dprint(f"mps: {torch.backends.mps.is_available()}")
    dprint()

    dprint(f"visible cuda devices: {os.environ['CUDA_VISIBLE_DEVICES']}")
    if args.task == 'trivial':
      if args.no_parallel:
        rank = args.device
        dprint(f"rank: {rank}")
        trivial(rank, 1, args)
      elif args.multinode:
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ['SLURM_PROCID'])
        dprint(f"world size: {world_size}")
        dprint(f"rank: {rank}")
        trivial(rank, world_size, args)
      else:
        world_size = torch.cuda.device_count() if args.world_size == -1 else args.world_size
        dprint(f"world size: {world_size}")
        mp.spawn(trivial,
                args=(world_size, args),
                nprocs=world_size,
                join=True)
    elif args.no_parallel:
      world_size = 1 if args.world_size == -1 else args.world_size
      rank = args.device
      dprint(f"world size: {world_size}")
      dprint(f"rank: {rank}")
      fsdp_main(rank, world_size, args)
      # main(args)
    elif args.multinode:
      raise NotImplementedError("multinode training not implemented yet")
      assert torch.cuda.device_count() == int(os.environ['SLURM_GPUS_ON_NODE'])
      world_size = int(os.environ['SLURM_NTASKS'])
      rank = int(os.environ['SLURM_PROCID'])
      local_rank = int(os.environ['SLURM_LOCALID'])
      dprint(f"world size: {world_size}")
      dprint(f"rank: {rank}")
      dprint(f"local rank: {local_rank}")
      fsdp_main(rank, world_size, args)
    else:
      world_size = torch.cuda.device_count() if args.world_size == -1 else args.world_size
      dprint(f"world size: {world_size}")
      mp.spawn(mnist_main if args.example else fsdp_main,
          args=(world_size, args),
          nprocs=world_size,
          join=True)

# TODO
# [x] --world-size flag
# [] --plot-gpu flag
# [] --plot-time flag
# [] --plot-backend flag
# [] --multinode w/ srun
# [] mmidas-smartseq no parallel

# [] handle mmidas vs mnist repeated args
# [] multiple runs flag?
# [] allow people to run without wandb
# [] print out flags to console
# [] maybe decouple dataset from dataloader
# [] test no sampler w/o printing
# [] fix mnist main
# [] play with fsdp settings
# [] options for memory logging format
# [] make --plot take a list of params
# [] allow you to change the plot memory interval
# [] give different seeds to different processes


# distributed sampler: slowed process, gave zero cuda usage

# mnist plot 
# 1 node, 1-4gpu, w/wo sampler
# [] plot average gpu usage 
# [] plot average elapsed time 


# min-num-params
# [] 1000
# [] 10000
# [] 20000


# if time:
# [] test w/ multiple nodes

# remember: srun vs sbatch vs torchrun
# [] remove seconds at beginning from plot

# [] disable some logging when python -O
# [] fix plot legend
# [] add different default run configs
# [] change the way I log memory
# [] try out pytorch ignite
# [] add config to wandb log
# [] setup github utils
# [] default cuda device of new thread is cuda:0

# when num params is ~1mil, you don't see that much impact from

# 5 epochs, 2-3 times each
# if time, try ddp
# try ddp with the constructor
# results with
# (i) shallow network with ~1 million
# (ii) deep network with ~93 million params


# mmidas 
# n_arms = 2 -> ~22 million
# n_arms = 5 -> ~50 million
# n_arms = 10 -> ~100 million

# DeepNet: see if I can get to ~50 million params
# multiple of 8 tensor sizes


# class Net(nn.Module):
#   ...

# def main(rank, world_size, args):
#   setup(rank, world_size)
#   ...
#   model = Net().to(args.device)
#   model = FSDP(model, auto_wrap_policy=...) # new!
#   train(model, ...)
#   test(model, ...)

#   cleanup()

# if __name__ == '__main__':
#   world_size = torch.cuda.device_count()
#   mp.spawn(main, args=(args,), nprocs=args.world_size, join=True)

# import torch.distributed as dist 

# def setup(rank, world_size):
#     os.environ['MASTER_ADDR'] = 'localhost'
#     os.environ['MASTER_PORT'] = '12355'
#     dist.init_process_group("nccl", rank=rank, world_size=world_size)

# def cleanup():
#     dist.destroy_process_group()