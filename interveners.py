import torch
import torch.nn as nn

def wrapper(intervener):
    def wrapped(*args, **kwargs):
        return intervener(*args, **kwargs)
    return wrapped

class Collector():
    collect_state = True
    collect_action = False  
    def __init__(self, multiplier, head, module_type, num_heads=None, head_dim=None):
        self.head = head
        self.module_type = module_type
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.states = []
        self.actions = []
    def reset(self):
        self.states = []
        self.actions = []
    def __call__(self, b, s): 
        if self.head == -1:
            self.states.append(b[0, -1].detach().clone())  # original b is (batch_size, seq_len, #key_value_heads x D_head)
        else:
            vec = b[0, -1]
            num_heads = self.num_heads
            if num_heads is None:
                if self.head_dim is None:
                    raise ValueError("Collector requires num_heads or head_dim when head != -1")
                if vec.shape[-1] % self.head_dim != 0:
                    raise ValueError(f"head_dim={self.head_dim} does not divide hidden size {vec.shape[-1]}")
                num_heads = vec.shape[-1] // self.head_dim
            head_dim = vec.shape[-1] // num_heads
            self.states.append(vec.view(num_heads, head_dim)[self.head].detach().clone())
        return b
    
class ITI_Intervener():
    collect_state = False
    collect_action = False
    attr_idx = -1
    def __init__(self, direction, multiplier):
        if not isinstance(direction, torch.Tensor):
            direction = torch.tensor(direction)
        self.direction = direction.cuda().half()
        self.multiplier = multiplier
        self.states = []
        self.actions = []
    def reset(self):
        self.states = []
        self.actions = []
    def __call__(self, b, s): 
        # self.states.append(b[0, -1].detach().clone())  # original b is (batch_size=1, seq_len, #head x D_head), now it's (#head x D_head)
        # action = self.direction.to(b.device)
        # self.actions.append(action.detach().clone())
        # b[0, -1] = b[0, -1] + action * self.multiplier
        # self.states.append(b.detach().clone())  # original b is (batch_size=1, seq_len, #head x D_head), now it's (#head x D_head)
        # B, L, H = b.size()
        # print("interven hidden dim", self.direction.size())
        action = self.direction.unsqueeze(0).unsqueeze(0).to(b.device)
        # print("action", action.size())
        # self.actions.append(action.detach().clone())
        b = b + action * self.multiplier
        
        return b

# import torch
import torch.nn.functional as F  
class VQC_Intervener():
    collect_state = False
    collect_action = False
    attr_idx = -1
    def __init__(self, head_vq_adaptors, head_num, multiplier):
        # if not isinstance(direction, torch.Tensor):
        #     direction = torch.tensor(direction)
        # self.direction = direction.cuda().half()
        self.multiplier = multiplier
        self.vq_adaptors = head_vq_adaptors
        self.non_none_indices = [i for i, item in enumerate(self.vq_adaptors) if item is not None]
        self.head_num = head_num
        self.states = []
        self.actions = []
    def reset(self):
        self.states = []
        self.actions = []
    def __call__(self, b, s): 
        # states = b.detach().clone()
        states = b
        # self.states.append(states)  # original b is (batch_size=1, seq_len, #head x D_head), now it's (#head x D_head)
        # hidden_dim = states.size()[0]
        B, L, H = states.size()
        # print("interven hidden dim", hidden_dim)
        # print("states", states.device)
        
        # head_vq_adaptors = self.vq_adaptors.to(b.device)
        
        action = torch.zeros_like(states).to(b.device)
        head_dim = H // self.head_num
        # print("head_dim", head_dim)
        # for j in range(L):
        # non_none_indices = [i for i, item in enumerate(my_list) if item is not None]
        for i in self.non_none_indices:
            with torch.no_grad():
                if self.vq_adaptors[i] is not None:
                    adaptor = self.vq_adaptors[i].to(b.device)
                    adaptor.eval()
                    # print("states[i * head_dim: (i + 1) * head_dim].view(1,-1)", states[i * head_dim: (i + 1) * head_dim].view(1,-1).size())
                    _,_,_,_,z_q = adaptor(states[:, :,i * head_dim: (i + 1) * head_dim].view(-1, head_dim))
                    z_q_flat = z_q
                    truthdir = z_q_flat + adaptor.direction.unsqueeze(0)
                    falsedir = z_q_flat - adaptor.direction.unsqueeze(0)
                    # z_q = F.normalize(z_q, p=2, dim=-1)
                    truthdir = F.normalize(truthdir, p=2, dim=-1)
                    falsedir = F.normalize(falsedir, p=2, dim=-1)
                    decoded_truth = adaptor.decoder(truthdir) - adaptor.decoder(falsedir)
                    # decoded_truth = adaptor.decoder(truthdir)
                    action[:, :,i * head_dim: (i + 1) * head_dim] = F.normalize(decoded_truth, p=2, dim=-1).view(B,L, -1)
                    
                        # *torch.norm(states.view(1,-1), p=2, dim=-1)
                        # Delta = recon_x_pos - recon_x_neg
                    # Delta = Delta.contiguous().to(X.dtype)
                    # Delta = F.normalize(Delta, p=2, dim=-1).type_as(X) * torch.norm(
                    # X, p=2, dim=-1
                    #     ).unsqueeze(2)
        # print("action size", action.size())
        # print("states size", states.size())
        # # print("action", action.cpu().numpy())
        # print("action diff", F.mse_loss(states, action)) 
        # print("state norm" , torch.norm(states, p=2, dim=-1))
        # print("action norm" , torch.norm(action, p=2, dim=-1))
        # self.actions.append(action.detach().clone())
        b = b + action * self.multiplier
        
        # self.actions.append(action.detach().clone())
        
        
        # b[0, -1] = b[0, -1] + action * self.multiplier
        return b
