node_types = {
    'input': 0,
    'hidden': 1,
    'output': 2,
    'bias': 3
}

def print_time(func):
    import time
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time:.6f} seconds")
        return result
    return wrapper
import torch.nn as nn

class VecEnvironment:
    def __init__(self, num_agents = 100, num_inputs = 3, num_outputs = 1, 
                activation = nn.ReLU,
                device = 'cuda'):
        self.device = device
        self.num_agents = num_agents
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.activation = activation()

        # V start index, E start index, x, y, energy
        self.agents = torch.zeros((num_agents,5))
        self.v = torch.tensor([node_types['input']] * num_inputs + [node_types['output']] * num_outputs + [node_types['bias']])  # Example node types
        self.v = self.v.repeat(num_agents)
        self.bias_mask = (self.v == node_types['bias'])
        self.output_mask = (self.v == node_types['output'])

        self.states = torch.zeros((num_agents * (num_inputs + num_outputs + 1)))  # +1 for bias
        self.states[self.bias_mask] = 1.0  # Initialize bias nodes to 1.0

        self.v = torch.zeros((num_agents * (num_inputs + num_outputs + 1)))
        self.e = torch.zeros((num_agents * (num_inputs * num_outputs + num_outputs)), 3)  # connections



class Environment:
    def __init__(self, num_agents = 100, num_inputs = 3, num_outputs = 1, 
                activation = nn.ReLU,
                device = 'cuda'):
        self.agents = None
        self.device = device
        self.num_agents = num_agents
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.activation = activation()
        self.init_agents()


    def init_agents(self):
        self.agents = [Agent(self.num_inputs, self.num_outputs) for _ in range(self.num_agents)]

    @print_time
    def forward_agents(self):
        x = torch.randn(len(self.agents), self.num_inputs)
        s,v,e, max_nodes = self.gather_agents()
        messages = self.evaluate_agents(s, v, e)
        self.write_agents(messages, max_nodes)
    @print_time
    def gather_agents(self):
        max_nodes = [0] 
        x = torch.randn(len(self.agents), self.num_inputs)  # Example input for each agent
        s,v,e = [],[],[]
        
        for i, agent in enumerate(self.agents):
            sa, va, ea = agent.get_agent(x[i])
            
            ea[...,0] += max_nodes[-1]
            ea[...,1] += max_nodes[-1]
            max_nodes.append(max_nodes[-1] + len(va))
            s.append(sa)
            v.append(va)
            e.append(ea)

        s, v, e = torch.cat(s), torch.cat(v), torch.cat(e)
        return s, v, e, max_nodes
    @print_time
    def evaluate_agents(self, s, v, e):
        s = s.to(self.device)
        src = e[:, 0].long().to(self.device)
        dst = e[:, 1].long().to(self.device)
        weight = e[:, 2].to(self.device)

        messages = s[src] * weight
        aggregated_messages = torch.zeros_like(s)
        aggregated_messages.scatter_add_(0, dst, messages)
        #aggregated_messages = self.activation(aggregated_messages)
        return aggregated_messages

    @print_time
    def write_agents(self, s, max_nodes):
        s = s.cpu()
        for i, agent in enumerate(self.agents):
            start_idx = max_nodes[i]
            end_idx = max_nodes[i+1]
            agent.set_agent(s[start_idx:end_idx])
