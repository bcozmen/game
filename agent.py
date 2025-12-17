import torch
import torch.nn as nn
node_types = {
    'input': 0,
    'hidden': 1,
    'output': 2,
    'bias': 3
}

class Agent:
    def __init__(self, node_start_index, edge_start_index, num_inputs=5, num_outputs=3, activation=nn.ReLU):
        self.node_start_index = node_start_index
        self.edge_start_index = edge_start_index
        
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.init_genes()
        self.activation = activation()

    def init_genes(self):
        inps = torch.full((self.num_inputs,), node_types['input'], dtype=torch.int)
        outs = torch.full((self.num_outputs,), node_types['output'], dtype=torch.int)
        bias = torch.full((1,), node_types['bias'], dtype=torch.int)

        v = torch.cat((inps, outs, bias), dim=0)  # Example node genes
        e = torch.empty((0, 3), dtype=torch.float32)  # Example connection genes
        state = torch.zeros(len(v))
        for i in range(self.num_inputs):
            for j in range(self.num_outputs):
                weight = torch.randn(1).item()
                conn = torch.tensor([[i, self.num_inputs + j, weight]], dtype=torch.float32)
                e = torch.cat((e, conn), dim=0)
        #add bias connections
        for i in range(self.num_outputs):
            weight = torch.randn(1).item()
            conn = torch.tensor([[self.num_inputs + i, self.num_inputs + self.num_outputs, weight]], dtype=torch.float32)
            e = torch.cat((e, conn), dim=0)
        
    
    
class Agent:
    def __init__(self, num_inputs=3, num_outputs=1):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.init_genes()
        self.activation = nn.ReLU()
                

    def init_genes(self):
        inps = torch.full((self.num_inputs,), node_types['input'], dtype=torch.int)
        outs = torch.full((self.num_outputs,), node_types['output'], dtype=torch.int)
        bias = torch.full((1,), node_types['bias'], dtype=torch.int)

    


        self.v = torch.cat((inps, outs, bias), dim=0)  # Example node genes
        self.bias_mask = (self.v == node_types['bias'])
        self.output_mask = (self.v == node_types['output'])
        
        self.e = torch.empty((0, 3), dtype=torch.float32)  # Example connection genes
        for i in range(self.num_inputs):
            for j in range(self.num_outputs):
                weight = torch.randn(1).item()
                conn = torch.tensor([[i, self.num_inputs + j, weight]], dtype=torch.float32)
                self.e = torch.cat((self.e, conn), dim=0)
        
        #add bias connections
        for i in range(self.num_outputs):
            weight = torch.randn(1).item()
            conn = torch.tensor([[self.num_inputs + i, self.num_inputs + self.num_outputs, weight]], dtype=torch.float32)
            self.e = torch.cat((self.e, conn), dim=0)
        
        self.init_node_values()
    
    def init_node_values(self):
        n = len(self.v)
        self.state = torch.zeros(n)

    def get_agent(self,x):
        self.state[:self.num_inputs] = x  # Assuming first 3 nodes are input nodes
        return self.state, self.v, self.e

    def set_agent(self, s):
        self.state[~self.bias_mask] = s[~self.bias_mask]
        self.state[~self.output_mask] = self.activation(self.state[~self.output_mask])
             #apply activation to all neurons except outputs 
        #output_indices = (self.v == node_types['output']).nonzero(as_tuple=True)[0]
        #non_output_indices = torch.tensor([i for i in range(len(self.v)) if i not in output_indices])
        #self.state[non_output_indices] = self.activation(self.state[non_output_indices])
    
    def forward(self, s,v,e):

        src = e[:, 0].long()
        dst = e[:, 1].long()
        weight = e[:, 2]

        messages = s[src] * weight
        aggregated_messages = torch.zeros_like(s)
        aggregated_messages.scatter_add_(0, dst, messages)
        return aggregated_messages

    def abs_constract():
        # bias cannot have inocome connections
        pass

    def split(self):
        child = Agent()
        child.node_gene = self.v.clone()
        child.conn_gene = self.e.clone()
        child.node_values = self.state.clone()
        return child

    def weight_mutation(self):
        rand_idx = torch.randint(0, len(self.e), (1,))

        if torch.rand(1).item() < 0.8:
            # perturbation
            self.e[rand_idx, 2] += torch.randn(1).item() * 0.1
        else:
            # assign new weight
            self.e[rand_idx, 2] = torch.randn(1).item()

    def add_connection():
        from_node = torch.randint(0, len(self.v), (1,)).item()
        
        nodes = torch.arange(0, len(self.v))
        possible_to_nodes = nodes[nodes != from_node]
        existing_to_nodes = self.e[self.e[:, 0] == from_node][:, 1]
        possible_to_nodes = possible_to_nodes[~torch.isin(possible_to_nodes, existing_to_nodes)]

        if len(possible_to_nodes) == 0:
            return  # No possible connection can be added

        to_node = possible_to_nodes[torch.randint(0, len(possible_to_nodes), (1,)).item()].item()
        new_conn = torch.tensor([[from_node, to_node, torch.randn(1).item()]], dtype=torch.float32)
        self.e = torch.cat((self.e, new_conn), dim=0)

    def add_node():
        if len(self.e) == 0:
            return  # No connection to split

        conn_idx = torch.randint(0, len(self.e), (1,)).item()
        conn_to_split = self.e[conn_idx]

        from_node = int(conn_to_split[0].item())
        to_node = int(conn_to_split[1].item())
        weight = conn_to_split[2].item()

        new_node_idx = len(self.v)
        self.v = torch.cat((self.v, torch.tensor([node_types['hidden']])), dim=0)

        new_conn1 = torch.tensor([[from_node, new_node_idx, 1.0]], dtype=torch.float32)
        new_conn2 = torch.tensor([[new_node_idx, to_node, weight]], dtype=torch.float32)

        #delete old connection and add new connections
        self.e = torch.cat((self.e[:conn_idx], self.e[conn_idx+1:], new_conn1, new_conn2), dim=0)
        
    def remove_connection():
        if len(self.e) == 0:
            return  # No connection to remove

        conn_idx = torch.randint(0, len(self.e), (1,)).item()
        self.e = torch.cat((self.e[:conn_idx], self.e[conn_idx+1:]), dim=0)
