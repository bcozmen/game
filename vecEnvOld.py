class vecEnv:
    def __init__(self,num_agents = 1000, inp_size=2, out_size=4):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.activation = nn.ReLU().to(self.device)
        self.inp_size = inp_size
        self.out_size = out_size
        self.num_agents = num_agents


        self.agent_id = 0

        self.agents = None  # agent_id, x, y, energy
        self.e = None  # (agent_id, src_node, dst_node, weight, active)
        self.v = None # (agent_id, node_type, state, inp_mask, out_mask, bias_mask, hidden_mask, active)
        self.create_agents()
        
    def create_agents(self):
        agent_num_v = self.inp_size + self.out_size + 1  # +1 for bias node
        self.agents = torch.rand((self.num_agents, 4)).to(self.device)  # agent_id, x, y, energy
        self.agents[:, 3] = 1.0  # energy
        self.agents[:, 0] = torch.arange(self.num_agents).to(self.device)

        v = torch.tensor([node_types['input']] * self.inp_size + [node_types['output']] * self.out_size + [node_types['bias']])
        
        e = torch.empty((0, 5), dtype=torch.float32)  # Example connection genes
        for a in range(self.num_agents):
            num_v = a * (self.inp_size + self.out_size + 1)
            for i in range(self.inp_size):
                for j in range(self.out_size):
                    weight = torch.randn(1).item()
                    conn = torch.tensor([[a, num_v + i, num_v + self.inp_size + j, weight, 1.0]], dtype=torch.float32)
                    e = torch.cat((e, conn), dim=0)
            #add bias connections
            for i in range(self.out_size):
                weight = torch.randn(1).item()
                conn = torch.tensor([[a, num_v + self.inp_size + self.out_size, num_v + self.inp_size + i, weight, 1.0]], dtype=torch.float32)
                e = torch.cat((e, conn), dim=0)
        
        self.agent_id += self.num_agents

        self.v = torch.empty((self.num_agents * (self.inp_size + self.out_size + 1),8), dtype=torch.float).to(self.device)  # agent_id, node_type, state, inp_mask, out_mask, bias_mask, hidden_mask, active
        self.v[:, 0] = torch.arange(self.num_agents).repeat_interleave(self.inp_size + self.out_size + 1).to(self.device)  # agent_id
        self.v[:, 1] = v.repeat(self.num_agents).to(self.device)  #
        self.v[:, 2] = torch.randn_like(self.v[:, 2], dtype=torch.float32).to(self.device)
        self.v[:, 3] = (self.v[:, 1] == node_types['input']).to(self.device)
        self.v[:, 4] = (self.v[:, 1] == node_types['output']).to(self.device)
        self.v[:, 5] = (self.v[:, 1] == node_types['bias']).to(self.device)
        self.v[:, 6] = (self.v[:, 1] == node_types['hidden']).to(self.device)
        self.v[:, 7] = 1.0  # active
        self.e = e.to(self.device)  # connections    def __call__(self, X = None):

    def __call__(self, X = None):
        self.forward(X)
    def forward(self, X = None):
        X = self.sense() if X is None else X
        actions = self.decide(X)
        self.apply(actions)
    
    def sense(self):
        #get agent positions
        positions = self.agents[:, 0:3].reshape(-1, 3)  # (agent_id, x, y)
        return positions

    def deicde(self, X):
        # X : (agent_id, inp_1, inp_2, ...)
        states = self.v[:, 2]
        inp_indices = self.v[:, 3].bool()
        out_indices = self.v[:, 4].bool()

        inp_nodes = self.v[inp_indices]
        
        
        #Set input states to matching agent inputs
        
    def decide(self, X):
        # X : (agent_id, inp_1, inp_2, ...)
        #active_indices = self.v[]

        states = self.v[:, 2]
        inp_indices = self.v[:, 3].bool()
        out_indices = self.v[:, 4].bool()

        inp_indices
        states[inp_indices] = X
        src = self.e[:, 1].long()
        dst = self.e[:, 2].long()
        weight = self.e[:, 3]

        messages = states[src] * weight
        aggregated_messages = torch.zeros_like(states)
        aggregated_messages.scatter_add_(0, dst, messages)

        # Apply activation function to non output nodes (ReLU)
        states[~out_indices] = self.activation(aggregated_messages[~out_indices])
        states[out_indices] = aggregated_messages[out_indices]  # Linear for output nodes
        #apply softmax to outputs
        output_agent_indices = self.v[:,0][out_indices].view(-1, self.out_size)
        output_states = states[out_indices].view(-1, self.out_size)
        actions = torch.argmax(output_states, dim=1)
        
        
        return output_agent_indices[:,0], actions
        
    def apply(self, actions):
        #update agent positions based on actions in a vectorized way
        #0: up, 1: down, 2: left, 3: right
        self.agents[:,1] += ((actions == 3).float() - (actions == 2).float())/10.0
        self.agents[:,2] += ((actions == 0).float() - (actions == 1).float())/10.0

        self.agents[:,1] = torch.clamp(self.agents[:,1], 0, 1)
        self.agents[:,2] = torch.clamp(self.agents[:,2], 0, 1)
        x_coord = self.agents[:,1]
        self.agents[:,3] -= (x_coord - 0.5)**2  #reward for being near center x=0.5
    
    def kill(self):
        #remove agents with low energy
        dead_mask = self.agents[:,3] <= 0

        #get indices from agent_v_mask and agent_e_mask that correspond to dead agents
        dead_agent_indices = torch.nonzero(dead_mask).squeeze()
        if dead_agent_indices.numel() == 0:
            return
        
        dead_v_mask = torch.isin(self.agent_v_mask, dead_agent_indices)
        dead_e_mask = torch.isin(self.agent_e_mask, dead_agent_indices)

        #self.v = self.v[~dead_v_mask]
        #self.states = self.states[~dead_v_mask]
        self.e = self.e[~dead_e_mask]

        #self.bias_mask = self.bias_mask[~dead_v_mask]
        #self.inp_mask = self.inp_mask[~dead_v_mask]
        #self.out_mask = self.out_mask[~dead_v_mask]
        #self.agent_v_mask = self.agent_v_mask[~dead_v_mask]
        self.agent_e_mask = self.agent_e_mask[~dead_e_mask]
        #self.agents = self.agents[~dead_mask]
        self.alive_mask = self.alive_mask & ~dead_mask

    def add(v, e, state):
        pass


v = torch.tensor([0,0,0,0,0,0])
e = torch.tensor([[0,1],
                   [0,2],
                    [1,3],
                    [2,3],
                    [3,4],
                    [4,5]])