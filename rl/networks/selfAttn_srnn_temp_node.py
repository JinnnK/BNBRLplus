import torch.nn.functional as F
from .priors import *
from .srnn_model import *

class SpatialEdgeSelfAttn(nn.Module):
    """
    Class for the human-human attention,
    uses a multi-head self attention proposed by https://arxiv.org/abs/1706.03762
    """
    def __init__(self, args):
        super(SpatialEdgeSelfAttn, self).__init__()
        self.args = args

        # Store required sizes
        # todo: hard-coded for now
        # with human displacement: + 2
        # pred 4 steps + disp: 12
        # pred 4 steps + no disp: 10
        # pred 5 steps + no disp: 12
        # pred 5 steps + no disp + probR: 17
        # Gaussian pred 5 steps + no disp: 27
        # pred 8 steps + no disp: 18
        if args.env_name in ['CrowdSimPred-v0', 'CrowdSimPredRealGST-v0', 'rosTurtlebot2iEnv-v0']:
            self.input_size = 12
        elif args.env_name == 'CrowdSimVarNum-v0':
            self.input_size = 2 # 4
        else:
            raise NotImplementedError
        self.num_attn_heads=8
        self.attn_size=512


        # Linear layer to embed input
        self.embedding_layer = nn.Sequential(nn.Linear(self.input_size, 128), nn.ReLU(),
                                             nn.Linear(128, self.attn_size), nn.ReLU()
                                             )

        self.q_linear = nn.Linear(self.attn_size, self.attn_size)
        self.v_linear = nn.Linear(self.attn_size, self.attn_size)
        self.k_linear = nn.Linear(self.attn_size, self.attn_size)

        # multi-head self attention
        self.multihead_attn=torch.nn.MultiheadAttention(self.attn_size, self.num_attn_heads)


    # Given a list of sequence lengths, create a mask to indicate which indices are padded
    # e.x. Input: [3, 1, 4], max_human_num = 5
    # Output: [[1, 1, 1, 0, 0], [1, 0, 0, 0, 0], [1, 1, 1, 1, 0]]
    def create_attn_mask(self, each_seq_len, seq_len, nenv, max_human_num):
        # mask with value of False means padding and should be ignored by attention
        # why +1: use a sentinel in the end to handle the case when each_seq_len = 18
        if self.args.no_cuda:
            mask = torch.zeros(seq_len * nenv, max_human_num + 1).cpu()
        else:
            mask = torch.zeros(seq_len*nenv, max_human_num+1).cuda()
        mask[torch.arange(seq_len*nenv), each_seq_len.long()] = 1.
        mask = torch.logical_not(mask.cumsum(dim=1))
        # remove the sentinel
        mask = mask[:, :-1].unsqueeze(-2) # seq_len*nenv, 1, max_human_num
        return mask


    def forward(self, inp, each_seq_len):
        '''
        Forward pass for the model
        params:
        inp : input edge features
        each_seq_len:
        if self.args.sort_humans is True, the true length of the sequence. Should be the number of detected humans
        else, it is the mask itself
        '''
        # print("inp :", inp) # spatial edges
        # inp is padded sequence [seq_len, nenv, max_human_num, 2]
        seq_len, nenv, max_human_num, _ = inp.size()
        # print("seq_len :", seq_len)
        # print("inp :", inp.shape)
        if self.args.sort_humans:
            attn_mask = self.create_attn_mask(each_seq_len, seq_len, nenv, max_human_num)  # [seq_len*nenv, 1, max_human_num]
            attn_mask = attn_mask.squeeze(1)  # if we use pytorch builtin function
        else:
            # combine the first two dimensions
            attn_mask = each_seq_len.reshape(seq_len*nenv, max_human_num)
        # print("attn_mask :", attn_mask)

        input_emb=self.embedding_layer(inp).view(seq_len*nenv, max_human_num, -1)
        # print("before : ", input_emb.shape) # [nenv, max_human_num, 512]
        input_emb=torch.transpose(input_emb, dim0=0, dim1=1) # if we use pytorch builtin function, v1.7.0 has no batch first option
        # print("self.q_linear.weight :", self.q_linear.weight)
        # print("self.q_linear.weight.shape :", self.q_linear.weight.shape)
        # print("input_emb :", input_emb.shape)
        # print("input_emb :", input_emb)
        q=self.q_linear(input_emb)
        k=self.k_linear(input_emb)
        v=self.v_linear(input_emb)
        # print("q :", q.shape)
        # print("q :", q)

        # z=self.multihead_attn(q, k, v, mask=attn_mask)
        z,_=self.multihead_attn(q, k, v, key_padding_mask=torch.logical_not(attn_mask)) # if we use pytorch builtin function
        z=torch.transpose(z, dim0=0, dim1=1) # if we use pytorch builtin function
        # print("key_padding_mask :", torch.logical_not(attn_mask))
        # print("output :", z.shape) # [nenv, max_human_num, 512]
        return z

class EdgeAttention_M(nn.Module):
    '''
    Class for the robot-human attention module
    '''
    def __init__(self, args):
        '''
        Initializer function
        params:
        args : Training arguments
        infer : Training or test time (True at test time)
        '''
        super(EdgeAttention_M, self).__init__()

        self.args = args

        # Store required sizes
        self.human_human_edge_rnn_size = args.human_human_edge_rnn_size
        self.human_node_rnn_size = args.human_node_rnn_size
        self.attention_size = args.attention_size



        # Linear layer to embed temporal edgeRNN hidden state
        self.temporal_edge_layer=nn.ModuleList()
        self.spatial_edge_layer=nn.ModuleList()

        self.temporal_edge_layer.append(nn.Linear(self.human_human_edge_rnn_size, self.attention_size))

        # Linear layer to embed spatial edgeRNN hidden states
        self.spatial_edge_layer.append(nn.Linear(self.human_human_edge_rnn_size, self.attention_size))



        # number of agents who have spatial edges (complete graph: all 6 agents; incomplete graph: only the robot)
        self.agent_num = 1
        self.num_attention_head = 1

    def create_attn_mask(self, each_seq_len, seq_len, nenv, max_human_num):
        # mask with value of False means padding and should be ignored by attention
        # why +1: use a sentinel in the end to handle the case when each_seq_len = 18
        if self.args.no_cuda:
            mask = torch.zeros(seq_len * nenv, max_human_num + 1).cpu()
        else:
            mask = torch.zeros(seq_len * nenv, max_human_num + 1).cuda()
        mask[torch.arange(seq_len * nenv), each_seq_len.long()] = 1.
        mask = torch.logical_not(mask.cumsum(dim=1))
        # remove the sentinel
        mask = mask[:, :-1].unsqueeze(-2)  # seq_len*nenv, 1, max_human_num
        return mask

    def att_func(self, temporal_embed, spatial_embed, h_spatials, attn_mask=None):
        seq_len, nenv, num_edges, h_size = h_spatials.size()  # [1, 1, 4, 256]
        attn = temporal_embed * spatial_embed   # [1, 1, 5, 64] = [1, 1, 5, 64] * [1, 1, 5, 64] when max_human_num = 5
        attn = torch.sum(attn, dim=3)   # [1, 1, 5]

        # Variable length
        temperature = num_edges / np.sqrt(self.attention_size)
        attn = torch.mul(attn, temperature)

        # if we don't want to mask invalid humans, attn_mask is None and no mask will be applied
        # else apply attn masks
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 0, -1e9)

        # Softmax
        attn = attn.view(seq_len, nenv, self.agent_num, self.human_num)
        attn = torch.nn.functional.softmax(attn, dim=-1)
        # print(attn[0, 0, 0].cpu().numpy())

        # Compute weighted value
        # weighted_value = torch.mv(torch.t(h_spatials), attn)

        # reshape h_spatials and attn
        # shape[0] = seq_len, shape[1] = num of spatial edges (6*5 = 30), shape[2] = 256
        h_spatials = h_spatials.view(seq_len, nenv, self.agent_num, self.human_num, h_size)
        # print(h_spatials.shape)
        h_spatials = h_spatials.view(seq_len * nenv * self.agent_num, self.human_num, h_size).permute(0, 2,
                                                                                         1)  # [seq_len*nenv*6, 5, 256] -> [seq_len*nenv*6, 256, 5]
        # print(h_spatials.shape)

        attn = attn.view(seq_len * nenv * self.agent_num, self.human_num).unsqueeze(-1)  # [seq_len*nenv*6, 5, 1]
        # print(attn.shape)
        weighted_value = torch.bmm(h_spatials, attn)  # [seq_len*nenv*6, 256, 1]
        # print(weighted_value.shape)

        # reshape back
        weighted_value = weighted_value.squeeze(-1).view(seq_len, nenv, self.agent_num, h_size)  # [seq_len, 12, 6 or 1, 256]
        return weighted_value, attn



    # h_temporal: [seq_len, nenv, 1, 256]
    # h_spatials: [seq_len, nenv, 5, 256]
    def forward(self, h_temporal, h_spatials, each_seq_len):
        '''
        Forward pass for the model
        params:
        h_temporal : Hidden state of the temporal edgeRNN
        h_spatials : Hidden states of all spatial edgeRNNs connected to the node.
        each_seq_len:
            if self.args.sort_humans is True, the true length of the sequence. Should be the number of detected humans
            else, it is the mask itself
        '''
        seq_len, nenv, max_human_num, _ = h_spatials.size() # [1, 1, 4, 256]
        # find the number of humans by the size of spatial edgeRNN hidden state
        self.human_num = max_human_num // self.agent_num

        weighted_value_list, attn_list=[],[]
        for i in range(self.num_attention_head):

            # Embed the temporal edgeRNN hidden state
            temporal_embed = self.temporal_edge_layer[i](h_temporal)
            # temporal_embed = temporal_embed.squeeze(0)

            # Embed the spatial edgeRNN hidden states
            spatial_embed = self.spatial_edge_layer[i](h_spatials)

            # Dot based attention
            temporal_embed = temporal_embed.repeat_interleave(self.human_num, dim=2)

            if self.args.sort_humans:
                attn_mask = self.create_attn_mask(each_seq_len, seq_len, nenv, max_human_num)  # [seq_len*nenv, 1, max_human_num]
                attn_mask = attn_mask.squeeze(-2).view(seq_len, nenv, max_human_num)
            else:
                attn_mask = each_seq_len
            # print("temporal_embed.shape :", temporal_embed.shape)
            # print("spatial_embed.shape :", spatial_embed.shape)
            weighted_value,attn=self.att_func(temporal_embed, spatial_embed, h_spatials, attn_mask=attn_mask)
            # print("weighted_value.shape :", weighted_value.shape)
            # print("attn.shape :", attn.shape)
            weighted_value_list.append(weighted_value)
            attn_list.append(attn)

        if self.num_attention_head > 1:
            return self.final_attn_linear(torch.cat(weighted_value_list, dim=-1)), attn_list
        else:
            return weighted_value_list[0], attn_list[0]

class BayesLinear_Normalq(nn.Module):
    """Linear Layer where weights are sampled from a fully factorised Normal with learnable parameters. The likelihood
     of the weight samples under the prior and the approximate posterior are returned with each forward pass in order
     to estimate the KL term in the ELBO.
    """
    def __init__(self, n_in, n_out, prior_class):
        super(BayesLinear_Normalq, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.prior = prior_class

        # Learnable parameters -> Initialisation is set empirically.
        self.W_mu = nn.Parameter(torch.Tensor(self.n_in, self.n_out).uniform_(-0.3, 0.3))
        self.W_p = nn.Parameter(torch.Tensor(self.n_in, self.n_out).uniform_(-2.0, -1.0))

        self.b_mu = nn.Parameter(torch.Tensor(self.n_out).uniform_(-0.3, 0.3))
        self.b_p = nn.Parameter(torch.Tensor(self.n_out).uniform_(-2.0, -1.0))

        self.lpw = 0
        self.lqw = 0

    def forward(self, X, sample=False, masks = None):

        if not self.training and not sample:  # When training return MLE of w for quick validation
            output = torch.mm(X, self.W_mu) + self.b_mu.expand(X.size()[0], self.n_out)
            return output, 0, 0

        else:
            # Tensor.new()  Constructs a new tensor of the same data type as self tensor.
            # the same random sample is used for every element in the minibatch
            eps_W = Variable(self.W_mu.data.new(self.W_mu.size()).normal_())
            eps_b = Variable(self.b_mu.data.new(self.b_mu.size()).normal_())

            # sample parameters
            std_w = 1e-6 + F.softplus(self.W_p, beta=1, threshold=20)
            std_b = 1e-6 + F.softplus(self.b_p, beta=1, threshold=20)

            W = self.W_mu + 1 * std_w * eps_W
            W = W / 5
            b = self.b_mu + 1 * std_b * eps_b
            b = b / 5

            X_inp = X.clone()

            if masks is not None:
                # make inputs zero for not detected humans.
                X_inp[X_inp > 14.0] = 0.0
                # print("X_inp :", X_inp)

            if X_inp.ndim == 3:
                output = torch.einsum("ijk,km->ijm", X_inp, W) + b.unsqueeze(0).expand(X_inp.shape[0], X_inp.shape[1], -1)
            else:
                output = torch.einsum("ijkl,lm->ijkm", X_inp, W) + b.unsqueeze(0).expand(X_inp.shape[0], X_inp.shape[1], X_inp.shape[2], -1)
            lqw = isotropic_gauss_loglike(W, self.W_mu, std_w) + isotropic_gauss_loglike(b, self.b_mu, std_b)
            lpw = self.prior.loglike(W) + self.prior.loglike(b)

            return output, lqw, lpw

class Bayes_linear_2L(nn.Module):
    """2 hidden layer Bayes By Backprop (VI) Network"""
    def __init__(self, args, inp_size):
        super(Bayes_linear_2L, self).__init__()

        self.prior_instance = isotropic_gauss_prior(mu=0, sigma=0.1)
        # prior_instance = spike_slab_2GMM(mu1=0, mu2=0, sigma1=0.135, sigma2=0.001, pi=0.5)
        # prior_instance = isotropic_gauss_loglike(mu=0, sigma=0.1)
        # prior_instance = laplace_prior(mu=0, sigma=0.1)
        # prior_instance = chi_squared_prior(df=0.1)
        # prior_instance = student_t_prior(df=0.1)
        # prior_instance = beta_prior(alpha=0.1, beta=0.1)

        self.input_dim = inp_size
        self.n_hid = args.hidden_dim
        self.output_dim = args.output_dim

        self.bfc1 = BayesLinear_Normalq(self.input_dim, self.n_hid, self.prior_instance)
        self.bfc2 = BayesLinear_Normalq(self.n_hid, self.n_hid, self.prior_instance)
        self.bfc3 = BayesLinear_Normalq(self.n_hid, self.output_dim, self.prior_instance)

        # choose your non linearity
        # self.act = nn.Tanh()
        # self.act = nn.Sigmoid()
        self.act = nn.ReLU()
        # self.act = nn.ELU(inplace=True)
        # self.act = nn.SELU(inplace=True)

    def forward(self, x, sample=False):
        tlqw = 0
        tlpw = 0

        #TODO : must adapt this code to multi env

        # print("x :", x) # [1, 1, 1, 48]
        # -----------------
        x, lqw, lpw = self.bfc1(x, sample, masks = True)
        # print("x :", x.shape) # [1, 1, 1, 64]
        tlqw = tlqw + lqw
        tlpw = tlpw + lpw
        # -----------------
        x = self.act(x)
        # print("x :", x.shape) # [1, 1, 1, 64]
        # -----------------
        x, lqw, lpw = self.bfc2(x, sample)
        tlqw = tlqw + lqw
        tlpw = tlpw + lpw
        # -----------------
        x = self.act(x)
        # -----------------
        y, lqw, lpw = self.bfc3(x, sample)
        tlqw = tlqw + lqw
        tlpw = tlpw + lpw
        y = self.act(y)
        # print("tlqw :", tlqw.shape) # uncertainty
        # print("tlpw :", tlpw.shape) # uncertainty
        # print("y : ", y.shape) # [1, 1, 1, 128]

        return y, tlqw, tlpw

    def sample_predict(self, x, Nsamples):
        """Used for estimating the data's likelihood by approximately marginalising the weights with MC"""
        # Just copies type from x, initializes new vector
        predictions = x.data.new(Nsamples, x.shape[0], self.output_dim)
        tlqw_vec = np.zeros(Nsamples)
        tlpw_vec = np.zeros(Nsamples)

        for i in range(Nsamples):
            y, tlqw, tlpw = self.forward(x, sample=True)
            predictions[i] = y
            tlqw_vec[i] = tlqw
            tlpw_vec[i] = tlpw

        return predictions, tlqw_vec, tlpw_vec

class Bayes_SpatialEdgeSelfAttn(nn.Module):
    def __init__(self, args):
        super(Bayes_SpatialEdgeSelfAttn, self).__init__()
        self.args = args

        self.num_attn_heads=8
        self.attn_size=512

        self.input_dim = 12
        self.n_hid = args.hidden_dim

        self.act = nn.ReLU()
        self.prior_instance = isotropic_gauss_prior(mu=0, sigma=0.1)

        self.bfc1 = BayesLinear_Normalq(self.input_dim, 128, self.prior_instance)
        self.bfc2 = BayesLinear_Normalq(128, self.attn_size, self.prior_instance)

        self.bfc_q = BayesLinear_Normalq(self.attn_size, self.attn_size, self.prior_instance)
        self.bfc_v = BayesLinear_Normalq(self.attn_size, self.attn_size, self.prior_instance)
        self.bfc_k = BayesLinear_Normalq(self.attn_size, self.attn_size, self.prior_instance)

        # multi-head self attention
        self.bfc_multihead_attn=torch.nn.MultiheadAttention(self.attn_size, self.num_attn_heads)

    # Given a list of sequence lengths, create a mask to indicate which indices are padded
    # e.x. Input: [3, 1, 4], max_human_num = 5
    # Output: [[1, 1, 1, 0, 0], [1, 0, 0, 0, 0], [1, 1, 1, 1, 0]]
    def create_attn_mask(self, each_seq_len, seq_len, nenv, max_human_num):
        # mask with value of False means padding and should be ignored by attention
        # why +1: use a sentinel in the end to handle the case when each_seq_len = 18
        if self.args.no_cuda:
            mask = torch.zeros(seq_len * nenv, max_human_num + 1).cpu()
        else:
            mask = torch.zeros(seq_len*nenv, max_human_num+1).cuda()
        mask[torch.arange(seq_len*nenv), each_seq_len.long()] = 1.
        mask = torch.logical_not(mask.cumsum(dim=1))
        # print("bnn mask :", mask.shape)
        # remove the sentinel
        mask = mask[:, :-1].unsqueeze(-2) # seq_len*nenv, 1, max_human_num
        # print("bnn mask :", mask.shape)
        return mask

    def forward(self, inp, each_seq_len, sample = False):
        '''
        Forward pass for the model
        params:
        inp : input edge features
        each_seq_len:
        if self.args.sort_humans is True, the true length of the sequence. Should be the number of detected humans
        else, it is the mask itself
        '''

        # print("inp :", inp) # spatial edges
        # inp is padded sequence [seq_len, nenv, max_human_num, 2]
        seq_len, nenv, max_human_num, _ = inp.size()
        # print("seq_len :", seq_len)
        # print("inp :", inp)
        if self.args.sort_humans:
            attn_mask = self.create_attn_mask(each_seq_len, seq_len, nenv, max_human_num)  # [seq_len*nenv, 1, max_human_num]
            attn_mask = attn_mask.squeeze(1)  # if we use pytorch builtin function
        else:
            # combine the first two dimensions
            attn_mask = each_seq_len.reshape(seq_len*nenv, max_human_num)
            # print("attn_mask :", attn_mask)

        input_emb, _, __ =self.bfc1(inp, sample)
        input_emb=self.act(input_emb)
        input_emb, _, __ =self.bfc2(input_emb, sample)
        input_emb=self.act(input_emb).view(seq_len*nenv, max_human_num, -1)

        input_emb=torch.transpose(input_emb, dim0=0, dim1=1) # if we use pytorch builtin function, v1.7.0 has no batch first option

        q, _, __ = self.bfc_q(input_emb)
        k, _, __ = self.bfc_k(input_emb)
        v, _, __ = self.bfc_v(input_emb)

        # z=self.multihead_attn(q, k, v, mask=attn_mask)
        z, _ = self.bfc_multihead_attn(q, k, v, key_padding_mask=torch.logical_not(attn_mask)) # if we use pytorch builtin function
        z = torch.transpose(z, dim0=0, dim1=1) # if we use pytorch builtin function
        # print("key_padding_mask :", torch.logical_not(attn_mask))
        # print("output :", z.shape) # [nenv, max_human_num, 512]
        return z

class EndRNN(RNNBase):
    '''
    Class for the GRU
    '''
    def __init__(self, args):
        '''
        Initializer function
        params:
        args : Training arguments
        infer : Training or test time (True at test time)
        '''
        super(EndRNN, self).__init__(args, edge=False)

        self.args = args

        # Store required sizes
        self.rnn_size = args.human_node_rnn_size
        self.output_size = args.human_node_output_size
        self.embedding_size = args.human_node_embedding_size
        self.input_size = args.human_node_input_size
        self.edge_rnn_size = args.human_human_edge_rnn_size

        # Linear layer to embed input
        self.encoder_linear = nn.Linear(256, self.embedding_size)
        if self.args.use_bnn:
            if self.args.bnn_policy == 'BNBRL+' or self.args.bnn_policy == 'BNDNN':
                self.bnn_linear = nn.Linear(256, self.embedding_size)
            elif self.args.bnn_policy == 'BNBRL':
                self.bnn_linear = nn.Linear(128, self.embedding_size)
            else:
                raise NotImplementedError

        # ReLU and Dropout layers
        self.relu = nn.ReLU()

        # Linear layer to embed attention module output
        self.edge_attention_embed = nn.Linear(self.edge_rnn_size, self.embedding_size)


        # Output linear layer
        self.output_linear = nn.Linear(self.rnn_size, self.output_size)



    def forward(self, robot_s, h_spatial_other, h, bnn_h, masks):
        '''
        Forward pass for the model
        params:
        pos : input position
        h_temporal : hidden state of the temporal edgeRNN corresponding to this node
        h_spatial_other : output of the attention module
        h : hidden state of the current nodeRNN
        c : cell state of the current nodeRNN
        '''
        # print("robot_s :", robot_s.shape) # [1, 1, 1, 256]
        # print("h_spatial_other :", h_spatial_other.shape) # [1, 1, 1, 256]
        # print("h :", h.shape) # [1, 1, 1, 128]
        # print("bnn_h :", bnn_h.shape) # [1, 1, 1, 128]
        # print("masks :", masks.shape) # [1, 1, 1]

        # Encode the input position
        encoded_input = self.encoder_linear(robot_s)    # 256 -> 64
        encoded_input = self.relu(encoded_input)    # f(w_s) 64 -> 64
        # print("encoded_input :", encoded_input.shape) # [1, 1, 1, 64]

        h_edges_embedded = self.relu(self.edge_attention_embed(h_spatial_other))    # 256 -> 64
        # print("h_edges_embedded :", h_edges_embedded.shape) # [1, 1, 1, 64]
        # print("bnn_h :", bnn_h.shape) # [1, 1, 1, 64]

        bnn_inputs = self.relu(self.bnn_linear(bnn_h))    # 128 -> 64

        concat_encoded = torch.cat((encoded_input, h_edges_embedded, bnn_inputs), -1)   # 64 + 64 + 64 = 192
        # print("concat_encoded :", concat_encoded.shape) # [1, 1, 1, 192]

        # update hidden state
        x, h_new = self._forward_gru(concat_encoded, h, masks)
        # print("masks :", masks)
        # print(masks.shape)

        outputs = self.output_linear(x)
        # print(self.output_linear.weight)


        return outputs, h_new

class selfAttn_merge_SRNN(nn.Module):
    """
    Class for the proposed network
    """
    def __init__(self, obs_space_dict, args, infer=False):
        """
        Initializer function
        params:
        args : Training arguments
        infer : Training or test time (True at test time)
        """
        super(selfAttn_merge_SRNN, self).__init__()
        self.infer = infer
        self.is_recurrent = True
        self.args=args

        self.human_num = obs_space_dict['spatial_edges'].shape[0]
        if self.args.use_bnn:
            self.belief_masks = obs_space_dict['belief_masks']
            self.belief_inp_size = obs_space_dict['belief_edges'].shape[0] * obs_space_dict['belief_edges'].shape[1]

        self.seq_length = args.seq_length
        self.nenv = args.num_processes
        self.nminibatch = args.num_mini_batch

        # Store required sizes
        self.human_node_rnn_size = args.human_node_rnn_size
        self.human_human_edge_rnn_size = args.human_human_edge_rnn_size
        self.output_size = args.human_node_output_size

        # Initialize the Node and Edge RNNs
        self.humanNodeRNN = EndRNN(args)

        # Initialize attention module
        self.attn = EdgeAttention_M(args)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        num_inputs = hidden_size = self.output_size

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())


        self.critic_linear = init_(nn.Linear(hidden_size, 1))
        robot_size = 9
        self.robot_linear = nn.Sequential(init_(nn.Linear(robot_size, 256)), nn.ReLU()) # todo: check dim
        self.human_node_final_linear=init_(nn.Linear(self.output_size,2))

        if self.args.use_self_attn:
            self.spatial_attn = SpatialEdgeSelfAttn(args)
            self.spatial_linear = nn.Sequential(init_(nn.Linear(512, 256)), nn.ReLU())
        else:
            self.spatial_linear = nn.Sequential(init_(nn.Linear(obs_space_dict['spatial_edges'].shape[1], 128)), nn.ReLU(),
                                                init_(nn.Linear(128, 256)), nn.ReLU())

        # Initialize BNN modules
        if self.args.use_bnn:
            if self.args.bnn_policy == 'BNDNN':
                self.bnn = SpatialEdgeSelfAttn(args)
                self.bnn_attn = EdgeAttention_M(args)
                self.bnn_spatial_linear = nn.Sequential(init_(nn.Linear(512, 256)), nn.ReLU())
            elif self.args.bnn_policy == 'BNBRL':
                self.bnn = Bayes_linear_2L(args, self.belief_inp_size)
            elif self.args.bnn_policy == 'BNBRL+':
                self.bnn = Bayes_SpatialEdgeSelfAttn(args)
                self.bnn_attn = EdgeAttention_M(args)
                self.bnn_spatial_linear = nn.Sequential(init_(nn.Linear(512, 256)), nn.ReLU())
            else:
                raise NotImplementedError

        self.temporal_edges = [0]
        self.spatial_edges = np.arange(1, self.human_num+1)

        dummy_human_mask = [0] * self.human_num
        dummy_human_mask[0] = 1
        if self.args.no_cuda:
            self.dummy_human_mask = Variable(torch.Tensor([dummy_human_mask]).cpu())
        else:
            self.dummy_human_mask = Variable(torch.Tensor([dummy_human_mask]).cuda())



    def forward(self, inputs, rnn_hxs, masks, infer=False):
        if infer:
            # Test/rollout time
            seq_length = 1
            nenv = self.nenv

        else:
            # Training time
            seq_length = self.seq_length
            nenv = self.nenv // self.nminibatch

        robot_node = reshapeT(inputs['robot_node'], seq_length, nenv)
        temporal_edges = reshapeT(inputs['temporal_edges'], seq_length, nenv)
        spatial_edges = reshapeT(inputs['spatial_edges'], seq_length, nenv)
        if self.args.use_bnn:
            belief_edges = reshapeT(inputs['belief_edges'], seq_length, nenv)
            belief_masks = reshapeT(inputs['belief_masks'], seq_length, nenv).float()
            belief_masks_num = belief_masks.sum(dim=2)
            belief_masks_num = belief_masks_num.view(belief_masks_num.shape[0] * belief_masks_num.shape[1]).cpu().int()
            belief_masks_num[belief_masks_num == 0] = 1

        # to prevent errors in old models that does not have sort_humans argument
        if not hasattr(self.args, 'sort_humans'):
            self.args.sort_humans = True
        if self.args.sort_humans:
            detected_human_num = inputs['detected_human_num'].squeeze(-1).cpu().int()
        else:
            human_masks = reshapeT(inputs['visible_masks'], seq_length, nenv).float() # [seq_len, nenv, max_human_num]
            # if no human is detected (human_masks are all False, set the first human to True)
            human_masks[human_masks.sum(dim=-1)==0] = self.dummy_human_mask


        hidden_states_node_RNNs = reshapeT(rnn_hxs['human_node_rnn'], 1, nenv)
        masks = reshapeT(masks, seq_length, nenv)


        if self.args.no_cuda:
            all_hidden_states_edge_RNNs = Variable(
                torch.zeros(1, nenv, 1+self.human_num, rnn_hxs['human_human_edge_rnn'].size()[-1]).cpu())
        else:
            all_hidden_states_edge_RNNs = Variable(
                torch.zeros(1, nenv, 1+self.human_num, rnn_hxs['human_human_edge_rnn'].size()[-1]).cuda())

        robot_states = torch.cat((temporal_edges, robot_node), dim=-1)
        robot_states = self.robot_linear(robot_states)

        # attention modules
        if self.args.sort_humans:
            # human-human attention
            if self.args.use_self_attn:
                # forward of SpatialEdgeSelfAttn
                spatial_attn_out=self.spatial_attn(spatial_edges, detected_human_num).view(seq_length, nenv, self.human_num, -1)
                # spatial_attn_out = [nenv, max_human_num, 512]
            else:
                spatial_attn_out = spatial_edges
            # change the dimension from 512 to 256
            output_spatial = self.spatial_linear(spatial_attn_out)

            # robot-human attention
            # print("robot_states :", robot_states.shape) # [1, 1, 1, 256]
            # print("output_spatial :", output_spatial.shape) # [seq_len, nenv, max_human_num, 256]
            # print("detected_human_num :", detected_human_num.shape) # [1]
            hidden_attn_weighted, _ = self.attn(robot_states, output_spatial, detected_human_num)
            # print("hidden_attn_weighted :", hidden_attn_weighted.shape) # [1, 1, 1, 256]
        else:
            # human-human attention
            if self.args.use_self_attn:
                spatial_attn_out = self.spatial_attn(spatial_edges, human_masks).view(seq_length, nenv, self.human_num, -1)
            else:
                spatial_attn_out = spatial_edges
            output_spatial = self.spatial_linear(spatial_attn_out)

            # robot-human attention
            hidden_attn_weighted, _ = self.attn(robot_states, output_spatial, human_masks)

        # bnn modules
        if self.args.use_bnn:
            if self.args.bnn_policy == 'BNBRL':
                belief_edges = belief_edges.view(seq_length, nenv, 1, self.human_num * 12)
                hidden_bnn_attn_weighted, _, __ = self.bnn(belief_edges)
            elif self.args.bnn_policy == 'BNBRL+' or self.args.bnn_policy == 'BNDNN':
                bnn_outputs = self.bnn(belief_edges, belief_masks_num).view(seq_length, nenv, self.human_num, -1)
                bnn_outputs_ = self.bnn_spatial_linear(bnn_outputs)
                hidden_bnn_attn_weighted, _ = self.bnn_attn(robot_states, bnn_outputs_, belief_masks_num)
            else:
                raise NotImplementedError

        # Do a forward pass through GRU2
        # print("robot_states :", robot_states)
        # print("hidden_attn_weighted :", hidden_attn_weighted)
        # print("bnn_outputs :", bnn_outputs)
        if self.args.use_bnn:
            outputs, h_nodes \
                = self.humanNodeRNN(robot_states, hidden_attn_weighted, hidden_states_node_RNNs, hidden_bnn_attn_weighted, masks)
        else:
            outputs, h_nodes \
                = self.humanNodeRNN(robot_states, hidden_attn_weighted, hidden_states_node_RNNs, masks)


        # Update the hidden and cell states
        all_hidden_states_node_RNNs = h_nodes
        outputs_return = outputs

        rnn_hxs['human_node_rnn'] = all_hidden_states_node_RNNs
        rnn_hxs['human_human_edge_rnn'] = all_hidden_states_edge_RNNs # is this necessary? all of components are zero.
        # print("all_hidden_states_node_RNNs :", all_hidden_states_node_RNNs)
        # print("all_hidden_states_edge_RNNs :", all_hidden_states_edge_RNNs)


        # x is the output and will be sent to actor and critic
        x = outputs_return[:, :, 0, :]

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        for key in rnn_hxs:
            rnn_hxs[key] = rnn_hxs[key].squeeze(0)

        if infer:
            return self.critic_linear(hidden_critic).squeeze(0), hidden_actor.squeeze(0), rnn_hxs
        else:
            return self.critic_linear(hidden_critic).view(-1, 1), hidden_actor.view(-1, self.output_size), rnn_hxs

def reshapeT(T, seq_length, nenv):
    shape = T.size()[1:]
    return T.unsqueeze(0).reshape((seq_length, nenv, *shape))
