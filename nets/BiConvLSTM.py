import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F


from nets.ops.dcn.deform_conv import ModulatedDeformConv

# from ops.dcn.deform_conv import ModulatedDeformConv

class Align_Net(nn.Module):   
    def __init__(self, in_nc, out_nc, base_ks=3, deform_nc=256, deform_ks=3, deform_group=8):
        super(Align_Net, self).__init__()
        self.in_nc = in_nc
        self.deform_ks = deform_ks
        self.deform_group = deform_group
        self.offset_mask = nn.Conv2d(in_nc, deform_group*3*(deform_ks**2), base_ks, padding=base_ks//2)
        self.deform_conv = ModulatedDeformConv(deform_nc, out_nc, deform_ks, padding=deform_ks//2, deformable_groups=deform_group)
    
    def forward(self, x, y):
        off_msk = self.offset_mask(x)
        off = off_msk[:, :self.deform_group*2*(self.deform_ks**2), ...]
        msk = torch.sigmoid(off_msk[:, self.deform_group*2*(self.deform_ks**2):, ...])
    
        fused_feat = F.relu(self.deform_conv(y, off, msk), inplace=True)
        return fused_feat
    
class BiConvLSTMCell(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
        super(BiConvLSTMCell, self).__init__()
        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        # NOTE: This keeps height and width the same
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

        # TODO: we may want this to be different than the conv we use inside each cell
        self.conv_concat = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                                     out_channels=self.hidden_dim,
                                     kernel_size=self.kernel_size,
                                     padding=self.padding,
                                     bias=self.bias)
        
        self.fuse_conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
        
        self.dcn = Align_Net(self.hidden_dim, self.input_dim + self.hidden_dim, 3, self.hidden_dim, 3, 8)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        # print(input_tensor.shape, h_cur.shape, c_cur.shape)  # torch.Size([b, 64, 64, 64]) torch.Size([b, 64, 64, 64]) torch.Size([b, 64, 64, 64])

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
       
        combined = self.fuse_conv(combined)
        combined = self.dcn(combined, h_cur)

        combined_conv = self.conv(combined)  # torch.Size([b, 256, 64, 64])

        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        # print(cc_i.shape, cc_f.shape, cc_o.shape, cc_g.shape)  # torch.Size([b, 64, 64, 64]) torch.Size([b, 64, 64, 64]) torch.Size([b, 64, 64, 64]) torch.Size([b, 64, 64, 64])

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class BiConvLSTM(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
                 bias=True, return_all_layers=False):
        super(BiConvLSTM, self).__init__()
        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(BiConvLSTMCell(input_size=(self.height, self.width),
                                            input_dim=cur_input_dim,
                                            hidden_dim=self.hidden_dim[i],
                                            kernel_size=self.kernel_size[i],
                                            bias=self.bias))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor):  # b, t, c, h, w
        hidden_state = self._init_hidden(batch_size=input_tensor.size(0), cuda=input_tensor.is_cuda)

        layer_output_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            backward_states = []
            forward_states = []
            output_inner = []

            hb, cb = hidden_state[layer_idx]
            # print(hb.shape, cb.shape)  # torch.Size([2, 64, 64, 64]) torch.Size([2, 64, 64, 64])
            for t in range(seq_len):
                hb, cb = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, seq_len-t-1, :, :, :], cur_state=[hb, cb])

                backward_states.append(hb)

            hf, cf = hidden_state[layer_idx]
            for t in range(seq_len):
                hf, cf = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :], cur_state=[hf, cf])
                # print('hf:',hf.shape)
                forward_states.append(hf)

            for t in range(seq_len):
                h = self.cell_list[layer_idx].conv_concat(torch.cat((forward_states[t], backward_states[seq_len - t - 1]), dim=1))
                # print('h',h.shape)
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)

        if not self.return_all_layers:
            return layer_output_list[-1]
        return layer_output_list

    def _init_hidden(self, batch_size, cuda):
        init_states = []
        for i in range(self.num_layers):
            if(cuda):
                init_states.append((Variable(torch.zeros(batch_size, self.hidden_dim[i], self.height, self.width)).cuda(),
                                    Variable(torch.zeros(batch_size, self.hidden_dim[i], self.height, self.width)).cuda()))
            else:
                init_states.append((Variable(torch.zeros(batch_size, self.hidden_dim[i], self.height, self.width)),
                                    Variable(torch.zeros(batch_size, self.hidden_dim[i], self.height, self.width))))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
    
if __name__ == "__main__":
    h = 64
    w = 64
    input_size = [h, w]
    c = 64
    
    biconvlstm  = BiConvLSTM(input_size=(input_size[0], input_size[1]), input_dim=c, 
                             hidden_dim=c, kernel_size=(3, 3), num_layers=2).cuda()

    CNN_seq = []
    x = torch.rand(2, c, h, w).cuda()
    CNN_seq.append(x)
    CNN_seq.append(x)
    CNN_seq.append(x)
    CNN_seq.append(x)
    CNN_seq.append(x)
    CNN_seq_out      = torch.stack(CNN_seq, dim=1)
    # CNN_seq_feature_maps = biconvlstm(CNN_seq_out)


    from thop import profile
    flops, params = profile(biconvlstm, inputs=(CNN_seq_out, ))
    print(flops/(10**9), params/(10**6))  # 86.8220928 1.134768
