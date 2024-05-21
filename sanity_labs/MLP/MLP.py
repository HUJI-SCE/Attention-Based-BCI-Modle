import torch.nn as nn


class PolyMod(nn.Module):
    def __init__(self, topology):
        """
        This function sets up the netless_data, via the given topology.

        Activation:
        The activation function for each layer is RELU (experimental: counter-measure for vanishing gradient).
        The activation function for the output layer is sigmoid (output must be boolean).

        :topology: list of lengths of the layers (number of neurons in each layer).
        """

        super(PolyMod, self).__init__()

        # unpack topology
        pep_chain_length, HL1_length, HL2_length, HL3_length = topology

        relu_act = nn.ReLU()
        sig = nn.Sigmoid()

        forward_seq_list = [nn.Linear(pep_chain_length, HL1_length, bias=True), relu_act,
                            nn.Linear(HL1_length, HL2_length, bias=True), relu_act,
                            nn.Dropout(p=0.2),
                            nn.Linear(HL2_length, HL3_length, bias=True), relu_act,
                            nn.Dropout(p=0.2),
                            nn.Linear(HL3_length, 1, bias=True), sig]

        self.forward_seq = nn.Sequential(*forward_seq_list)

    def forward(self, x):
        """
        This method computes a forward pass through the layers.
        """
        binary_output = self.forward_seq(x)

        return binary_output
