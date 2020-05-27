
import torch
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.utils.collections import namedarraytuple
from rlpyt.models.mlp import MlpModel


RnnState = namedarraytuple("RnnState", ["h", "c"])

class CatMlpR2D1Model(torch.nn.Module):
    """Multilayer Perceptron with last layer logit.

    Args:
        input_size (int): number of inputs
        hidden_sizes (list): can be empty list for none (linear model).
        output_size: linear layer at output, or if ``None``, the last hidden size will be the output size and will have nonlinearity applied
        nonlinearity: torch nonlinearity Module (not Functional).
    """

    def __init__(
            self,
            input_size,
            hidden_sizes,  # Can be empty list for none.
            output_size=None,  # if None, last layer has nonlinearity applied.
            nonlinearity=torch.nn.ReLU,  # Module, not Functional.
            n_atoms=5,
            lstm_size=512,
            head_size=512
            ):
        super().__init__()
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]
        hidden_layers = [torch.nn.Linear(n_in, n_out) for n_in, n_out in
            zip([input_size] + hidden_sizes[:-1], hidden_sizes)]
        sequence = list()
        for layer in hidden_layers:
            sequence.extend([layer, nonlinearity()])
        if output_size is not None:
            last_size = hidden_sizes[-1] if hidden_sizes else input_size
            sequence.append(torch.nn.Linear(last_size, output_size))
        self.model = torch.nn.Sequential(*sequence)
        self._output_size = (hidden_sizes[-1] if output_size is None
            else output_size)
        self._n_atoms = n_atoms

        self.lstm = torch.nn.LSTM(output_size + input_size + 2, lstm_size)
        self.head = MlpModel(lstm_size, head_size, output_size=output_size)

    def forward(self, observation, prev_action, prev_reward, init_rnn_state):
        """Compute the model on the input, assuming input shape [B,input_size]."""
        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        # For single valued observations the dim argument to infer_leading_dims should be 0
        lead_dim, T, B, img_shape = infer_leading_dims(observation, 1)

        p = self.model(observation.view(T * B, -1).float())

        lstm_input = torch.cat([
            p.view(T, B, -1),
            prev_action.view(T, B, -1),  # Assumed onehot.
            prev_reward.view(T, B, 1),
            ], dim=2)
        init_rnn_state = None if init_rnn_state is None else tuple(init_rnn_state)
        lstm_out, (hn, cn) = self.lstm(lstm_input, init_rnn_state)

        p = self.head(lstm_out.view(T * B, -1))

        p = restore_leading_dims(p, lead_dim, T, B)
        next_rnn_state = RnnState(h=hn, c=cn)
        return p, next_rnn_state

    @property
    def output_size(self):
        """Retuns the output size of the model."""
        return self._output_size
