import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class Prenet(nn.Module):
  """Prenet is a multi-layer fully-connected network with ReLU activations.
  During training and testing (i.e., feature extraction), each input frame is
  passed into the Prenet, and the Prenet output is then fed to the RNN. If
  Prenet configuration is None, the input frames will be directly fed to the
  RNN without any transformation.
  """

  def __init__(self, input_size, num_layers, hidden_size, dropout):
    super(Prenet, self).__init__()
    input_sizes = [input_size] + [hidden_size] * (num_layers - 1)
    output_sizes = [hidden_size] * num_layers

    self.layers = nn.ModuleList(
      [nn.Linear(in_features=in_size, out_features=out_size)
      for (in_size, out_size) in zip(input_sizes, output_sizes)])

    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(dropout)


  def forward(self, inputs):
    # inputs: (batch_size, seq_len, mel_dim)
    for layer in self.layers:
      inputs = self.dropout(self.relu(layer(inputs)))

    return inputs
    # inputs: (batch_size, seq_len, out_dim)


class Postnet(nn.Module):
  """Postnet is a simple linear layer for predicting the target frames given
  the RNN context during training. We don't need the Postnet for feature
  extraction.
  """

  def __init__(self, input_size, output_size=40, n_component=None):
    super(Postnet, self).__init__()
    self.n_component = n_component
    if self.n_component is None:
        self.layer = nn.Conv1d(in_channels=input_size, out_channels=output_size,
                           kernel_size=1, stride=1)
    else:
        self.layer = nn.Conv1d(in_channels=input_size, out_channels=n_component*output_size,
                           kernel_size=1, stride=1)
    #else:
    #    self.layers = nn.ModuleList([nn.Conv1d(in_channels=input_size, out_channels=output_size,
    #                       kernel_size=1, stride=1) for i in range(n_component)])   
    #self.layer = nn.Linear(in_features=input_size, out_features=output_size, bias=False)

  def forward(self, inputs):
    # inputs: (batch_size, seq_len, hidden_size)
    inputs = torch.transpose(inputs, 1, 2)

    #if self.n_component is None:
    return torch.transpose(self.layer(inputs), 1, 2)
    # (batch_size, seq_len, output_size) -- back to the original shape



class APCModel(nn.Module):
  """This class defines Autoregressive Predictive Coding (APC), a model that
  learns to extract general speech features from unlabeled speech data. These
  features are shown to contain rich speaker and phone information, and are
  useful for a wide range of downstream tasks such as speaker verification
  and phone classification.

  An APC model consists of a Prenet (optional), a multi-layer GRU network,
  and a Postnet. For each time step during training, the Prenet transforms
  the input frame into a latent representation, which is then consumed by
  the GRU network for generating internal representations across the layers.
  Finally, the Postnet takes the output of the last GRU layer and attempts to
  predict the target frame.

  After training, to extract features from the data of your interest, which
  do not have to be i.i.d. with the training data, simply feed-forward the
  the data through the APC model, and take the the internal representations
  (i.e., the GRU hidden states) as the extracted features and use them in
  your tasks.
  """

  def __init__(self, mel_dim, prenet_config, rnn_config, n_component=None):
    super(APCModel, self).__init__()
    self.mel_dim = mel_dim
    self.n_component = n_component

    if prenet_config is not None:
      # Make sure the dimensionalities are correct
      assert prenet_config.input_size == mel_dim
      assert prenet_config.hidden_size == rnn_config.input_size
      assert rnn_config.input_size == rnn_config.hidden_size
      self.prenet = Prenet(
        input_size=prenet_config.input_size,
        num_layers=prenet_config.num_layers,
        hidden_size=prenet_config.hidden_size,
        dropout=prenet_config.dropout)
    else:
      assert rnn_config.input_size == mel_dim
      self.prenet = None

    in_sizes = ([rnn_config.input_size] +
                [rnn_config.hidden_size] * (rnn_config.num_layers - 1))
    out_sizes = [rnn_config.hidden_size] * rnn_config.num_layers
    
    if rnn_config.cell == "LSTM":
      self.rnns = nn.ModuleList(
        [nn.LSTM(input_size=in_size, hidden_size=out_size, batch_first=True)
        for (in_size, out_size) in zip(in_sizes, out_sizes)])
    elif rnn_config.cell == "GRU":
      self.rnns = nn.ModuleList(
        [nn.GRU(input_size=in_size, hidden_size=out_size, batch_first=True)
        for (in_size, out_size) in zip(in_sizes, out_sizes)])
    else:
      raise NotImplementedError("RNN cell type is not supported")

    self.rnn_dropout = nn.Dropout(rnn_config.dropout)
    self.rnn_residual = rnn_config.residual

    if n_component is None:
      self.postnet = Postnet(
        input_size=rnn_config.hidden_size,
        output_size=self.mel_dim)
    else:
      self.postnet = Postnet(
        input_size=rnn_config.hidden_size,
        output_size=self.mel_dim, n_component=n_component)

  def forward(self, inputs, lengths):
    """Forward function for both training and testing (feature extraction).

    input:
      inputs: (batch_size, seq_len, mel_dim)
      lengths: (batch_size,)

    return:
      predicted_mel: (batch_size, seq_len, mel_dim)
      internal_reps: (num_layers + x, batch_size, seq_len, rnn_hidden_size),
        where x is 1 if there's a prenet, otherwise 0
    """
    seq_len = inputs.size(1)

    if self.prenet is not None:
      rnn_inputs = self.prenet(inputs)
      # rnn_inputs: (batch_size, seq_len, rnn_input_size)
      internal_reps = [rnn_inputs]
      # also include prenet_outputs in internal_reps
    else:
      rnn_inputs = inputs
      internal_reps = []

    # packed_rnn_inputs = pack_padded_sequence(rnn_inputs, lengths, True)

    for i, layer in enumerate(self.rnns):
      # packed_rnn_outputs, _ = layer(packed_rnn_inputs)
      rnn_outputs, _ = layer(rnn_inputs)

      # rnn_outputs, _ = pad_packed_sequence(
        # packed_rnn_outputs, True, total_length=seq_len)
      # outputs: (batch_size, seq_len, rnn_hidden_size)

      if i + 1 < len(self.rnns):
        # apply dropout except the last rnn layer
        rnn_outputs = self.rnn_dropout(rnn_outputs)

      # rnn_inputs, _ = pad_packed_sequence(
        # packed_rnn_inputs, True, total_length=seq_len)
      # rnn_inputs: (batch_size, seq_len, rnn_hidden_size)

      if self.rnn_residual and rnn_inputs.size(-1) == rnn_outputs.size(-1):
        # Residual connections
        rnn_outputs = rnn_outputs + rnn_inputs

      internal_reps.append(rnn_outputs)

      # packed_rnn_inputs = pack_padded_sequence(rnn_outputs, lengths, True)
      rnn_inputs = rnn_outputs
   
    predicted_mel = self.postnet(rnn_outputs)
    
    #if self.n_component is None:
    #  predicted_mel = self.postnet(rnn_outputs)
    ## predicted_mel: (batch_size, seq_len, mel_dim)
    #else:
    #  predicted_mel = self.postnet(rnn_outputs)#.view(1, lengths[0], self.mel_dim, self.n_component)
    #  predicted_mel = predicted_mel.reshape(self.n_component, 1, lengths[0], self.mel_dim)

    #  predicted_mel = torch.stack([self.postnet[i](rnn_outputs) for i in range(self.n_component)])  

    internal_reps = torch.stack(internal_reps)
    
    return predicted_mel, internal_reps
    # predicted_mel is only for training; internal_reps is the extracted
    # features
