from collections import OrderedDict

import torch
import torch.nn as nn


class FrequencyPositionalEmbedding(nn.Module):
    """The sin/cosine positional embedding. Given an input tensor `x` of shape [n_batch, ..., c_dim], it converts
    each feature dimension of `x[..., i]` into:
        [
            sin(x[..., i]),
            sin(f_1*x[..., i]),
            sin(f_2*x[..., i]),
            ...
            sin(f_N * x[..., i]),
            cos(x[..., i]),
            cos(f_1*x[..., i]),
            cos(f_2*x[..., i]),
            ...
            cos(f_N * x[..., i]),
            x[..., i]     # only present if include_input is True.
        ], here f_i is the frequency.

    Denote the space is [0 / num_freqs, 1 / num_freqs, 2 / num_freqs, 3 / num_freqs, ..., (num_freqs - 1) / num_freqs].
    If logspace is True, then the frequency f_i is [2^(0 / num_freqs), ..., 2^(i / num_freqs), ...];
    Otherwise, the frequencies are linearly spaced between [1.0, 2^(num_freqs - 1)].

    Args:
        num_freqs (int): the number of frequencies, default is 6;
        logspace (bool): If logspace is True, then the frequency f_i is [..., 2^(i / num_freqs), ...],
            otherwise, the frequencies are linearly spaced between [1.0, 2^(num_freqs - 1)];
        input_dim (int): the input dimension, default is 3;
        include_input (bool): include the input tensor or not, default is True.

    Attributes:
        frequencies (torch.Tensor): If logspace is True, then the frequency f_i is [..., 2^(i / num_freqs), ...],
                otherwise, the frequencies are linearly spaced between [1.0, 2^(num_freqs - 1);

        out_dim (int): the embedding size, if include_input is True, it is input_dim * (num_freqs * 2 + 1),
            otherwise, it is input_dim * num_freqs * 2.

    """

    def __init__(
        self,
        num_freqs: int = 6,
        logspace: bool = True,
        input_dim: int = 3,
        include_input: bool = True,
        include_pi: bool = True,
    ) -> None:
        """The initialization"""

        super().__init__()

        if logspace:
            frequencies = 2.0 ** torch.arange(num_freqs, dtype=torch.float32)
        else:
            frequencies = torch.linspace(
                1.0, 2.0 ** (num_freqs - 1), num_freqs, dtype=torch.float32
            )

        if include_pi:
            frequencies *= torch.pi

        self.register_buffer("frequencies", frequencies, persistent=False)
        self.include_input = include_input
        self.num_freqs = num_freqs

        self.out_dim = self._get_dims(input_dim)

    def _get_dims(self, input_dim):
        temp = 1 if self.include_input or self.num_freqs == 0 else 0
        out_dim = input_dim * (self.num_freqs * 2 + temp)

        return out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward process.

        Args:
            x: tensor of shape [..., dim]

        Returns:
            embedding: an embedding of `x` of shape [..., dim * (num_freqs * 2 + temp)]
                where temp is 1 if include_input is True and 0 otherwise.
        """

        if self.num_freqs > 0:
            embed = (x[..., None].contiguous() * self.frequencies.to(device=x.device)).view(
                *x.shape[:-1], -1
            )
            if self.include_input:
                return torch.cat((x, embed.sin(), embed.cos()), dim=-1)
            else:
                return torch.cat((embed.sin(), embed.cos()), dim=-1)
        else:
            return x


class Model(nn.Module):
    ''' Runtime Model
    Example model:
        static parameters:
            input.weight.shape: (64, 30) => (1920,)
            input.bias.shape: (64,) => (64,)
            input_bn.alpha.shape: (64,) => (64,)
            input_bn.beta.shape: (64,) => (64,)
            1920 + 64 + 64 + 64 = 2112

            h.weight.shape: (64, 64) => (4096,)
            h.bias.shape: (64,) => (64,)
            h_bn.alpha.shape: (64,) => (64,)
            h_bn.beta.shape: (64,) => (64,)
            4096 + 64 + 64 + 64 = 4288

            crt.weight.shape: (75, 64) => (4800,)
            crt.bias.shape: (75,) => (75,)
            4800 + 75 = 4875

            2112 + 4288 + 4875 = 11275

        dynamic parameters:
            output.weight.shape: (2462, 64) => (157568,)
            output.bias.shape: (2462,) => (2462,)

    Total parameters: 11275 + N + N * 64 = 11275 + N * 65
    '''

    def __init__(
        self,
        input_layer: int,
        output_layer: int,
        hidden_layer: int,
        crt_layer: int,
    ) -> None:
        super().__init__()
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.hidden_layer = hidden_layer
        self.model = nn.Sequential(OrderedDict({
            'input': nn.Linear(input_layer, hidden_layer),
            'input_bn': nn.BatchNorm1d(hidden_layer),
            'input_lrelu': nn.LeakyReLU(inplace=True),
            'h': nn.Linear(hidden_layer, hidden_layer),
            'h_bn': nn.BatchNorm1d(hidden_layer),
            'h_lrelu': nn.LeakyReLU(inplace=True),
        }))
        self.model.apply(init_weights)

        self.mesh_layer = nn.Linear(hidden_layer, output_layer)
        self.mesh_layer.apply(init_weights)

        self.crt_layer = nn.Linear(hidden_layer, crt_layer)
        self.crt_layer.apply(init_weights)

    def forward(self, x: torch.Tensor):
        x = self.model(x)
        return self.mesh_layer(x), self.crt_layer(x)


class FakeFullLayer():
    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor,
        alpha: torch.Tensor = None,
        beta: torch.Tensor = None,
    ) -> None:
        self.weight = weight
        self.bias = bias
        self.alpha = alpha
        self.beta = beta

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # if len(x.shape) == 3:
        #     x = x.reshape(-1, x.shape[1] * x.shape[2])
        # elif len(x.shape) == 2:
        #     x = x.reshape(-1, x.shape[1])

        weights = self.weight
        bias = self.bias
        alpha = self.alpha
        beta = self.beta
        if x.shape[0] != self.weight.shape[0]:
            bluk_size = x.shape[0] // self.weight.shape[0]
            weights = self.weight.repeat(bluk_size, 1, 1)
            bias = self.bias.repeat(bluk_size, 1)
            if self.alpha is not None and self.beta is not None:
                alpha = self.alpha.repeat(bluk_size, 1)
                beta = self.beta.repeat(bluk_size, 1)

        x = do_fc(x, weights, bias)
        if alpha is not None and beta is not None:
            x = fused_bn(x, alpha, beta)
            x = leaky_relu(x)
        return x

    def loss_fn(
        self,
        loss_fn: callable,
        weight: torch.Tensor,
        bias: torch.Tensor,
        alpha: torch.Tensor = None,
        beta: torch.Tensor = None,
        debug: bool = False,
    ) -> torch.Tensor:
        weight_loss = loss_fn(self.weight, weight.reshape(self.weight.shape))
        bias_loss = loss_fn(self.bias, bias.reshape(self.bias.shape))
        loss = weight_loss + bias_loss
        if self.alpha is not None and self.beta is not None:
            alpha_loss = loss_fn(self.alpha, alpha.reshape(self.alpha.shape))
            beta_loss = loss_fn(self.beta, beta.reshape(self.beta.shape))
            loss += alpha_loss + beta_loss
            if debug:
                print(f'debug losses weight: {weight_loss.item()}, bias: {bias_loss.item()}, alpha: {alpha_loss.item()}, beta: {beta_loss.item()}')
        return loss


def make_layers(
    name: str,
    input_channel: int,
    hidden_channel: int,
    output_channel: int,
    hidden_layer_count: int,
    batch_norm: bool = False,
    output_with_activation: bool = True,
) -> nn.Sequential:
    d = {
        f'{name}_i': nn.Linear(input_channel, hidden_channel),
    }
    if batch_norm:
        d[f'{name}_i_bn'] = nn.BatchNorm1d(hidden_channel)
    d[f'{name}_i_lrelu'] = nn.LeakyReLU(inplace=True)

    for i in range(hidden_layer_count):
        d[f'{name}_{i}'] = nn.Linear(hidden_channel, hidden_channel)
        if batch_norm:
            d[f'{name}_{i}_bn'] = nn.BatchNorm1d(hidden_channel)
        d[f'{name}_{i}_lrelu'] = nn.LeakyReLU(inplace=True)
    d[f'{name}_o'] = nn.Linear(hidden_channel, output_channel)
    if output_with_activation:
        if batch_norm:
            d[f'{name}_o_bn'] = nn.BatchNorm1d(output_channel)
        d[f'{name}_o_lrelu'] = nn.LeakyReLU(inplace=True)

    return nn.Sequential(OrderedDict(d))


class TrainingModel(nn.Module):
    ''' Training Model
    '''
    def __init__(
        self,
        input_channel: int,
        encoder_hidden_layer = 128,
        encoder_output_layer: int = 768,
        encoder_state_dict: dict = None,
    ) -> None:
        super().__init__()
        self.input_channel = input_channel

        self.pos = FrequencyPositionalEmbedding(input_dim=self.input_channel)
        self.encoder_model = make_layers(
            'encoder',
            input_channel=26,
            hidden_channel=encoder_hidden_layer,
            output_channel=encoder_output_layer,
            hidden_layer_count=2,
            batch_norm=True,
            output_with_activation=True,
        )
        self.encoder_model.apply(init_weights)

        self.input_layer_model = make_layers(
            'input',
            input_channel=encoder_output_layer,
            hidden_channel=encoder_output_layer,
            output_channel=2112,
            hidden_layer_count=2,
            batch_norm=False,
            output_with_activation=False,
        )
        self.input_layer_model.apply(init_weights)

        self.hidden_layer_model = make_layers(
            'hidden',
            input_channel=encoder_output_layer,
            hidden_channel=1024,
            output_channel=4288,
            hidden_layer_count=2,
            batch_norm=False,
            output_with_activation=False,
        )
        self.hidden_layer_model.apply(init_weights)
        self.crt_layer_model = make_layers(
            'crt',
            input_channel=encoder_output_layer,
            hidden_channel=encoder_output_layer,
            output_channel=4875,
            hidden_layer_count=2,
            batch_norm=False,
            output_with_activation=False,
        )
        self.crt_layer_model.apply(init_weights)
        self.mesh_model = make_layers(
            'output',
            input_channel=encoder_output_layer,
            hidden_channel=encoder_output_layer,
            output_channel=65 * input_channel,
            hidden_layer_count=2,
            batch_norm=True,
            output_with_activation=False,
        )
        self.mesh_model.apply(init_weights)

    def _run_layer(
        self,
        input: torch.Tensor,
        layer: nn.Sequential,
        weight_shape: tuple,
        is_linear_only: bool = False,
    ) -> FakeFullLayer:
        x = layer(input)

        first_layer = weight_shape[0] * weight_shape[1]
        hidden_layer = weight_shape[0]

        start = 0
        end = first_layer
        weight = x[:, start:end].reshape(-1, weight_shape[0], weight_shape[1]).clone()

        start = end
        end = end + hidden_layer

        bias = x[:, start:end].clone()

        if is_linear_only:
            assert x.shape[1] == end, f'Expected {end} features, got {x.shape[1]}'
            alpha = None
            beta = None
        else:
            start = end
            end = end + hidden_layer
            alpha=x[:, start:end].clone()
            start = end
            end = end + hidden_layer
            beta = x[:, start:end].clone()
            assert x.shape[1] == end, f'Expected {end} features, got {x.shape[1]}'

        return FakeFullLayer(
            weight=weight,
            bias=bias,
            alpha=alpha,
            beta=beta,
        )

    def _run_mesh_layer(
        self,
        input: torch.Tensor,
        layer: nn.Sequential,
        weight_shape: tuple,
    ) -> FakeFullLayer:
        x = layer(input)

        x = x.reshape([weight_shape[0], -1])
        return FakeFullLayer(
            weight=x[None, :, :weight_shape[1]].clone(),
            bias=x[None, :, weight_shape[1]].clone(),
        )

    def forward(
        self,
        input_vertices: torch.Tensor,
    ) -> tuple[FakeFullLayer, FakeFullLayer, FakeFullLayer, FakeFullLayer]:
        p = self.pos(input_vertices)[0]
        x = self.encoder_model(p)

        x_mean = x.mean(dim=0, keepdim=True)
        input_layer_weights = self._run_layer(x_mean, self.input_layer_model, (64, 30))
        hidden_layer_weights = self._run_layer(x_mean, self.hidden_layer_model, (64, 64))
        crt_layer_weights = self._run_layer(x_mean, self.crt_layer_model, (75, 64), is_linear_only=True)

        mesh_weights = self._run_mesh_layer(
            x,
            self.mesh_model,
            (input_vertices.shape[-1] * input_vertices.shape[-2], 64),
        )
        return input_layer_weights, hidden_layer_weights, crt_layer_weights, mesh_weights


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(1e-5)


def do_fc(
    input_arr: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    x = torch.bmm(weight, input_arr)
    x = x + bias.reshape(x.shape)
    return x


def fused_bn(
    x: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    x = x * alpha.reshape(x.shape)
    x = x + beta.reshape(x.shape)
    return x


def leaky_relu(x: torch.Tensor, alpha: float = 1e-2) -> torch.Tensor:
    return torch.where(x > 0, x, x * alpha)


def runtime_model(
    input_arr: torch.Tensor,
    input_layer: FakeFullLayer,
    hidden_layer: FakeFullLayer,
    mesh_layer: FakeFullLayer,
    crt_layer: FakeFullLayer,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = input_arr.shape[0]
    bluk_size = 1
    if len(input_arr.shape) == 3:
        input_arr = input_arr.reshape(batch_size, -1, 1)
    elif len(input_arr.shape) == 4:
        bluk_size = input_arr.shape[1]
        input_arr = input_arr.reshape(batch_size * bluk_size, -1, 1)
    x = input_layer(input_arr)
    x = hidden_layer(x)
    mesh_output = mesh_layer(x)
    crt_output = crt_layer(x)
    if bluk_size == 1:
        mesh_output = mesh_output.reshape(batch_size, -1, 2)
        crt_output = crt_output.reshape(batch_size, 15, 5) 
    else:
        mesh_output = mesh_output.reshape(batch_size, bluk_size, -1, 2)
        crt_output = crt_output.reshape(batch_size, bluk_size, 15, 5) 
    return mesh_output, crt_output
