import math

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


class Encoder(nn.Module):
    """Maps a graph represented as an input sequence
    to a hidden vector"""

    def __init__(self, input_dim, hidden_dim, use_cuda):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.use_cuda = use_cuda
        self.enc_init_state = self.init_hidden(hidden_dim)

    def forward(self, x, hidden):
        output, hidden = self.lstm(x, hidden)
        return output, hidden

    def init_hidden(self, hidden_dim):
        """Trainable initial hidden state"""
        enc_init_hx = Variable(torch.zeros(hidden_dim), requires_grad=False)
        if self.use_cuda:
            enc_init_hx = enc_init_hx.cuda()

        # enc_init_hx.data.uniform_(-(1. / math.sqrt(hidden_dim)),
        #        1. / math.sqrt(hidden_dim))

        enc_init_cx = Variable(torch.zeros(hidden_dim), requires_grad=False)
        if self.use_cuda:
            enc_init_cx = enc_init_cx.cuda()

        # enc_init_cx = nn.Parameter(enc_init_cx)
        # enc_init_cx.data.uniform_(-(1. / math.sqrt(hidden_dim)),
        #        1. / math.sqrt(hidden_dim))
        return (enc_init_hx, enc_init_cx)


class Attention(nn.Module):
    """A generic attention module for a decoder in seq2seq"""

    def __init__(self, dim, use_tanh=False, C=10, use_cuda=True):
        super(Attention, self).__init__()
        self.use_tanh = use_tanh
        self.project_query = nn.Linear(dim, dim)
        self.project_ref = nn.Conv1d(dim, dim, 1, 1)
        self.C = C  # tanh exploration
        self.tanh = nn.Tanh()

        v = torch.FloatTensor(dim)
        if use_cuda:
            v = v.cuda()
        self.v = nn.Parameter(v)
        self.v.data.uniform_(-(1. / math.sqrt(dim)), 1. / math.sqrt(dim))

    def forward(self, query, ref):
        """
        Args:
            query: is the hidden state of the decoder at the current
                time step. batch x dim
            ref: the set of hidden states from the encoder.
                sourceL x batch x hidden_dim
        """
        # ref is now [batch_size x hidden_dim x sourceL]
        ref = ref.permute(1, 2, 0)
        q = self.project_query(query).unsqueeze(2)  # batch x dim x 1
        e = self.project_ref(ref)  # batch_size x hidden_dim x sourceL
        # expand the query by sourceL
        # batch x dim x sourceL
        expanded_q = q.repeat(1, 1, e.size(2))
        # batch x 1 x hidden_dim
        v_view = self.v.unsqueeze(0).expand(
            expanded_q.size(0), len(self.v)).unsqueeze(1)
        # [batch_size x 1 x hidden_dim] * [batch_size x hidden_dim x sourceL]
        u = torch.bmm(v_view, self.tanh(expanded_q + e)).squeeze(1)
        if self.use_tanh:
            logits = self.C * self.tanh(u)
        else:
            logits = u
        return e, logits


class Decoder(nn.Module):
    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 max_length,
                 tanh_exploration,
                 terminating_symbol,
                 use_tanh,
                 decode_type,
                 n_glimpses=1,
                 use_cuda=True):
        super(Decoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_glimpses = n_glimpses
        self.max_length = max_length
        self.terminating_symbol = terminating_symbol
        self.decode_type = decode_type
        self.use_cuda = use_cuda

        self.input_weights = nn.Linear(embedding_dim, 4 * hidden_dim)
        self.hidden_weights = nn.Linear(hidden_dim, 4 * hidden_dim)

        self.pointer = Attention(hidden_dim, use_tanh=use_tanh, C=tanh_exploration, use_cuda=self.use_cuda)
        self.glimpse = Attention(hidden_dim, use_tanh=False, use_cuda=self.use_cuda)
        self.sm = nn.Softmax(dim=1)

    def apply_mask_to_logits(self, step, logits, mask, prev_idxs):
        if mask is None:
            mask = torch.zeros(logits.size()).byte()
            if self.use_cuda:
                mask = mask.cuda()

        maskk = mask.clone()

        if step == 0:
            maskk[[x for x in range(logits.size(0))], 0] = 1
            logits[maskk.bool()] = -np.inf

        else:

            # to prevent them from being reselected.
            # Or, allow re-selection and penalize in the objective function
            if prev_idxs is not None:
                if step == 1:
                    maskk[[x for x in range(logits.size(0))], 0] = 0
                # set most recently selected idx values to 1
                maskk[[x for x in range(logits.size(0))],
                      prev_idxs.data] = 1

            logits[maskk.bool()] = -np.inf

        return logits, maskk

    def forward(self, decoder_input, embedded_inputs, hidden, context):
        """
        Args:
            decoder_input: The initial input to the decoder
                size is [batch_size x embedding_dim]. Trainable parameter.
            embedded_inputs: [sourceL x batch_size x embedding_dim]
            hidden: the prev hidden state, size is [batch_size x hidden_dim].
                Initially this is set to (enc_h[-1], enc_c[-1])
            context: encoder outputs, [sourceL x batch_size x hidden_dim]
        """

        def recurrence(x, hidden, logit_mask, prev_idxs, step):

            hx, cx = hidden  # batch_size x hidden_dim

            gates = self.input_weights(x) + self.hidden_weights(hx)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * torch.tanh(cy)  # batch_size x hidden_dim

            g_l = hy
            for i in range(self.n_glimpses):
                ref, logits = self.glimpse(g_l, context)
                logits, logit_mask = self.apply_mask_to_logits(step, logits, logit_mask, prev_idxs)
                # [batch_size x h_dim x sourceL] * [batch_size x sourceL x 1] =
                # [batch_size x h_dim x 1]
                g_l = torch.bmm(ref, self.sm(logits).unsqueeze(2)).squeeze(2)
            _, logits = self.pointer(g_l, context)

            logits, logit_mask = self.apply_mask_to_logits(step, logits, logit_mask, prev_idxs)
            probs = self.sm(logits)
            return hy, cy, probs, logit_mask

        batch_size = context.size(1)
        outputs = []
        selections = []
        steps = range(self.max_length)  # or until terminating symbol ?
        inps = []
        idxs = None
        mask = None

        for i in steps:
            hx, cx, probs, mask = recurrence(decoder_input, hidden, mask, idxs, i)
            hidden = (hx, cx)
            # select the next inputs for the decoder [batch_size x hidden_dim]
            if self.decode_type == "stochastic":
                decoder_input, idxs = self.decode_stochastic(
                    probs,
                    embedded_inputs,
                    selections)
            if self.decode_type == 'greedy':
                decoder_input, idxs = self.decode_greedy(
                    probs,
                    embedded_inputs,
                    selections)

            inps.append(decoder_input)
            # use outs to point to next object
            outputs.append(probs)
            selections.append(idxs)
        return (outputs, selections), hidden

    def decode_stochastic(self, probs, embedded_inputs, selections):
        """
        Return the next input for the decoder by selecting the
        input corresponding to the max output

        Args:
            probs: [batch_size x sourceL]
            embedded_inputs: [sourceL x batch_size x embedding_dim]
            selections: list of all of the previously selected indices during decoding
       Returns:
            Tensor of size [batch_size x sourceL] containing the embeddings
            from the inputs corresponding to the [batch_size] indices
            selected for this iteration of the decoding, as well as the
            corresponding indicies
        """
        batch_size = probs.size(0)
        # idxs is [batch_size]
        # idxs = probs.multinomial().squeeze(1)

        idxs = probs.multinomial(1).squeeze(1)

        # due to race conditions, might need to resample here
        for old_idxs in selections:
            # compare new idxs
            # elementwise with the previous idxs. If any matches,
            # then need to resample
            if old_idxs.eq(idxs).data.any():
                print(' [!] resampling due to race condition')
                idxs = probs.multinomial(1).squeeze(1)
                break

        sels = embedded_inputs[idxs.data, [i for i in range(batch_size)], :]
        return sels, idxs

    def decode_greedy(self, probs, embedded_inputs, selections):
        """
        Return the next input for the decoder by selecting the
        input corresponding to the max output

        Args:
            probs: [batch_size x sourceL]
            embedded_inputs: [sourceL x batch_size x embedding_dim]
            selections: list of all of the previously selected indices during decoding
       Returns:
            Tensor of size [batch_size x sourceL] containing the embeddings
            from the inputs corresponding to the [batch_size] indices
            selected for this iteration of the decoding, as well as the
            corresponding indicies
        """
        batch_size = probs.size(0)
        # idxs is [batch_size]
        # idxs = probs.multinomial().squeeze(1)

        idxs = torch.argmax(probs, 1)

        # due to race conditions, might need to resample here
        for old_idxs in selections:
            # compare new idxs
            # elementwise with the previous idxs. If any matches,
            # then need to resample
            if old_idxs.eq(idxs).data.any():
                print(' [!] resampling due to race condition')
                idxs = probs.multinomial(1).squeeze(1)
                break

        sels = embedded_inputs[idxs.data, [i for i in range(batch_size)], :]
        return sels, idxs


class PointerNetwork(nn.Module):
    """The pointer network, which is the core seq2seq
    model"""

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 max_decoding_len,
                 terminating_symbol,
                 n_glimpses,
                 tanh_exploration,
                 use_tanh,
                 use_cuda):
        super(PointerNetwork, self).__init__()

        self.encoder = Encoder(
            embedding_dim,
            hidden_dim,
            use_cuda)

        self.decoder = Decoder(
            embedding_dim,
            hidden_dim,
            max_length=max_decoding_len,
            tanh_exploration=tanh_exploration,
            use_tanh=use_tanh,
            terminating_symbol=terminating_symbol,
            decode_type="stochastic",
            n_glimpses=n_glimpses,
            use_cuda=use_cuda)

        # Trainable initial hidden states
        dec_in_0 = torch.FloatTensor(embedding_dim)
        if use_cuda:
            dec_in_0 = dec_in_0.cuda()

        self.decoder_in_0 = nn.Parameter(dec_in_0)
        self.decoder_in_0.data.uniform_(-(1. / math.sqrt(embedding_dim)),
                                        1. / math.sqrt(embedding_dim))

    def forward(self, inputs):
        """ Propagate inputs through the network
        Args:
            inputs: [sourceL x batch_size x embedding_dim]
        """

        (encoder_hx, encoder_cx) = self.encoder.enc_init_state
        encoder_hx = encoder_hx.unsqueeze(0).repeat(inputs.size(1), 1).unsqueeze(0)
        encoder_cx = encoder_cx.unsqueeze(0).repeat(inputs.size(1), 1).unsqueeze(0)

        # encoder forward pass
        enc_h, (enc_h_t, enc_c_t) = self.encoder(inputs, (encoder_hx, encoder_cx))

        dec_init_state = (enc_h_t[-1], enc_c_t[-1])

        # repeat decoder_in_0 across batch
        decoder_input = self.decoder_in_0.unsqueeze(0).repeat(inputs.size(1), 1)

        (pointer_probs, input_idxs), dec_hidden_t = self.decoder(decoder_input,
                                                                 inputs,
                                                                 dec_init_state,
                                                                 enc_h)

        return pointer_probs, input_idxs


class NeuralCombOptRL(nn.Module):
    """
    This module contains the PointerNetwork (actor) and
    CriticNetwork (critic). It requires
    an application-specific reward function
    """

    def __init__(self,
                 input_dim,
                 embedding_dim,
                 hidden_dim,
                 max_decoding_len,
                 terminating_symbol,
                 n_glimpses,
                 n_process_block_iters,
                 tanh_exploration,
                 use_tanh,
                 is_train,
                 use_cuda):
        super(NeuralCombOptRL, self).__init__()
        self.input_dim = input_dim
        self.is_train = is_train
        self.use_cuda = use_cuda

        self.actor_net = PointerNetwork(
            embedding_dim,
            hidden_dim,
            max_decoding_len,
            terminating_symbol,
            n_glimpses,
            tanh_exploration,
            use_tanh,
            use_cuda)

        embedding_ = torch.FloatTensor(input_dim,
                                       embedding_dim)
        if self.use_cuda:
            embedding_ = embedding_.cuda()
        self.embedding = nn.Parameter(embedding_)
        self.embedding.data.uniform_(-(1. / math.sqrt(embedding_dim)),
                                     1. / math.sqrt(embedding_dim))

    def forward(self, inputs):
        """
        Args:
            inputs: [batch_size, input_dim, sourceL]
        """
        batch_size = inputs.size(0)
        input_dim = inputs.size(1)
        sourceL = inputs.size(2)

        # repeat embeddings across batch_size
        # result is [batch_size x input_dim x embedding_dim]
        embedding = self.embedding.repeat(batch_size, 1, 1)
        embedded_inputs = []
        # result is [batch_size, 1, input_dim, sourceL]
        ips = inputs.unsqueeze(1)

        for i in range(sourceL):
            # [batch_size x 1 x input_dim] * [batch_size x input_dim x embedding_dim]
            # result is [batch_size, embedding_dim]
            embedded_inputs.append(torch.bmm(
                ips[:, :, :, i].float(),
                embedding).squeeze(1))

        # Result is [sourceL x batch_size x embedding_dim]
        embedded_inputs = torch.cat(embedded_inputs).view(
            sourceL,
            batch_size,
            embedding.size(2))

        # query the actor net for the input indices
        # making up the output, and the pointer attn
        probs_, action_idxs = self.actor_net(embedded_inputs)

        # Select the actions (inputs pointed to
        # by the pointer net) and the corresponding
        # logits
        # should be size [batch_size x
        actions = []
        # inputs is [batch_size, input_dim, sourceL]
        inputs_ = inputs.transpose(1, 2)
        # inputs_ is [batch_size, sourceL, input_dim]
        for action_id in action_idxs:
            actions.append(inputs_[[x for x in range(batch_size)], action_id.data, :])

        if self.is_train:
            # probs_ is a list of len sourceL of [batch_size x sourceL]
            probs = []
            for prob, action_id in zip(probs_, action_idxs):
                probs.append(prob[[x for x in range(batch_size)], action_id.data])
        else:
            # return the list of len sourceL of [batch_size x sourceL]
            probs = probs_

        _ = None

        # return _, v, probs, actions, action_idxs
        return _, probs, actions, action_idxs
