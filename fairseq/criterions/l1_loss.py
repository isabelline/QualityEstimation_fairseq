# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import torch.nn.functional as F
import torch
from scipy.stats import pearsonr

from fairseq import utils

from . import FairseqCriterion, register_criterion


@register_criterion('l1_loss')
class MAECriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        target = model.get_hters(sample).view(-1)
        
        net_output = net_output[0]
        net_output = torch.squeeze(net_output)
        
        hter_np = target.data.cpu().numpy()
        output_np = net_output.data.cpu().numpy()
        
        p = pearsonr(hter_np, output_np)[0]
        print("")
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        print(p)
        print("hter")
        print(hter_np)
        print("output")
        print(output_np)

        loss = F.l1_loss(net_output, target, size_average=None, reduce=None, reduction='sum')
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'sample_size': sample_size,
        }
        print("++++++++++++++++++++++++")
        print(loss)
        return loss, sample_size, logging_output

    
    
    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        agg_output = {
            'loss': loss_sum / sample_size ,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        if sample_size != ntokens:
            agg_output['nll_loss'] = loss_sum / ntokens / math.log(2)
        return agg_output
