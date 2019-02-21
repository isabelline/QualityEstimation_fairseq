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
import numpy as np
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
        n_sentences = target.shape[0]
        
        net_output = net_output[0]
        net_output = torch.squeeze(net_output)
        
        hter_np = target.data.cpu().numpy()
        output_np = net_output.data.cpu().numpy()
        
        p = pearsonr(hter_np, output_np)[0]
#        loss = torch.abs(net_output-target)
#        print("^^^^^^^^^^^^^^^^^^^^^^^")
#        print(loss)
#        loss = torch.sum(loss)
#        print("(((((((((((((((((((((((")
#        print(loss)

        loss = F.l1_loss(net_output, target, size_average=None, reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        print("~~~~~~~~~~~~~~~~~")
        print(loss)
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': n_sentences,
            'sample_size': sample_size,
            'hter' :hter_np,
            'pred': output_np
        }

        return loss, sample_size, logging_output

    
    
    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        err = 0
        for log in logging_outputs:
            err += log.get('loss', 0) * log.get('nsentences', 0)
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        mean_err = float(err) /float(nsentences)
        hter_all = np.asarray([])
        pred_all = np.asarray([])
        print("AAAAAAAAAAAAAA")

        for log in logging_outputs:
            hter_all =np.concatenate((log.get('hter', 0),hter_all))
            pred_all =np.concatenate((log.get('pred', 0),pred_all))
            print("WWWWWWWWWWWWWWWWWWWWWW")
            print(hter_all)
            print(pred_all)
                                     
        print("FFFFFFFFFFFFFFFFFFFFFFFFFF")
        print(hter_all)
        p = pearsonr(hter_all, pred_all)[0]

        agg_output = {
            'loss': err / nsentences ,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
            'pearson': p,
        }
        if sample_size != ntokens:
            agg_output['nll_loss'] = loss_sum / ntokens / math.log(2)
        return agg_output
