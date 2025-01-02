import torch
import torch.nn as nn


##########################################################
########################BLDS FocalLoss#########################
##########################################################
class BinaryFocalLossLogits(nn.Module):
    """ binary focal loss with logits as input """

    def __init__(self, alpha, gamma):
        """ constructor
        :param alpha                    the nodule alpha in focal loss
        :param gamma                    the gamma in focal loss
        """
        super(BinaryFocalLossLogits, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, label):
        """ compute focal loss using logits and binary label as inputs
        :param logits                   the output logits of the network
        :param label                    binary label, 0 or 1
        """
        alphas = torch.tensor([1 - self.alpha, self.alpha], dtype=torch.float, device=logits.device)
        object_p = torch.sigmoid(logits).unsqueeze(1)
        p = torch.cat((1 - object_p, object_p), dim=1)

        one_hot = torch.zeros_like(p)
        one_hot.scatter_(1, label.unsqueeze(1), 1)

        p = (p * one_hot).sum(1) + 1e-10
        alpha = alphas[label]

        batch_loss = - alpha * torch.pow(1-p, self.gamma) * p.log()
        return batch_loss.mean()


class FpnFocalLoss(nn.Module):
    """ loss function of feature pyramid network """

    def __init__(self, focal_alpha, focal_gamma, negative_positive_ratio, minimum_negatives, clsloss_weight):
        """ constructor
        :param focal_alpha                      nodule class weight in focal loss
        :param focal_gamma                      the gamma in focal loss
        :param negative_positive_ratio          the ratio between numbers of negatives and positives
        :param minimum_negatives                the minimum number of negatives
        :param clsloss_weight                   the weight for classification loss
        """
        super(FpnFocalLoss, self).__init__()
        self.regression_loss = nn.SmoothL1Loss()
        self.classification_loss = BinaryFocalLossLogits(alpha=focal_alpha, gamma=focal_gamma)

        self.negative_positive_ratio = negative_positive_ratio
        self.minimum_negatives = minimum_negatives
        self.clsloss_weight = clsloss_weight

    def forward(self, out_targets, gt_targets):
        """ compute FPN loss between network output and ground-truth """

        # diagnostic metrics dictionary
        metric_dict = {}

        # convert anchor tensor to a list of anchors
        out = []
        for target in out_targets:
            out.append(target.view(-1, 7))
        out = torch.cat(out)

        gt = []
        for target in gt_targets:
            gt.append(target.view(-1, 7))
        gt = torch.cat(gt)

        # get positive and negative anchor indexes
        label = gt[:, 0].long()
        pidxs = (label == 1).data.nonzero()
        pidxs = pidxs.squeeze(1) if pidxs.dim() > 1 else pidxs
        nidxs = (label == -1).data.nonzero().squeeze(1)

        label[nidxs] = -1
        assert len((label[nidxs] == -1).nonzero()) == len(nidxs), 'negative label must be -1'
        assert len((label[pidxs] == 1).nonzero()) == len(pidxs), 'positive label must be 1'

        if len(pidxs) > 0 and len(nidxs) > 0:

            # pick proportional negative samples to positive samples
            num_hard_negatives = len(pidxs) * self.negative_positive_ratio
            num_hard_negatives = max(num_hard_negatives, self.minimum_negatives)

            # find top-k negative samples with largest logits/probability
            negative_logits, negative_labels = out[nidxs, 0], label[nidxs]
            num_hard_negatives = min(num_hard_negatives, len(negative_logits))

            hard_negative_logits, hard_negative_idxs = torch.topk(negative_logits, k=num_hard_negatives)

            # concatenate with positive logits to form classification logits of all samples
            classification_logits = torch.cat((out[pidxs, 0], hard_negative_logits))
            classification_gt = torch.cat((label[pidxs], negative_labels[hard_negative_idxs] + 1))
            closs = self.classification_loss(classification_logits, classification_gt)
            closs *= self.clsloss_weight
            closs = closs.unsqueeze(0)

            # compute regression loss on positive samples only
            rloss = 0
            for i in range(1, 7):
                rloss_part = self.regression_loss(out[pidxs, i], gt[pidxs, i])
                rloss += rloss_part
            rloss *= (1.0 / 6.0)
            rloss = rloss.unsqueeze(0)

        elif len(pidxs) == 0 and len(nidxs) > 0:

            # pick proportional negative samples
            num_hard_negatives = self.minimum_negatives

            # pick top-k hard negative samples
            negative_logits, negative_labels = out[nidxs, 0], label[nidxs]
            num_hard_negatives = min(num_hard_negatives, len(negative_logits))

            hard_negative_logits, hard_negative_idxs = torch.topk(negative_logits, k=num_hard_negatives)
            closs = self.classification_loss(hard_negative_logits, negative_labels[hard_negative_idxs] + 1)
            closs *= self.clsloss_weight
            closs = closs.unsqueeze(0)

            # no regression loss because of no positive samples
            rloss = torch.tensor([0], dtype=torch.float32, device=out.device)

        else:
            msg = 'Rare case happened: {} positives, {} negatives'.format(len(pidxs), len(nidxs))
            raise ValueError(msg)

        # return closs, rloss, metric_dict
        return closs, rloss
