# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import numpy.random as npr

from maskrcnn_benchmark.layers import smooth_l1_loss, Label_Smoothing_Regression
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.utils import cat

class RelationLossComputation(object):
    """
    Computes the loss for relation triplet.
    Also supports FPN
    """

    def __init__(
        self,
        attri_on,
        num_attri_cat,
        max_num_attri,
        attribute_sampling,
        attribute_bgfg_ratio,
        use_label_smoothing,
        predicate_proportion,
        bf_loss_flag,
    ):
        """
        Arguments:
            bbox_proposal_matcher (Matcher)
            rel_fg_bg_sampler (RelationPositiveNegativeSampler)
        """
        self.attri_on = attri_on
        self.num_attri_cat = num_attri_cat
        self.max_num_attri = max_num_attri
        self.attribute_sampling = attribute_sampling
        self.attribute_bgfg_ratio = attribute_bgfg_ratio
        self.use_label_smoothing = use_label_smoothing
        self.pred_weight = (1.0 / torch.FloatTensor([0.5,] + predicate_proportion)).cuda()
        self.bf_loss_flag = bf_loss_flag
        #self.softmax_func=nn.LogSoftmax(dim=1)
        weight_ce = torch.ones(51).cuda()
        weight_ce[0] = 1.0
        self.thre = 1.3 #10%
        #self.criterion_loss_rel = nn.NLLLoss(weight=weight_ce)
        self.criterion_maskloss_rel = nn.CrossEntropyLoss(weight=weight_ce)
        weight_bce = torch.FloatTensor([1,1]).cuda()
        self.bce_loss_bf = nn.BCEWithLogitsLoss(weight = weight_bce)

        if self.use_label_smoothing:
            self.criterion_loss = Label_Smoothing_Regression(e=0.01)
        else:
            self.criterion_loss = nn.CrossEntropyLoss()
            

    def background_cancel(self,relation_logits,rel_labels):
        soft_output = torch.log(F.softmax(relation_logits))
        max_value = soft_output.max(axis = 1).indices
        n_rel_labels= rel_labels[torch.where(~((max_value == 0) & (max_value == rel_labels)))]
        n_relation_logits= soft_output[torch.where(~((max_value == 0) & (max_value == rel_labels)))]
        return n_relation_logits, n_rel_labels
    
    #def background_append_loss(self,relation_logits,rel_labels):
    #    n_relation_logits, n_rel_labels = self.background_cancel(relation_logits,rel_labels)
    #    #soft_output = torch.log(F.Softmax(relation_logits))
    #    #max_value = soft_output.max(axis = 1).indices
    #    #n_rel_labels= rel_labels[torch.where(~((max_value == 0) & (max_value == rel_labels)))]
    #    #n_relation_logits= soft_output[torch.where(~((max_value == 0) & (max_value == rel_labels)))]
    #    rel_loss=self.criterion_loss_rel(n_relation_logits,n_rel_labels)
    #    return rel_loss

    def foregroundset(self,relation_logits,rel_labels):
        N,C = relation_logits.shape
        fore_weights = np.zeros(C)
        fore_weights[0] = 1
        fore_weight = torch.Tensor(fore_weights).cuda()
        fore_weight = fore_weight.view(1, C).repeat(N, 1)
        return fore_weight

    def freqset(self,relation_logits):
        N,C = relation_logits.shape
        freq_weights = np.zeros(C)
        with open('/home/statlab/scenegraphcode/code1/Scene-Graph-Benchmark/freq.csv', 'r') as f:
            for lidx, freq in enumerate(f):
                if float(freq) > self.thre:
                    freq_weights[lidx] = 1
        freq_weights = torch.Tensor(freq_weights).cuda()
        freq_weight = freq_weights.view(1, C).repeat(N, 1)
        return freq_weight

    def groundset(self,relation_logits,rel_labels):
        N,C = relation_logits.shape
        ground_weight = rel_labels.new_zeros((N, C)).float()
        ground_weight[torch.arange(N), rel_labels] = 1
        return ground_weight

    def replace_masked_values(self,tensor, mask, replace_with):
        assert tensor.dim() == mask.dim(), '{} vs {}'.format(tensor.shape, mask.shape)
        one_minus_mask = 1 - mask
        values_to_add = replace_with * one_minus_mask
        return tensor * mask + values_to_add

    def EQLloss(self,relation_logits,rel_labels,qbj = False):
        if qbj:
            relation_logits,rel_labels = self.background_cancel(relation_logits,rel_labels)
        fore_weight = self.foregroundset(relation_logits,rel_labels)
        freq_weight = self.freqset(relation_logits)
        ground_weight = self.groundset(relation_logits,rel_labels)
        mask =  ((fore_weight + freq_weight + ground_weight) > 0).float()
        eql_relation_logit = self.replace_masked_values(relation_logits,mask,-1e7)
        rel_loss = self.criterion_maskloss_rel(eql_relation_logit,rel_labels)
        return rel_loss
    
    def convert_label(self,rel_l):
        num_l = len(rel_l)
        rel_flag = torch.zeros(num_l)
        rel_flag[torch.where(rel_l > 0)] = 1
        bf_labels = rel_l.new_zeros(num_l,2)
        bf_labels[torch.arange(num_l), rel_flag.long()] = 1
        return bf_labels
    
    def convert_logits(self,relation_logits):
        N,C = relation_logits.shape
        soft_output = F.softmax(relation_logits,dim=1)
        max_value = soft_output.max(1)[1]
        bf_relation_logits = relation_logits.new_zeros(N,2)
        bf_relation_logits[:,0] = relation_logits[:,0]
        for i in range(len(max_value)):
            if max_value[i]>0:
                bf_relation_logits[i,1] = relation_logits[i,1:-1].max()
                #bf_relation_logits[i,0] = bf_relation_logits[i,0]/(C-1)
                bf_relation_logits[i,0] = 0.0
            else:
                bf_relation_logits[i,1] = relation_logits[i,1:-1].mean()
        return bf_relation_logits
    
    def bf_binary_loss(self,relation_logits,rel_labels):
        bf_labels = self.convert_label(rel_labels)
        bf_relation_logits = self.convert_logits(relation_logits)
        bf_loss = self.bce_loss_bf(bf_relation_logits,bf_labels.float())
        return bf_loss


    def __call__(self, proposals, rel_labels, relation_logits, refine_logits):
        """
        Computes the loss for relation triplet.
        This requires that the subsample method has been called beforehand.

        Arguments:
            relation_logits (list[Tensor])
            refine_obj_logits (list[Tensor])

        Returns:
            predicate_loss (Tensor)
            finetune_obj_loss (Tensor)
        """
        if self.attri_on:
            if isinstance(refine_logits[0], (list, tuple)):
                refine_obj_logits, refine_att_logits = refine_logits
            else:
                # just use attribute feature, do not actually predict attribute
                self.attri_on = False
                refine_obj_logits = refine_logits
        else:
            refine_obj_logits = refine_logits

        relation_logits = cat(relation_logits, dim=0)
        refine_obj_logits = cat(refine_obj_logits, dim=0)

        fg_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        rel_labels = cat(rel_labels, dim=0)

        if self.bf_loss_flag:
            loss_bf = self.bf_binary_loss(relation_logits,rel_labels)
        else:
            loss_bf = 0

        if self.use_label_smoothing:
            loss_relation = self.criterion_loss(relation_logits, rel_labels.long())
            loss_refine_obj = self.criterion_loss(refine_obj_logits, fg_labels.long())
        else:
            loss_relation = self.EQLloss(relation_logits,rel_labels.long(), qbj = True)
            #loss_relation = self.background_append_loss(relation_logits,rel_labels.long())
            loss_refine_obj = self.criterion_loss(refine_obj_logits, fg_labels.long())



        # The following code is used to calcaulate sampled attribute loss
        if self.attri_on:
            refine_att_logits = cat(refine_att_logits, dim=0)
            fg_attributes = cat([proposal.get_field("attributes") for proposal in proposals], dim=0)

            attribute_targets, fg_attri_idx = self.generate_attributes_target(fg_attributes)
            if float(fg_attri_idx.sum()) > 0:
                # have at least one bbox got fg attributes
                refine_att_logits = refine_att_logits[fg_attri_idx > 0]
                attribute_targets = attribute_targets[fg_attri_idx > 0]
            else:
                refine_att_logits = refine_att_logits[0].view(1, -1)
                attribute_targets = attribute_targets[0].view(1, -1)

            loss_refine_att = self.attribute_loss(refine_att_logits, attribute_targets, 
                                             fg_bg_sample=self.attribute_sampling, 
                                             bg_fg_ratio=self.attribute_bgfg_ratio)
            return loss_relation, (loss_refine_obj, loss_refine_att)
        else:
            return loss_relation, loss_refine_obj,loss_bf

    def generate_attributes_target(self, attributes):
        """
        from list of attribute indexs to [1,0,1,0,0,1] form
        """
        assert self.max_num_attri == attributes.shape[1]
        device = attributes.device
        num_obj = attributes.shape[0]

        fg_attri_idx = (attributes.sum(-1) > 0).long()
        attribute_targets = torch.zeros((num_obj, self.num_attri_cat), device=device).float()

        for idx in torch.nonzero(fg_attri_idx).squeeze(1).tolist():
            for k in range(self.max_num_attri):
                att_id = int(attributes[idx, k])
                if att_id == 0:
                    break
                else:
                    attribute_targets[idx, att_id] = 1
        return attribute_targets, fg_attri_idx

    def attribute_loss(self, logits, labels, fg_bg_sample=True, bg_fg_ratio=3):
        if fg_bg_sample:
            loss_matrix = F.binary_cross_entropy_with_logits(logits, labels, reduction='none').view(-1)
            fg_loss = loss_matrix[labels.view(-1) > 0]
            bg_loss = loss_matrix[labels.view(-1) <= 0]

            num_fg = fg_loss.shape[0]
            # if there is no fg, add at least one bg
            num_bg = max(int(num_fg * bg_fg_ratio), 1)   
            perm = torch.randperm(bg_loss.shape[0], device=bg_loss.device)[:num_bg]
            bg_loss = bg_loss[perm]

            return torch.cat([fg_loss, bg_loss], dim=0).mean()
        else:
            attri_loss = F.binary_cross_entropy_with_logits(logits, labels)
            attri_loss = attri_loss * self.num_attri_cat / 20.0
            return attri_loss



class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, input, target):
        target = target.view(-1)

        logpt = F.log_softmax(input)
        logpt = logpt.index_select(-1, target).diag()
        logpt = logpt.view(-1)
        pt = logpt.exp()

        logpt = logpt * self.alpha * (target > 0).float() + logpt * (1 - self.alpha) * (target <= 0).float()

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()



def make_roi_relation_loss_evaluator(cfg):

    loss_evaluator = RelationLossComputation(
        cfg.MODEL.ATTRIBUTE_ON,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.MAX_ATTRIBUTES,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.ATTRIBUTE_BGFG_SAMPLE,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.ATTRIBUTE_BGFG_RATIO,
        cfg.MODEL.ROI_RELATION_HEAD.LABEL_SMOOTHING_LOSS,
        cfg.MODEL.ROI_RELATION_HEAD.REL_PROP,
        cfg.MODEL.USE_BFLOSS,
    )

    return loss_evaluator
