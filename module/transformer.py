

from typing import Optional

import torch
from torch import nn, Tensor
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
from module.attention import MultiheadAttentionRelative
from utilities.misc import get_clones

layer_idx = 0
visualize=False

class Transformer(nn.Module):
    """
    Transformer computes self (intra image) and cross (inter image) attention
    """

    def __init__(self, hidden_dim: int = 128, nhead: int = 8, num_attn_layers: int = 6):
        super().__init__()

        self_attn_layer = TransformerSelfAttnLayer(hidden_dim, nhead)
        self.self_attn_layers = get_clones(self_attn_layer, num_attn_layers)

        cross_attn_layer = TransformerCrossAttnLayer(hidden_dim, nhead)
        self.cross_attn_layers = get_clones(cross_attn_layer, num_attn_layers)

        self.norm = nn.LayerNorm(hidden_dim)

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.num_attn_layers = num_attn_layers

    def _alternating_attn(self, feat: torch.Tensor,feat1: torch.Tensor,
                          pos_enc: torch.Tensor, pos_indexes: Tensor,pos_enc1: torch.Tensor, pos_indexes1: Tensor,
                          pos_enc_y: torch.Tensor, pos_indexes_y: Tensor,pos_enc_y1: torch.Tensor, pos_indexes_y1: Tensor,
                          hn: int):
        """
        Alternate self and cross attention with gradient checkpointing to save memory

        :param feat: image feature concatenated from left and right, [W,2HN,C]
        :param feat1: small image feature concatenated from left and right, [W,2HN,C]
        :param pos_enc: positional encoding, [W,HN,C]
        :param pos_indexes1: indexes to slice positional encoding, [W,HN,C]
        :param pos_enc_y: positional encoding along y axis
        :param pos_indexes_y: indexes to slice positional encoding along y axis
        :param hn: size of HN
        :return: attention weight [N,H,W,W]
        """

        global layer_idx
        # alternating
        for idx, (self_attn, cross_attn) in enumerate(zip(self.self_attn_layers, self.cross_attn_layers)):
            layer_idx = idx

            # checkpoint self attn
            def create_custom_self_attn(module):
                def custom_self_attn(*inputs):
                    return module(*inputs)

                return custom_self_attn

            if visualize == True:
                torch.save(feat,'feat_self_attn_input_' + str(layer_idx) + '.dat')
            feat,feat1 = checkpoint(create_custom_self_attn(self_attn), feat,feat1,
                               pos_enc, pos_indexes,pos_enc1, pos_indexes1,
                               pos_enc_y, pos_indexes_y,pos_enc_y1, pos_indexes_y1)
            #feat, feat_new = checkpoint(create_custom_self_attn(self_attn), feat, feat1,
                                     #pos_enc, pos_indexes, pos_enc1, pos_indexes1,
                                     #pos_enc_y, pos_indexes_y, pos_enc_y1, pos_indexes_y1)

            #add a flag for last layer of cross attention
            if idx == self.num_attn_layers - 1:
                # checkpoint cross attn
                def create_custom_cross_attn(module):
                    def custom_cross_attn(*inputs):
                        return module(*inputs, True)

                    return custom_cross_attn
            else:
                # checkpoint cross attn
                def create_custom_cross_attn(module):
                    def custom_cross_attn(*inputs):
                        return module(*inputs, False)

                    return custom_cross_attn
            if visualize==True:
                torch.save(feat,'feat_cross_attn_input_' + str(layer_idx) + '.dat')
            feat, attn_weight = checkpoint(create_custom_cross_attn(cross_attn), feat[:, :hn], feat[:, hn:], feat1[:, :hn], feat1[:, hn:],pos_enc,
                                            pos_indexes,pos_enc1,
                                            pos_indexes1)
            #feat, attn_weight = checkpoint(create_custom_cross_attn(cross_attn), feat[:, :hn], feat[:, hn:],
                                           #feat_new[:, :hn], feat_new[:, hn:], pos_enc,
                                           #pos_indexes, pos_enc1,
                                           #pos_indexes1)

        layer_idx = 0
        return attn_weight

    def forward(self, feat_left: torch.Tensor, feat_right: torch.Tensor, feat_left1: torch.Tensor, feat_right1: torch.Tensor,
                pos_enc: Optional[Tensor] = None, pos_enc_y: Optional[Tensor] = None,
                pos_enc1: Optional[Tensor] = None, pos_enc_y1: Optional[Tensor] = None):
        """
        :param feat_left: feature descriptor of left image, [N,C,H,W]
        :param feat_left1: feature descriptor of left small image, [N,C,H,W]
        :param feat_right: feature descriptor of right image, [N,C,H,W]
        :param feat_right1: small feature descriptor of right image, [N,C,H,W]
        :param pos_enc: relative positional encoding, [N,C,H,2W-1]
        :return: cross attention values [N,H,W,W], dim=2 is left image, dim=3 is right image
        """

        # flatten NxCxHxW to WxHNxC
        bs, c, h, w = feat_left.shape
        bs1, c1, h1, w1 = feat_left1.shape
        feat_left = feat_left.permute(1, 3, 2, 0).flatten(2).permute(1, 2, 0)  # CxWxHxN -> CxWxHN -> WxHNxC
        feat_right = feat_right.permute(1, 3, 2, 0).flatten(2).permute(1, 2, 0)
        feat_left1 = feat_left1.permute(1, 3, 2, 0).flatten(2).permute(1, 2, 0)  # CxWxHxN -> CxWxHN -> WxHNxC
        feat_right1 = feat_right1.permute(1, 3, 2, 0).flatten(2).permute(1, 2, 0)
        if pos_enc is not None:
            with torch.no_grad():
                # indexes to shift rel pos encoding
                indexes_r = torch.linspace(w - 1, 0, w).view(w, 1).to(feat_left.device)
                indexes_c = torch.linspace(0, w - 1, w).view(1, w).to(feat_left.device)
                pos_indexes = (indexes_r + indexes_c).view(-1).long()  # WxW' -> WW'
                indexes_r1 = torch.linspace(w1 - 1, 0, w1).view(w1, 1).to(feat_left1.device)
                indexes_c1 = torch.linspace(0, w1 - 1, w1).view(1, w1).to(feat_left1.device)
                pos_indexes1 = (indexes_r1 + indexes_c1).view(-1).long()  # WxW' -> WW'
        else:
            pos_indexes = None
        if pos_enc_y is not None:
            with torch.no_grad():
                # indexes to shift rel pos encoding
                indexes_r1 = torch.linspace(h - 1, 0, h).view(h, 1).to(feat_left.device)
                indexes_c1 = torch.linspace(0, h - 1, h).view(1, h).to(feat_left.device)
                pos_indexes_y = (indexes_r1 + indexes_c1).view(-1).long()  # WxW' -> WW'
                indexes_r1 = torch.linspace(h1 - 1, 0, h1).view(h1, 1).to(feat_left.device)
                indexes_c1 = torch.linspace(0, h1 - 1, h1).view(1, h1).to(feat_left.device)
                pos_indexes_y1 = (indexes_r1 + indexes_c1).view(-1).long()  # WxW' -> WW'
        else:
            pos_indexes_y = None
        # concatenate left and right features
        feat = torch.cat([feat_left, feat_right], dim=1)  # Wx2HNxC
        feat1 = torch.cat([feat_left1, feat_right1], dim=1)
        # compute attention
        attn_weight = self._alternating_attn(feat,feat1, pos_enc, pos_indexes,pos_enc1, pos_indexes1,
                                             pos_enc_y, pos_indexes_y,pos_enc_y1, pos_indexes_y1, bs * h)
        attn_weight = attn_weight.view(h, bs, w, w).permute(1, 0, 2, 3)  # NxHxWxW, dim=2 left image, dim=3 right image

        return attn_weight


class TransformerSelfAttnLayer(nn.Module):
    """
    Self attention layer
    """

    def __init__(self, hidden_dim: int, nhead: int):
        super().__init__()
        self.self_attn1 = MultiheadAttentionRelative(hidden_dim, nhead)
        self.self_attn2 = MultiheadAttentionRelative(hidden_dim, nhead)
        self.self_attn3 = MultiheadAttentionRelative(hidden_dim, nhead)
        self.self_attn4 = MultiheadAttentionRelative(hidden_dim, nhead)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
    def forward(self, feat: Tensor,feat1: Tensor,
                pos: Optional[Tensor] = None,
                pos_indexes: Optional[Tensor] = None,
                pos1: Optional[Tensor] = None,
                pos_indexes1: Optional[Tensor] = None,
                pos_y: Optional[Tensor] = None,
                pos_indexes_y: Optional[Tensor] = None,
                pos_y1: Optional[Tensor] = None,
                pos_indexes_y1: Optional[Tensor] = None
                ):
        """
        :param feat: image feature [W,2HN,C]
        :param pos: pos encoding [2W-1,HN,C]
        :param pos_indexes: indexes to slice pos encoding [W,W]
        :return: updated image feature
        """
        feat2 = self.norm1(feat)
        feat4 = self.norm2(feat1)

        w,h2,c=feat2.shape
        w1, h4, c1 = feat4.shape

        

        feat2=feat2.reshape(w,2,h2//2,c).permute(2,1,0,3).reshape(h2//2,2*w,c)
        feat4 = feat4.reshape(w1, 2, h4 // 2, c1).permute(2, 1, 0, 3).reshape(h4 // 2, 2 * w1, c1)

        feat2, attn_weight, _ = self.self_attn2(query=feat2, key=feat2, value=feat2, pos_enc=pos_y,
                                               pos_indexes=pos_indexes_y)
        feat2=feat2.reshape(h2//2,2,w,c).permute(2,1,0,3).reshape(w,h2,c)

        feat4, attn_weight1, _ = self.self_attn4(query=feat4, key=feat4, value=feat4, pos_enc=pos_y1,
                                                pos_indexes=pos_indexes_y1)
        feat4 = feat4.reshape(h4 // 2, 2, w1, c1).permute(2, 1, 0, 3).reshape(w1, h4, c1)

        feat2, attn_weight, _ = self.self_attn1(query=feat2, key=feat2, value=feat2, pos_enc=pos,
                                               pos_indexes=pos_indexes)
        feat4, attn_weight1, _ = self.self_attn3(query=feat4, key=feat4, value=feat4, pos_enc=pos1,
                                                pos_indexes=pos_indexes1)
        # bachsize(N)=1 [W,2HN,C]->[H,2WN,C]
        
        # torch.save(attn_weight, 'self_attn_' + str(layer_idx) + '.dat')
       
        feat = feat + feat2
        feat_new=feat1+feat4
        
        return feat,feat_new


class TransformerCrossAttnLayer(nn.Module):
    """
    Cross attention layer  test
    """

    def __init__(self, hidden_dim: int, nhead: int):
        super().__init__()
        self.cross_attn = MultiheadAttentionRelative(hidden_dim, nhead)
        self.cross_attn1 = MultiheadAttentionRelative(hidden_dim, nhead)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.merge=nn.Sequential(nn.Conv2d(in_channels=256,out_channels=128,kernel_size=1,stride=1,padding=0),
                                 nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1))
        #self.merge1=nn.Conv2d(in_channels=2,out_channels=1,kernel_size=3,stride=1,padding=1)
    def forward(self, feat_left: Tensor, feat_right: Tensor,feat_left1: Tensor, feat_right1: Tensor,
                pos: Optional[Tensor] = None,
                pos_indexes: Optional[Tensor] = None,
                pos1: Optional[Tensor] = None,
                pos_indexes1: Optional[Tensor] = None,
                last_layer: Optional[bool] = False):
        """
        :param feat_left: left image feature, [W,HN,C]
        :param feat_right: right image feature, [W,HN,C]
        :param pos: pos encoding, [2W-1,HN,C]
        :param pos_indexes: indexes to slicer pos encoding [W,W]
        :param last_layer: Boolean indicating if the current layer is the last layer
        :return: update image feature and attention weight
        """
        feat_left_2 = self.norm1(feat_left)
        feat_right_2 = self.norm1(feat_right)
        feat_left_4 = self.norm2(feat_left1)
        feat_right_4 = self.norm2(feat_right1)
        # torch.save(torch.cat([feat_left_2, feat_right_2], dim=1), 'feat_cross_attn_input_' + str(layer_idx) + '.dat')

        # update right features
        if pos is not None:
            pos_flipped = torch.flip(pos, [0])
        else:
            pos_flipped = pos
        if pos1 is not None:
            pos_flipped1 = torch.flip(pos1, [0])
        else:
            pos_flipped1 = pos1

        feat_right_2 = self.cross_attn(query=feat_right_2, key=feat_left_2, value=feat_left_2, pos_enc=pos_flipped,
                                       pos_indexes=pos_indexes)[0]
        feat_right_4 = self.cross_attn1(query=feat_right_4, key=feat_left_4, value=feat_left_4, pos_enc=pos_flipped1,
                                       pos_indexes=pos_indexes1)[0]

        feat_right = feat_right + feat_right_2
        feat_right_new = feat_right1 + feat_right_4
        # update left features
        # use attn mask for last layer
        if last_layer:
            w = feat_left_2.size(0)
            w1 = feat_left_4.size(0)
            attn_mask = self._generate_square_subsequent_mask(w).to(feat_left.device)  # generate attn mask
            attn_mask1 = self._generate_square_subsequent_mask(w1).to(feat_left.device)  # generate attn mask
        else:
            attn_mask = None
            attn_mask1 = None

        # normalize again the updated right features
        feat_right_2 = self.norm1(feat_right)
        feat_left_2, attn_weight, raw_attn = self.cross_attn(query=feat_left_2, key=feat_right_2, value=feat_right_2,
                                                             attn_mask=attn_mask, pos_enc=pos,
                                                             pos_indexes=pos_indexes)
        feat_right_4 = self.norm2(feat_right1)
        feat_left_4, attn_weight, raw_attn1 = self.cross_attn1(query=feat_left_4, key=feat_right_4, value=feat_right_4,
                                                             attn_mask=attn_mask1, pos_enc=pos1,
                                                             pos_indexes=pos_indexes1)

        # torch.save(attn_weight, 'cross_attn_' + str(layer_idx) + '.dat')

        feat_left = feat_left + feat_left_2
        feat_left_new = feat_left1 + feat_left_4
        # concat features
        feat = torch.cat([feat_left, feat_right], dim=1)  # Wx2HNxC
        feat_new = torch.cat([feat_left_new, feat_right_new], dim=1)  # Wx2HNxC
        feat_new=F.interpolate(feat_new.permute(2,1,0),feat.shape[0],mode="linear").permute(2,1,0)
        feat_new=torch.cat([feat_new,feat],dim=2)
        feat_new=self.merge(feat_new.permute(2,1,0).unsqueeze(0)).squeeze().permute(2,1,0)

        # raw_attn1=F.interpolate(raw_attn1.unsqueeze(0),raw_attn.shape[1],mode="bilinear").squeeze()
        # att_in=torch.cat([raw_attn1.unsqueeze(1),raw_attn.unsqueeze(1)],dim=1)
        # raw_attn=self.merge1(att_in).squeeze(1)

        return feat_new, raw_attn

    @torch.no_grad()
    def _generate_square_subsequent_mask(self, sz: int):
        """
        Generate a mask which is upper triangular

        :param sz: square matrix size
        :return: diagonal binary mask [sz,sz]
        """
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask[mask == 1] = float('-inf')
        return mask


def build_transformer(args):
    return Transformer(
        hidden_dim=args.channel_dim,
        nhead=args.nheads,
        num_attn_layers=args.num_attn_layers
    )
