import torch

from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.modeling.sam.mask_decoder import MaskDecoder
from sam2ct.gsps_prompt_encoder import GSPSPromptEncoder
from sam2.modeling.sam.transformer import TwoWayTransformer
from sam2.modeling.sam2_utils import MLP
from sam2.modeling.sam2_utils import get_1d_sine_pe
import numpy as np

class SAM2CTVolumePredictor(SAM2VideoPredictor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.use_mem_cond_mems = True

    def _build_sam_heads(self):
        self.sam_prompt_embed_dim = self.hidden_dim
        self.sam_image_embedding_size = self.image_size // self.backbone_stride

        # build PromptEncoder and MaskDecoder from SAM
        # (their hyperparameters like `mask_in_chans=16` are from SAM code)
        self.sam_prompt_encoder = GSPSPromptEncoder(
            embed_dim=self.sam_prompt_embed_dim,
            image_embedding_size=(
                self.sam_image_embedding_size,
                self.sam_image_embedding_size,
            ),
            input_image_size=(self.image_size, self.image_size),
            mask_in_chans=16,
        )
        self.sam_mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=self.sam_prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=self.sam_prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            use_high_res_features=self.use_high_res_features_in_sam,
            iou_prediction_use_sigmoid=self.iou_prediction_use_sigmoid,
            pred_obj_scores=self.pred_obj_scores,
            pred_obj_scores_mlp=self.pred_obj_scores_mlp,
            use_multimask_token_for_obj_ptr=self.use_multimask_token_for_obj_ptr,
            **(self.sam_mask_decoder_extra_args or {}),
        )
        if self.use_obj_ptrs_in_encoder:
            # a linear projection on SAM output tokens to turn them into object pointers
            self.obj_ptr_proj = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
            if self.use_mlp_for_obj_ptr_proj:
                self.obj_ptr_proj = MLP(
                    self.hidden_dim, self.hidden_dim, self.hidden_dim, 3
                )
        else:
            self.obj_ptr_proj = torch.nn.Identity()
        if self.proj_tpos_enc_in_obj_ptrs:
            # a linear projection on temporal positional encoding in object pointers to
            # avoid potential interference with spatial positional encoding
            self.obj_ptr_tpos_proj = torch.nn.Linear(self.hidden_dim, self.mem_dim)
        else:
            self.obj_ptr_tpos_proj = torch.nn.Identity()

    def encode_volume(self, image_data, device):
        N, C, H, W = image_data.shape

        chunk_size = 8
        backbone_out = None
        for i in range(0, N, chunk_size):
            end = min(i+chunk_size, N)
            image_chunk = image_data[i:end].to(device)

            chunk_out = self.image_encoder(image_chunk)
            if backbone_out is None:
                backbone_out = chunk_out
            else:
                backbone_out['vision_features'] = torch.cat((backbone_out['vision_features'], chunk_out['vision_features']), dim=0)

                for idx in range(len(chunk_out['vision_pos_enc'])):
                    backbone_out['vision_pos_enc'][idx] = torch.cat((backbone_out['vision_pos_enc'][idx], chunk_out['vision_pos_enc'][idx]), dim=0)

                for idx in range(len(chunk_out['backbone_fpn'])):
                    backbone_out['backbone_fpn'][idx] = torch.cat((backbone_out['backbone_fpn'][idx], chunk_out['backbone_fpn'][idx]), dim=0)


        if self.use_high_res_features_in_sam:
            # precompute projected level 0 and level 1 features in SAM decoder
            # to avoid running it again on every SAM click
            backbone_out["backbone_fpn"][0] = self.sam_mask_decoder.conv_s0(
                backbone_out["backbone_fpn"][0]
            )
            backbone_out["backbone_fpn"][1] = self.sam_mask_decoder.conv_s1(
                backbone_out["backbone_fpn"][1]
            )

        return backbone_out
    
    def _encode_new_memory(
        self,
        current_vision_feats,
        current_high_res_mem_features,
        feat_sizes,
        pred_masks_high_res,
        object_score_logits,
        is_mask_from_pts,
        pix_feat=None
    ):
        """Encode the current image and its prediction into a memory feature."""
        if pix_feat is None:
            B = current_vision_feats[-1].size(1)  # batch size on this frame
            C = self.hidden_dim
            H, W = feat_sizes[-1]  # top-level (lowest-resolution) feature size
            # top-level feature, (HW)BC => BCHW
            pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)

        if self.non_overlap_masks_for_mem_enc and not self.training:
            # optionally, apply non-overlapping constraints to the masks (it's applied
            # in the batch dimension and should only be used during eval, where all
            # the objects come from the same video under batch size 1).
            pred_masks_high_res = self._apply_non_overlapping_constraints(
                pred_masks_high_res
            )
        # scale the raw mask logits with a temperature before applying sigmoid
        binarize = self.binarize_mask_from_pts_for_mem_enc and is_mask_from_pts
        if binarize and not self.training:
            mask_for_mem = (pred_masks_high_res > 0).float()
        else:
            # apply sigmoid on the raw mask logits to turn them into range (0, 1)
            mask_for_mem = torch.sigmoid(pred_masks_high_res)
        # apply scale and bias terms to the sigmoid probabilities
        if self.sigmoid_scale_for_mem_enc != 1.0:
            mask_for_mem = mask_for_mem * self.sigmoid_scale_for_mem_enc
        if self.sigmoid_bias_for_mem_enc != 0.0:
            mask_for_mem = mask_for_mem + self.sigmoid_bias_for_mem_enc

        use_high_res_for_mem = False
        if use_high_res_for_mem:
            maskmem_out = self.memory_encoder(
                current_high_res_mem_features, mask_for_mem, skip_mask_sigmoid=True)
        else:
            maskmem_out = self.memory_encoder(
                pix_feat, mask_for_mem, skip_mask_sigmoid=True)
        maskmem_features = maskmem_out["vision_features"]
        maskmem_pos_enc = maskmem_out["vision_pos_enc"]
        # add a no-object embedding to the spatial memory to indicate that the frame
        # is predicted to be occluded (i.e. no object is appearing in the frame)
        if self.no_obj_embed_spatial is not None:
            is_obj_appearing = (object_score_logits > 0).float()
            maskmem_features += (
                1 - is_obj_appearing[..., None, None]
            ) * self.no_obj_embed_spatial[..., None, None].expand(
                *maskmem_features.shape
            )

        return maskmem_features, maskmem_pos_enc
    
    def get_memory_conditioned_features(
            self,
            current_idx,
            is_init_slice,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
            conditioned_mems,
            unconditioned_mems):
        B = current_vision_feats[-1].size(1)  # batch size on this frame
        C = self.hidden_dim
        H, W = feat_sizes[-1]  # top-level (lowest-resolution) feature size
        device = current_vision_feats[-1].device

        to_cat_memory, to_cat_memory_pos_embed = [], []
        if is_init_slice:
            if self.directly_add_no_mem_embed:
                # directly add no-mem embedding (instead of using the transformer encoder)
                pix_feat_with_mem = current_vision_feats[-1] + self.no_mem_embed
                pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
                return pix_feat_with_mem
            # Use a dummy token on the first frame (to avoid empty memory input to tranformer encoder)
            to_cat_memory = [self.no_mem_embed.expand(1, B, self.mem_dim)]
            to_cat_memory_pos_embed = [self.no_mem_pos_enc.expand(1, B, self.mem_dim)]
        else:

            # 1. conditioned memory
            for idx, memory in conditioned_mems.items():
                feats = memory['maskmem_features']
                maskmem_enc = memory["maskmem_pos_enc"][-1]
                feats = feats.flatten(2).permute(2, 0, 1)
                maskmem_enc = maskmem_enc.flatten(2).permute(2, 0, 1)
                temporal_idx = len(self.maskmem_tpos_enc) - 1

                maskmem_enc = (maskmem_enc + self.maskmem_tpos_enc[temporal_idx])
                to_cat_memory.append(feats)
                to_cat_memory_pos_embed.append(maskmem_enc)
            # 2. unconditioned memory
            max_non_cond_mems = self.num_maskmem - 1

            non_cond_idx = [idx for idx in unconditioned_mems.keys() if np.abs(idx - current_idx) <= max_non_cond_mems]

            for idx in non_cond_idx:
                feats = unconditioned_mems[idx]['maskmem_features']
                maskmem_enc = unconditioned_mems[idx]["maskmem_pos_enc"][-1]
                feats = feats.flatten(2).permute(2, 0, 1)
                maskmem_enc = maskmem_enc.flatten(2).permute(2, 0, 1)
                temporal_idx = len(self.maskmem_tpos_enc) - abs(current_idx - idx) - 1
                maskmem_enc = (maskmem_enc + self.maskmem_tpos_enc[temporal_idx])

                to_cat_memory.append(feats)
                to_cat_memory_pos_embed.append(maskmem_enc)
            
            # 3. add cond object pointers
            pos_and_ptrs = []
            for idx, memory in conditioned_mems.items():
                obj_ptr = memory['obj_ptr']
                if obj_ptr is None:
                    continue
                # fixed_slice = memory['fixed_slice']
                # start = fixed_slice*1024
                # end = fixed_slice*1024
                pos_and_ptrs.append((0, obj_ptr))


            # 4. add non cond object pointers
            num_frames = 8
            max_obj_ptrs_in_encoder = min(num_frames, self.max_obj_ptrs_in_encoder)
            non_cond_idx = [idx for idx in unconditioned_mems.keys() if np.abs(idx - current_idx) <= max_obj_ptrs_in_encoder]
            for idx in non_cond_idx:
                obj_ptr = unconditioned_mems[idx]['obj_ptr']
                # fixed_slice = unconditioned_mems[idx]['fixed_slice']
                # start = fixed_slice*1024
                # end = fixed_slice*1024
                pos = abs(current_idx - idx)
                pos_and_ptrs.append((pos, obj_ptr))



            if len(pos_and_ptrs) > 0:
                # start_pos, end_pos, ptrs_list = zip(*pos_and_ptrs)
                pos_list, ptrs_list = zip(*pos_and_ptrs)
                obj_ptrs = torch.stack(ptrs_list, dim=0)
                # obj_start_pos = torch.tensor(start_pos).to(device=device, non_blocking=True)
                # obj_end_pos = torch.tensor(end_pos).to(device=device, non_blocking=True)
                tpos_dim = C if self.proj_tpos_enc_in_obj_ptrs else self.mem_dim
                t_diff_max = max_obj_ptrs_in_encoder - 1
                obj_pos = torch.tensor(pos_list).to(
                    device=device, non_blocking=True
                )
                # obj_start_pos = get_1d_sine_pe(obj_start_pos, dim=tpos_dim / 2)
                # obj_end_pos = get_1d_sine_pe(obj_end_pos, dim=tpos_dim / 2)
                #obj_pos = torch.cat((obj_start_pos, obj_end_pos), dim=1)
                obj_pos = get_1d_sine_pe(obj_pos / t_diff_max, dim=tpos_dim)
                obj_pos = self.obj_ptr_tpos_proj(obj_pos)
                obj_pos = obj_pos.unsqueeze(1).expand(-1, B, self.mem_dim)

                if self.mem_dim < C:
                    # split a pointer into (C // self.mem_dim) tokens for self.mem_dim < C
                    obj_ptrs = obj_ptrs.reshape(
                        -1, B, C // self.mem_dim, self.mem_dim
                    )
                    obj_ptrs = obj_ptrs.permute(0, 2, 1, 3).flatten(0, 1)
                    obj_pos = obj_pos.repeat_interleave(C // self.mem_dim, dim=0)


                to_cat_memory.append(obj_ptrs)
                to_cat_memory_pos_embed.append(obj_pos)
                num_obj_ptr_tokens = obj_ptrs.shape[0]

        memory = torch.cat(to_cat_memory, dim=0)
        memory_pos_embed = torch.cat(to_cat_memory_pos_embed, dim=0)
        pix_feat_with_mem = self.memory_attention(
            curr=current_vision_feats,
            curr_pos=current_vision_pos_embeds,
            memory=memory,
            memory_pos=memory_pos_embed,
            num_obj_ptr_tokens=num_obj_ptr_tokens,
        )
        # reshape the output (HW)BC => BCHW
        pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
        return pix_feat_with_mem
    
    def process_slice(self, 
                    idx,
                    mem_idx,
                    is_init_slice,
                    backbone_out,
                    vision_feats, 
                    vision_pos_embeds, 
                    feat_sizes, 
                    conditioned_mems, 
                    unconditioned_mems,
                    use_prompt):
        current_vision_feats = [x[:, [idx]] for x in vision_feats]
        current_vision_pos_embeds = [x[:,[idx]] for x in vision_pos_embeds]

        if len(current_vision_feats) > 1:
            high_res_features = [
                x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                for x, s in zip(current_vision_feats[:-1], feat_sizes[:-1])
            ]
        else:
            high_res_features = None
        pix_feat = self.get_memory_conditioned_features(
                            current_idx=mem_idx,
                            is_init_slice=is_init_slice,
                            current_vision_feats=current_vision_feats[-1:],
                            current_vision_pos_embeds=current_vision_pos_embeds[-1:],
                            feat_sizes=feat_sizes[-1:],
                            conditioned_mems=conditioned_mems,
                            unconditioned_mems=unconditioned_mems)
        
        point_inputs = None
        mask_inputs = None

        if use_prompt:
            if len(backbone_out["point_inputs_per_frame"].keys()) > 0:
                point_inputs = backbone_out["point_inputs_per_frame"][0]
            if len(backbone_out["mask_inputs_per_frame"].keys()) > 0:
                mask_inputs = backbone_out["mask_inputs_per_frame"][0]

        multimask_output = self._use_multimask(is_init_slice, point_inputs)
        multimask_output = False
        sam_outputs = self._forward_sam_heads(
            backbone_features=pix_feat,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            high_res_features=high_res_features,
            multimask_output=multimask_output,
        )
        gt_mask = False
        (_,_,_,low_res_masks,high_res_masks,obj_ptr,object_score_logits,) = sam_outputs
        if mask_inputs is not None:
            high_res_masks = mask_inputs
        #high_res_masks = high_res_masks > 0
        # encode new memory

        if not self.use_mem_cond_mems:
            pix_feat = None
        
        maskmem_features, maskmem_pos_enc = self._encode_new_memory(current_vision_feats=current_vision_feats, 
                                current_high_res_mem_features=None,
                                feat_sizes=feat_sizes,
                                pred_masks_high_res=high_res_masks,
                                object_score_logits=object_score_logits,
                                is_mask_from_pts=(point_inputs is not None),
                                pix_feat=pix_feat)
                

        memory = {'maskmem_features': maskmem_features, 
                    'maskmem_pos_enc': maskmem_pos_enc,
                    'obj_ptr': obj_ptr}
        

        pred_mask = high_res_masks > 0

        if idx == 0 and (mask_inputs is not None or point_inputs is not None):
            conditioned_mems[mem_idx] = memory
        else:
            unconditioned_mems[mem_idx] = memory

            all_mem_idx = unconditioned_mems.keys()
            remove_idx = [j for j in all_mem_idx if np.abs(j-mem_idx) > 8]
            for j in remove_idx:
                unconditioned_mems.pop(j)

        return pred_mask, conditioned_mems, unconditioned_mems

    def create_prediction(self, prompt_slice, image_data, device, mask_prompt, point_prompt, center=None):
        conditioned_mems = {}
        unconditioned_mems = {}
        N, C, W, H = image_data.shape
        pred_volume = torch.zeros(size=(N, 1, H, W))

        end = False
        # forward pass
        for i in range(prompt_slice, N, 8):
            max_slice = min(N, i+8)
            image_chunk = image_data[i:max_slice]
            backbone_out = self.encode_volume(image_chunk, device)
            backbone_out["mask_inputs_per_frame"] = {} 
            backbone_out["point_inputs_per_frame"] = {}
            if i == prompt_slice:
                if mask_prompt is not None:
                    backbone_out["mask_inputs_per_frame"][0] = mask_prompt
                if point_prompt is not None:
                    backbone_out["point_inputs_per_frame"][0] = point_prompt
            (_, vision_feats, vision_pos_embeds, feat_sizes) = self._prepare_backbone_features(backbone_out)

            for slice_idx in range(len(image_chunk)):
                global_slice_idx = slice_idx + i
                use_prompt = global_slice_idx == prompt_slice
                if use_prompt:
                    mem_idx = 0
                else:
                    mem_idx = global_slice_idx
                is_init_slice = use_prompt
                
                pred_slice, conditioned_mems, unconditioned_mems = self.process_slice(
                                                        idx=slice_idx,
                                                        mem_idx=mem_idx,
                                                        is_init_slice=is_init_slice,
                                                        backbone_out=backbone_out,
                                                        vision_feats=vision_feats,
                                                        vision_pos_embeds=vision_pos_embeds,
                                                        feat_sizes=feat_sizes,
                                                        conditioned_mems=conditioned_mems,
                                                        unconditioned_mems=unconditioned_mems,
                                                        use_prompt=use_prompt)
                pred_volume[global_slice_idx] = pred_slice
                if torch.all(pred_slice == 0):
                    end = True
                    break

            if end:
                break

        unconditioned_mems = {}
        end = False
        #backward pass
        for i in range(prompt_slice-1, 0, -8):
            min_slice = max(0, i-8)
            
            image_chunk = torch.flip(image_data[min_slice+1:i+1], dims=[0])

            backbone_out = self.encode_volume(image_chunk, device)
            backbone_out["mask_inputs_per_frame"] = {} 
            backbone_out["point_inputs_per_frame"] = {}
            if i == prompt_slice:
                if mask_prompt is not None:
                    backbone_out["mask_inputs_per_frame"][0] = mask_prompt
                if point_prompt is not None:
                    backbone_out["point_inputs_per_frame"][0] = point_prompt
            (_, vision_feats, vision_pos_embeds, feat_sizes) = self._prepare_backbone_features(backbone_out)

            for slice_idx in range(len(image_chunk)):
                global_slice_idx = i-slice_idx
                pred_slice, conditioned_mems, unconditioned_mems = self.process_slice(
                                                        idx=slice_idx,
                                                        mem_idx=global_slice_idx,
                                                        is_init_slice=False,
                                                        backbone_out=backbone_out,
                                                        vision_feats=vision_feats,
                                                        vision_pos_embeds=vision_pos_embeds,
                                                        feat_sizes=feat_sizes,
                                                        conditioned_mems=conditioned_mems,
                                                        unconditioned_mems=unconditioned_mems,
                                                        use_prompt=False)
                pred_volume[global_slice_idx] = pred_slice
                
                if torch.all(pred_slice == 0):
                    end = True
                    break
            if end:
                break
        #pred_volume = self.keep_largest_3d_component(pred_volume)
        if mask_prompt is not None:
            pred_volume[prompt_slice][0] = mask_prompt.squeeze()
        pred_volume = (pred_volume > 0).float()
        return pred_volume