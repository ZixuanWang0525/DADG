import spacy
import torch
import torch.nn as nn

from transformers import CLIPTokenizer, CLIPModel

import torch.nn.functional as F
import sng_parser
import re

from utils.open_vocabulary_detection import GradientDetection


class DenseConceptAlignmentLoss(nn.Module):
    def __init__(
        self,
        device: str = "cuda",
        base_models=None,
    ):
        super(DenseConceptAlignmentLoss, self).__init__()

        # en_core_web_sm has no word vector
        self.sentence_process_model = spacy.load("en_core_web_md")
        self.scene_graph_parser = sng_parser.Parser("spacy", model="en_core_web_md")

        # CLIP (VIT-Large in Original StableDiffusion)
        self.CLIPModel = base_models.CLIPModel
        self.CLIPTokenizer = base_models.CLIPTokenizer

        self.device = device
        self.gradient_detector = base_models.gradient_detector

        # hyperparameter
        # according to ddim step 50
        self.fine_grained_guidance = 5

    def _calc_clip_score(self, local_global_image, local_global_language):
        device = local_global_image.device
        c, h, w = local_global_image.shape
        # clip process
        # image process
        # rgb (unnecessary)
        # resize + crop
        clip_image = local_global_image
        if h != 224 or w != 224:
            clip_image = F.interpolate(clip_image.unsqueeze(0), size=(224, 224), mode="bicubic").squeeze(0)
        # normalization
        clip_image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device)
        clip_image_standard = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device)
        if clip_image.ndimension() == 3 and clip_image.shape[0] in [1, 3]:
            clip_image = (clip_image - clip_image_mean[:, None, None]) / clip_image_standard[:, None, None]
        else:
            clip_image = (clip_image - clip_image_mean) / clip_image_standard
        pixel_values = clip_image.unsqueeze(0)
        # language process
        clip_language_dictionary = self.CLIPTokenizer(local_global_language, return_tensors="pt")
        input_ids = clip_language_dictionary["input_ids"].to(device)
        attention_mask = clip_language_dictionary["attention_mask"].to(device)

        #
        self.CLIPModel = self.CLIPModel.to(device)
        CLIPOutput = self.CLIPModel(input_ids, pixel_values, attention_mask)
        # temperature
        similarity_scale_divide = torch.tensor(100.).to(device)
        CLIPSimilarity = CLIPOutput.logits_per_image
        # print(CLIPSimilarity)
        CLIPSimilarity = - torch.log(CLIPSimilarity / similarity_scale_divide)

        return CLIPSimilarity

    def _dependency_tree_analysis(self, sentence):
        scene_graph = self.scene_graph_parser.parse(sentence)
        # head (instance), lemma_span (instance + attribute)
        entities = scene_graph["entities"]
        # remove article
        for i in range(len(entities)):
            lemma_span = entities[i]["lemma_span"]
            modify_pattern = r'\b(a|an|the)\b'
            new_lemma_span = re.sub(modify_pattern, '', lemma_span, flags=re.IGNORECASE)
            new_lemma_span = re.sub(r'\s+', ' ', new_lemma_span).strip()
            entities[i]["lemma_span"] = new_lemma_span
        relations = scene_graph["relations"]
        return entities, relations
    
    def open_vocabulary_detection(self, recons_image, class_names):
        image_batch = recons_image.unsqueeze(0)   # [1,3,H,W]

        det = self.gradient_detector.forward(
            image_batch,
            class_names,
            mode="detection"
        )

        boxes = det["boxes"]    # [1,N,4]
        scores = det["scores"]  # [1,N]

        if boxes.shape[1] == 0:
            return None

        best_idx = torch.argmax(scores[0]).item()
        best_box = boxes[0, best_idx]
        return best_box

    def _black_outside_rectangle(self, recons_image, left_top_x, left_top_y, right_bottom_x, right_bottom_y):
        # (0, 0, 0) black
        black_outside_mask = torch.zeros_like(recons_image)
        black_outside_mask[:, left_top_y: right_bottom_y, left_top_x: right_bottom_x] = 1
        sub_recons_image = recons_image * black_outside_mask
        return sub_recons_image

    def _overlay_images(self, recons_image, nsubj_image, dobj_image):
        # torch.ne not equal to
        mask_nsubj = torch.ne(nsubj_image, 0)
        mask_dobj = torch.ne(dobj_image, 0)
        mask_nsubj_dobj = torch.logical_or(mask_nsubj, mask_dobj)
        mask_nsubj_dobj = mask_nsubj_dobj.int()

        image_nsubj_dobj = recons_image * mask_nsubj_dobj
        return image_nsubj_dobj
    
    def _create_subimage_entities(self, entities, recons_image):
        subimage_entities = {}

        for k, entity in enumerate(entities):
            entity_description_subimage = {}
            
            if entity["head"] == entity["lemma_span"]:
                # no attribute
                entity_name_attribute = entity["head"] + ""
            else:
                entity_name_attribute = entity["lemma_span"]

            detection_prompts = [entity["head"]]

            best_box = self.open_vocabulary_detection(recons_image, detection_prompts)
            if best_box is None:
                continue
            
            left_top_x, left_top_y, right_bottom_x, right_bottom_y = best_box
            left_top_x = int(left_top_x)
            left_top_y = int(left_top_y)
            right_bottom_x = int(right_bottom_x)
            right_bottom_y = int(right_bottom_y)
            entity_description_subimage["description"] = entity_name_attribute
            entity_description_subimage["subimage"] = self._black_outside_rectangle(
                recons_image, 
                left_top_x,
                left_top_y,
                right_bottom_x, 
                right_bottom_y
            )

            # for relation description
            entity_description_subimage["entity_name"] = entity["head"]
            subimage_entities[k] = entity_description_subimage

        return subimage_entities

    def _create_subimage_relations(self, relations, subimage_entities, recons_image):
        subimages_relations = {}

        for k, relation in enumerate(relations):
            relation_description_subimage = {}
            nsubj_name = subimage_entities[relation["subject"]]["entity_name"]
            relation_name = relation["relation"]
            dobj_name = subimage_entities[relation["object"]]["entity_name"]
            relation_description_subimage["description"] = nsubj_name + " " + relation_name + " " + dobj_name
            relation_description_subimage["subimage"] = self._overlay_images(
                recons_image,
                subimage_entities[relation["subject"]]["subimage"],
                subimage_entities[relation["object"]]["subimage"])
            subimages_relations[k] = relation_description_subimage

        return subimages_relations

    def forward(self, recons_image, language_prompt, timestep, self_recurrent,
                subimages_entities=None, subimages_relations=None):
        # no bs
        # recons_image.shape must (c, h, w)
        device = recons_image.device
        dense_concept_alignment_loss = torch.tensor(0.).to(device)
        scene_loss = torch.tensor(0.).to(device)
        attr_rel_loss = torch.tensor(0.).to(device)
        gamma = torch.tensor(0.3).to(device)

        if timestep > self.fine_grained_guidance:
            if self_recurrent == 1:
                entities, relations = self._dependency_tree_analysis(language_prompt)

                subimages_entities = self._create_subimage_entities(
                    entities,
                    recons_image)
                subimages_relations = self._create_subimage_relations(
                    relations,
                    subimages_entities,
                    recons_image)

            attr_loss = torch.tensor(0.).to(device)
            rel_loss = torch.tensor(0.).to(device)
            if len(subimages_entities) != 0:
                for i in range(len(subimages_entities)):
                    language, image = subimages_entities[i]["description"], subimages_entities[i]["subimage"]
                    attr_loss += self._calc_clip_score(image, language).squeeze(0).squeeze(0)
            if len(subimages_relations) != 0:
                for i in range(len(subimages_relations)):
                    language, image = subimages_relations[i]["description"], subimages_relations[i]["subimage"]
                    rel_loss += self._calc_clip_score(image, language).squeeze(0).squeeze(0)
            attr_rel_loss = (attr_loss + rel_loss) / (len(subimages_entities) + len(subimages_relations))

        # global consistency
        scene_loss = self._calc_clip_score(recons_image, language_prompt)

        dense_concept_alignment_loss = (1 - gamma) * scene_loss + gamma * attr_rel_loss
        return dense_concept_alignment_loss, subimages_entities, subimages_relations