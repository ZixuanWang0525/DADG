class DenseGeometryAlignment(nn.Module):
    def __init__(self, nms_conf_hyperparam=0.5, nms_iou_hyperparam=0.65):
        super(DenseGeometryAlignment, self).__init__()

        # make segmentation (detection) network
        """
        with open('seg_models/data/hyp.scratch.mask.yaml') as f:
            self.hyp = yaml.load(f, Loader=yaml.FullLoader)
        weigths = torch.load('seg_models/yolov7-mask.pt')
        self.seg_model = weigths['model']
        # float16 -> float32
        self.seg_model = self.seg_model.float().eval()
        """

        # en_core_web_sm has no word vector
        self.sentence_process_model = spacy.load("en_core_web_md")

        """
        self.nms_conf_hyperparam = nms_conf_hyperparam
        self.nms_iou_hyperparam = nms_iou_hyperparam
        """