"""Actual DINO helper cuda() mutates in place and returns None; no model change."""
import trellis2_map_learning_train_v1 as base


def image_cond(plan):
    encoder=base.DinoV3FeatureExtractor(base.DINO)
    encoder.cuda()
    encoder.model.eval().requires_grad_(False)
    with base.torch.no_grad():
        c=encoder([base.Image.open(plan['appearance_condition']).convert('RGB')]).detach()
    del encoder
    base.torch.cuda.empty_cache()
    return c


base.image_cond=image_cond
if __name__=='__main__':base.main()
