import torch
import torch.nn.functional as F


class IdentityTrainer:
    def __init__(
            self,
            model, mean, std,
            img, img_size,
            c_weight, c2_weight):
        self.model = model
        self.mean = mean
        self.std = std
        self.img_size = img_size
        self.c_weight = c_weight
        self.c2_weight = c2_weight
        with torch.no_grad():
            code_true, content_true = self.pred(img)
        self.code_true = code_true
        self.content_true = content_true

    def pred(self, img):
        img_true = img.clone()
        img_true -= self.mean
        img_true /= self.std
        img_true = F.interpolate(img_true, size=self.img_size, mode='area')
        return self.model(img_true)


def load_img(path):
    from PIL import Image
    import torchvision.transforms as tfs
    image_size = 112
    device = "cuda"
    orig = Image.open(path).convert("RGB")
    img = orig.resize((image_size, image_size))
    return tfs.ToTensor()(img).to(device).unsqueeze(0).repeat(1, 1, 1, 1)


def test(img_path, img_path2):
    from torch.autograd import Variable
    device = "cuda"
    img = load_img(img_path)
    img2 = load_img(img_path2)
    combined = img.clone()
    combined[img2 != 0] = img2[img2 != 0]
    import insightface
    insightface_model = insightface.iresnet100(pretrained=True)
    insightface_model.eval()
    insightface_model.to(device)
    if_mean = torch.tensor([0.5] * 3).type(torch.float32).to(device).view([1, 3, 1, 1])
    if_std = torch.tensor([0.5 * 256.0 / 255.0] * 3).type(torch.float32).to(device).view([1, 3, 1, 1])

    insightface_trainer = IdentityTrainer(insightface_model, if_mean, if_std, img, 112, 1.0, 1.0)
    code_fake, _ = insightface_trainer.pred(combined)
    code_true = insightface_trainer.code_true
    print(torch.nn.CosineEmbeddingLoss()(code_fake, Variable(code_true, requires_grad=False), torch.ones_like(code_true[:, 0])).item())


if __name__ == "__main__":
    test(
        "C:/Users/user/Desktop/Desktop/temp/20210202_face_fitting_demo/2019/001358.png",
        "C:/Users/user/Desktop/Desktop/temp/20210202_face_fitting_demo/2019/001358_recon.png"
    )
