import torch
import numpy as np
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
from collections import OrderedDict


def find_tensor_peak_batch(heatmap, radius, downsample, threshold = 0.000001):
    assert heatmap.dim() == 3, 'The dimension of the heatmap is wrong : {}'.format(heatmap.size())
    num_pts, H, W = heatmap.size(0), heatmap.size(1), heatmap.size(2)
    assert W > 1 and H > 1, 'To avoid the normalization function divide zero'
    # find the approximate location:
    score, index = torch.max(heatmap.view(num_pts, -1), 1)
    index_w = (index % W).float()
    index_h = (index / float(W)).float()

    def normalize(x, L):
        return -1. + 2. * x.data / (L-1)
    boxes = [index_w - radius, index_h - radius, index_w + radius, index_h + radius]
    boxes[0] = normalize(boxes[0], W)
    boxes[1] = normalize(boxes[1], H)
    boxes[2] = normalize(boxes[2], W)
    boxes[3] = normalize(boxes[3], H)
    #affine_parameter = [(boxes[2]-boxes[0])/2, boxes[0]*0, (boxes[2]+boxes[0])/2,
    #                   boxes[0]*0, (boxes[3]-boxes[1])/2, (boxes[3]+boxes[1])/2]
    #theta = torch.stack(affine_parameter, 1).view(num_pts, 2, 3)

    affine_parameter = torch.zeros((num_pts, 2, 3))
    affine_parameter[:,0,0] = (boxes[2]-boxes[0])/2
    affine_parameter[:,0,2] = (boxes[2]+boxes[0])/2
    affine_parameter[:,1,1] = (boxes[3]-boxes[1])/2
    affine_parameter[:,1,2] = (boxes[3]+boxes[1])/2
    # extract the sub-region heatmap
    theta = affine_parameter.to(heatmap.device)
    grid_size = torch.Size([num_pts, 1, radius*2+1, radius*2+1])
    grid = F.affine_grid(theta, grid_size, align_corners=True)
    sub_feature = F.grid_sample(heatmap.unsqueeze(1), grid, align_corners=True).squeeze(1)
    sub_feature = F.threshold(sub_feature, threshold, np.finfo(float).eps)

    X = torch.arange(-radius, radius+1).to(heatmap).view(1, 1, radius*2+1)
    Y = torch.arange(-radius, radius+1).to(heatmap).view(1, radius*2+1, 1)

    sum_region = torch.sum(sub_feature.view(num_pts,-1),1)
    x = torch.sum((sub_feature*X).view(num_pts,-1),1) / sum_region + index_w
    y = torch.sum((sub_feature*Y).view(num_pts,-1),1) / sum_region + index_h

    x = x * downsample + downsample / 2.0 - 0.5
    y = y * downsample + downsample / 2.0 - 0.5
    return torch.stack([x, y],1)


class VGG16(nn.Module):
    def __init__(self, visualize=False):
        super(VGG16, self).__init__()

        self.config_stage = 3
        self.config_argmax = 4
        self.downsample = 8
        self.pts_num = 69
        self.visualize = visualize

        self.features = nn.Sequential(
            nn.Conv2d(  3,  64, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d( 64,  64, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d( 64, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True))

        self.CPM_feature = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True), #CPM_1
            nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True)) #CPM_2

        assert self.config_stage >= 1, 'stages of cpm must >= 1 not : {:}'.format(self.config_stage)
        stage1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, kernel_size=1, padding=0), nn.ReLU(inplace=True),
            nn.Conv2d(512, self.pts_num, kernel_size=1, padding=0))
        stages = [stage1]
        for i in range(1, self.config_stage):
            stagex = nn.Sequential(
                nn.Conv2d(128+self.pts_num, 128, kernel_size=7, dilation=1, padding=3), nn.ReLU(inplace=True),
                nn.Conv2d(128,              128, kernel_size=7, dilation=1, padding=3), nn.ReLU(inplace=True),
                nn.Conv2d(128,              128, kernel_size=7, dilation=1, padding=3), nn.ReLU(inplace=True),
                nn.Conv2d(128,              128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(128,              128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(128,              128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(128,              128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(128,              128, kernel_size=1, padding=0), nn.ReLU(inplace=True),
                nn.Conv2d(128,     self.pts_num, kernel_size=1, padding=0))
            stages.append( stagex )
        self.stages = nn.ModuleList(stages)

# return : cpm-stages, locations
    def forward(self, inputs):
        assert inputs.dim() == 4, 'This model accepts 4 dimension input tensor: {}'.format(inputs.size())
        batch_size, feature_dim = inputs.size(0), inputs.size(1)
        batch_cpms = []
        batch_locs = None

        feature  = self.features(inputs)
        xfeature = self.CPM_feature(feature)
        for i in range(self.config_stage):
            if i == 0: cpm = self.stages[i]( xfeature )
            else:      cpm = self.stages[i]( torch.cat([xfeature, batch_cpms[i-1]], 1) )
            batch_cpms.append(cpm)
        if self.visualize:
            batch_locs = torch.zeros([batch_size, 68, 2])
            for ibatch in range(batch_size):
                batch_location = find_tensor_peak_batch(batch_cpms[-1][ibatch], self.config_argmax, self.downsample)
                batch_locs[ibatch] = batch_location[:-1, :]
        return torch.cat(batch_cpms, 0), batch_locs


def draw_image_by_points(_image, pts, radius, color):
    from PIL import ImageDraw
    from PIL import ImageFont
    if isinstance(_image, str):
        # _image = datasets.pil_loader(_image)
        _image = Image.open(_image)
        _image = _image.resize((256, 256))
    assert isinstance(_image, Image.Image), 'image type is not PIL.Image.Image'
    assert isinstance(pts, np.ndarray) and (pts.shape[0] == 2 or pts.shape[0] == 3), 'input points are not correct'
    image, pts = _image.copy(), pts.copy()

    num_points = pts.shape[1]
    visiable_points = []
    for idx in range(num_points):
        if pts.shape[0] == 2 or bool(pts[2,idx]):
            visiable_points.append(True)
        else:
            visiable_points.append(False)
    visiable_points = np.array(visiable_points)
    print ('visiable points : {}'.format( np.sum(visiable_points) ))

    draw  = ImageDraw.Draw(image)
    for idx in range(num_points):
        if visiable_points[ idx ]:
            # draw hollow circle
            point = (pts[0,idx]-radius, pts[1,idx]-radius, pts[0,idx]+radius, pts[1,idx]+radius)
        if radius > 0:
            draw.ellipse(point, fill=color, outline=color)

    return image


if __name__ == "__main__":
    import torchvision.transforms as tfs
    from sampling import sample_landmarks
    img_path = "D:/DataSet/datas_1024/images_1024/000002.jpg"
    transform = tfs.Compose([
        tfs.Resize(256),
        tfs.ToTensor(),
        tfs.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    cropped_size = [1024, 1024, 0, 0]
    img = Image.open(img_path)
    inputs = transform(img).unsqueeze(0).cuda()
    weights = torch.load("./assets/SBR_lm.pth")
    net = VGG16(visualize=True).cuda()
    net.load_state_dict(weights)
    with torch.no_grad():
        batch_heatmaps, batch_locs = net(inputs)
    locations = batch_locs.cpu().numpy()[0]
    print(locations.transpose(1, 0).shape)
    image = draw_image_by_points(img_path, locations.transpose(1, 0), 2, (255, 0, 0))
    image.save("./out_256.jpg")
