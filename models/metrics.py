from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.utils import make_grid
from torchvision.utils import save_image
import torch
import torchvision


def unnormalize(x):
    x = x.transpose(1, 3)
    # mean, std
    x = x * torch.Tensor((0.5, )) + torch.Tensor((0.5, ))
    x = x.transpose(1, 3)
    return x


class Evaluator:
    def __init__(self, model):
        self.model = model
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.fid = FrechetInceptionDistance(feature=64)

    def evaluate(self, dataset, device, filename):
        image, mask, gt = zip(*[dataset[i] for i in range(8)])

        c = image[0].size(dim=0)
        h = image[0].size(dim=1)
        w = image[0].size(dim=2)
        # print('c = {}, h = {}, w = {}'.format(c, h, w))

        image = torch.stack(image)
        mask = torch.stack(mask)
        gt = torch.stack(gt)

        # float2uint8 = torchvision.transforms.ConvertImageDtype(torch.uint8)

        with torch.no_grad():
            output, _, _ = self.model(image.to(device), mask.to(device), gt)
        # output = output.to(torch.device('cpu'))
        output_comp = mask * image + (1 - mask) * output

        # reverse for display image
        image = mask * unnormalize(image) + (1 - mask)
        mask = (1 - mask)

        grid = make_grid(torch.cat((mask,
                                    unnormalize(output),
                                    image,
                                    unnormalize(output_comp),
                                    unnormalize(gt)
                                    ),
                                   dim=0))
        save_image(grid, filename)

        output = output.reshape(8, c, h, w)
        mask = mask.reshape(8, c, h, w)
        gt = gt.reshape(8, c, h, w)
        metrics = {'PSNR': 0.0, 'SSIM': 0.0, 'FID': 0.0}

        for op, m, g in zip(output, mask, gt):
            mask_w = int(torch.sum(m == 1).item() / (h * c))

            metrics['PSNR'] += self.psnr(op[m == 1].reshape(c, h, mask_w),
                                         g[m == 1].reshape(c, h, mask_w)).item()

            metrics['SSIM'] += self.ssim(op[m == 1].reshape(1, c, h, mask_w),
                                         g[m == 1].reshape(1, c, h, mask_w)).item()

            # self.fid.update(float2uint8.forward(g[:, :, int(w / 2):]).reshape(1, c, h, int(w / 2)), real=True)
            # self.fid.update(float2uint8.forward(op[:, :, int(w / 2):]).reshape(1, c, h, int(w / 2)), real=False)
            # metrics['FID'] += self.fid.compute().item()

        metrics['SSIM'] += self.ssim(output[mask == 1].reshape(8, c, h, int(torch.sum(mask[0] == 1).item() / (h * c))),
                                     gt[mask == 1].reshape(8, c, h, int(torch.sum(mask[0] == 1).item() / (h * c))))

        metrics['PSNR'] /= 8
        # metrics['SSIM'] /= 8
        metrics['FID'] /= 8

        return metrics


