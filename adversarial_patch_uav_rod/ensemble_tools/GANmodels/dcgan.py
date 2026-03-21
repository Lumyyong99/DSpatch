import torch
import torch.nn as nn
import math
import torch.nn.parallel

### -----------------------------------------------------------   Discriminators   ---------------------------------------------------------------------- ###
class DCGAN_D(nn.Module):
    def __init__(self, isize, nz, nc, ndf, ngpu, n_extra_layers=0):
        super(DCGAN_D, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        main = nn.Sequential()
        # input is nc x isize x isize
        main.add_module('initial:{0}-{1}:conv'.format(nc, ndf),
                        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))        # nn.Conv2d(input_channel, output_channel, kernel_size, stride, pad, bias)
        main.add_module('initial:{0}:relu'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = isize / 2, ndf

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}:{1}:conv'.format(t, cndf),
                            nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}:{1}:batchnorm'.format(t, cndf),
                            nn.BatchNorm2d(cndf))
            main.add_module('extra-layers-{0}:{1}:relu'.format(t, cndf),
                            nn.LeakyReLU(0.2, inplace=True))

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid:{0}-{1}:conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid:{0}:batchnorm'.format(out_feat),
                            nn.BatchNorm2d(out_feat))
            main.add_module('pyramid:{0}:relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2

        # state size. K x 4 x 4
        main.add_module('final:{0}-{1}:conv'.format(cndf, 1),
                        nn.Conv2d(cndf, 1, 4, 1, 0, bias=False))
        self.main = main

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        output = output.mean(0)
        return output.view(1)

class DCGAN_D_Rect(nn.Module):
    """
    处理非正方形输入图像，eg. input image size torch.Size[batchsize, 3, 96, 48]
    """
    def __init__(self, isize, nz, nc, ndf, ngpu, n_extra_layers=0):     # 注意nz没作用的
        super(DCGAN_D_Rect, self).__init__()
        self.ngpu = ngpu
        assert isize[0] % 16 == 0 and isize[1] % 16 == 0, "isize has to be a multiple of 16"

        main = nn.Sequential()
        # input is nc x isize[0] x isize[1]
        main.add_module(f'initial:{nc}-{ndf}:conv',nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        main.add_module(f'initial:{ndf}:relu',nn.LeakyReLU(0.2, inplace=True))

        csize = [isize[0] // 2, isize[1] // 2]
        cndf = ndf

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module(f'extra-layers-{t}:{cndf}:conv',nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
            main.add_module(f'extra-layers-{t}:{cndf}:batchnorm',nn.BatchNorm2d(cndf))
            main.add_module(f'extra-layers-{t}:{cndf}:relu',nn.LeakyReLU(0.2, inplace=True))

        while min(csize[0], csize[1]) > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module(f'pyramid:{in_feat}-{out_feat}:conv',nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module(f'pyramid:{out_feat}:batchnorm',nn.BatchNorm2d(out_feat))
            main.add_module(f'pyramid:{out_feat}:relu',nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = [csize[0] // 2, csize[1] // 2]
        
        # final layer, 一个卷积层，但是该卷积层的卷积核大小与csize相关
        main.add_module(f'final:{cndf}-{1}:conv',nn.Conv2d(cndf, 1, (csize[0], csize[1]), 1, 0, bias=False))
        self.main = main

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        output = output.mean(0)
        return output.view(1)


class DCGAN_D_nobn(nn.Module):
    def __init__(self, isize, nz, nc, ndf, ngpu, n_extra_layers=0):
        super(DCGAN_D_nobn, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        main = nn.Sequential()
        # input is nc x isize x isize
        # input is nc x isize x isize
        main.add_module('initial:{0}-{1}:conv'.format(nc, ndf),
                        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        main.add_module('initial:{0}:conv'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = isize / 2, ndf

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}:{1}:conv'.format(t, cndf),
                            nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}:{1}:relu'.format(t, cndf),
                            nn.LeakyReLU(0.2, inplace=True))

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid:{0}-{1}:conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid:{0}:relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2

        # state size. K x 4 x 4
        main.add_module('final:{0}-{1}:conv'.format(cndf, 1),
                        nn.Conv2d(cndf, 1, 4, 1, 0, bias=False))
        self.main = main

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        output = output.mean(0)
        return output.view(1)




### -----------------------------------------------------------   Generators   ---------------------------------------------------------------------- ###
class DCGAN_G(nn.Module):
    def __init__(self, isize, nz, nc, ngf, ngpu, n_extra_layers=0):
        super(DCGAN_G, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf // 2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential()
        # input is Z, going into a convolution
        main.add_module('initial:{0}-{1}:convt'.format(nz, cngf),
                        nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False))
        main.add_module('initial:{0}:batchnorm'.format(cngf),
                        nn.BatchNorm2d(cngf))
        main.add_module('initial:{0}:relu'.format(cngf),
                        nn.ReLU(True))

        csize, cndf = 4, cngf
        while csize < isize // 2:
            main.add_module('pyramid:{0}-{1}:convt'.format(cngf, cngf // 2),
                            nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1, bias=False))
            main.add_module('pyramid:{0}:batchnorm'.format(cngf // 2),
                            nn.BatchNorm2d(cngf // 2))
            main.add_module('pyramid:{0}:relu'.format(cngf // 2),
                            nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}:{1}:conv'.format(t, cngf),
                            nn.Conv2d(cngf, cngf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}:{1}:batchnorm'.format(t, cngf),
                            nn.BatchNorm2d(cngf))
            main.add_module('extra-layers-{0}:{1}:relu'.format(t, cngf),
                            nn.ReLU(True))

        main.add_module('final:{0}-{1}:convt'.format(cngf, nc),
                        nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
        main.add_module('final:{0}:tanh'.format(nc),
                        nn.Tanh())
        self.main = main

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

class DCGAN_G_Rect(nn.Module):
    # TODO: 这里有问题，要修改！
    def __init__(self, isize, nz, nc, ngf, ngpu, n_extra_layers=0):
        super(DCGAN_G_Rect, self).__init__()
        self.ngpu = ngpu
        assert isize[0] % 16 == 0 and isize[1] % 16 == 0 # isize [h, w] has to be a multiple of 16, 这里限制w:h必须是一个整数, 输入顺序是isize[h,w]

        r = isize[0] / isize[1]
        cngf, tisize_h, tisize_w = ngf // 2, 4 * r, 4
        while tisize_w != isize[1]:     # 刚开始tisize_w相当于从init层出来的张量尺寸
            cngf = cngf * 2
            tisize_h = tisize_h * 2
            tisize_w = tisize_w * 2

        main = nn.Sequential()
        # input is Z, going into a convolution
        main.add_module(f'initial:{nz}-{cngf}:convt', nn.ConvTranspose2d(nz, cngf, (int(r*4), 4), 1, 0, bias=False))
        main.add_module(f'initial:{cngf}:batchnorm', nn.BatchNorm2d(cngf))
        main.add_module(f'initial:{cngf}:relu', nn.ReLU(True))

        csize_h, csize_w = 4, 4 * r     # 这里时init出来之后的张量尺寸，定下来短边就是4
        while csize_h < isize[1] // 2:
            main.add_module(f'pyramid:{cngf}-{cngf // 2}:convt', nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1, bias=False))
            main.add_module(f'pyramid:{cngf // 2}:batchnorm', nn.BatchNorm2d(cngf // 2))
            main.add_module(f'pyramid:{cngf // 2}:relu', nn.ReLU(True))
            
            cngf = cngf // 2
            csize_h = csize_h * 2
            csize_w = csize_w * 2

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}:{1}:conv'.format(t, cngf),
                            nn.Conv2d(cngf, cngf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}:{1}:batchnorm'.format(t, cngf),
                            nn.BatchNorm2d(cngf))
            main.add_module('extra-layers-{0}:{1}:relu'.format(t, cngf),
                            nn.ReLU(True))

        main.add_module('final:{0}-{1}:convt'.format(cngf, nc),
                        nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
        main.add_module('final:{0}:tanh'.format(nc),
                        nn.Tanh())
        self.main = main

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class DCGAN_G_Rect_2(nn.Module):
    def __init__(self, isize, nz, nc, ngf, ngpu, n_extra_layers=0):
        super(DCGAN_G_Rect_2, self).__init__()
        self.ngpu = ngpu
        assert isize[0] % 16 == 0 and isize[1] % 16 == 0  # isize [h, w] has to be a multiple of 16

        # 计算长宽比
        h, w = isize[0], isize[1]
        
        # 计算初始特征图大小和通道数
        cngf = ngf
        # 从目标尺寸反向计算初始尺寸
        init_h, init_w = h // 16, w // 16  # 对于64x192，初始尺寸为4x12

        main = nn.Sequential()
        # 初始转置卷积层，将噪声向量转换为初始特征图
        main.add_module(f'initial:{nz}-{cngf}:convt', 
                       nn.ConvTranspose2d(nz, cngf, (init_h, init_w), 1, 0, bias=False))
        main.add_module(f'initial:{cngf}:batchnorm', nn.BatchNorm2d(cngf))
        main.add_module(f'initial:{cngf}:relu', nn.ReLU(True))

        csize_h, csize_w = init_h, init_w
        
        # 上采样层
        while csize_h < isize[0] // 2:  # 进行4次上采样，每次尺寸翻倍
            main.add_module(f'pyramid:{cngf}-{cngf // 2}:convt', 
                          nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1, bias=False))
            main.add_module(f'pyramid:{cngf // 2}:batchnorm', nn.BatchNorm2d(cngf // 2))
            main.add_module(f'pyramid:{cngf // 2}:relu', nn.ReLU(True))
            
            cngf = cngf // 2
            csize_h = csize_h * 2
            csize_w = csize_w * 2

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module(f'extra-layers-{t}:{cngf}:conv',
                          nn.Conv2d(cngf, cngf, 3, 1, 1, bias=False))
            main.add_module(f'extra-layers-{t}:{cngf}:batchnorm',
                          nn.BatchNorm2d(cngf))
            main.add_module(f'extra-layers-{t}:{cngf}:relu',
                          nn.ReLU(True))

        # 最终输出层
        main.add_module(f'final:{cngf}-{nc}:convt',
                       nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
        main.add_module(f'final:{nc}:tanh', nn.Tanh())
        self.main = main

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

class DCGAN_G_Rect_2_gpu(nn.Module):
    def __init__(self, isize, nz, nc, ngf, ngpu, n_extra_layers=0):
        super(DCGAN_G_Rect_2_gpu, self).__init__()
        self.ngpu = ngpu
        self.device = torch.device("cuda" if torch.cuda.is_available() and ngpu > 0 else "cpu")
        assert isize[0] % 16 == 0 and isize[1] % 16 == 0  # isize [h, w] has to be a multiple of 16

        # 计算长宽比
        h, w = isize[0], isize[1]
        
        # 计算初始特征图大小和通道数
        cngf = ngf
        # 从目标尺寸反向计算初始尺寸
        init_h, init_w = h // 16, w // 16  # 对于64x192，初始尺寸为4x12

        main = nn.Sequential()
        # 初始转置卷积层，将噪声向量转换为初始特征图
        main.add_module(f'initial:{nz}-{cngf}:convt', 
                       nn.ConvTranspose2d(nz, cngf, (init_h, init_w), 1, 0, bias=False))
        main.add_module(f'initial:{cngf}:batchnorm', nn.BatchNorm2d(cngf))
        main.add_module(f'initial:{cngf}:relu', nn.ReLU(True))

        csize_h, csize_w = init_h, init_w
        
        # 上采样层
        while csize_h < isize[0] // 2:  # 进行4次上采样，每次尺寸翻倍
            main.add_module(f'pyramid:{cngf}-{cngf // 2}:convt', 
                          nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1, bias=False))
            main.add_module(f'pyramid:{cngf // 2}:batchnorm', nn.BatchNorm2d(cngf // 2))
            main.add_module(f'pyramid:{cngf // 2}:relu', nn.ReLU(True))
            
            cngf = cngf // 2
            csize_h = csize_h * 2
            csize_w = csize_w * 2

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module(f'extra-layers-{t}:{cngf}:conv',
                          nn.Conv2d(cngf, cngf, 3, 1, 1, bias=False))
            main.add_module(f'extra-layers-{t}:{cngf}:batchnorm',
                          nn.BatchNorm2d(cngf))
            main.add_module(f'extra-layers-{t}:{cngf}:relu',
                          nn.ReLU(True))

        # 最终输出层
        main.add_module(f'final:{cngf}-{nc}:convt',
                       nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
        main.add_module(f'final:{nc}:tanh', nn.Tanh())
        self.main = main

        # load to device
        self.to(self.device)

    def forward(self, input):

        input = input.to(self.device)
        
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output



class DCGAN_G_CustomAspectRatio(nn.Module):
    def __init__(self, isize, nz, nc, ngf, ngpu, n_extra_layers=0):
        super(DCGAN_G_CustomAspectRatio, self).__init__()
        self.ngpu = ngpu
        assert isize[0] % 16 == 0 and isize[1] % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize_w= ngf // 2, 8
        csize_w = isize[0]
        while tisize_w != csize_w:
            cngf = cngf * 2
            tisize_w = tisize_w * 2

        main = nn.Sequential()
        # input is Z, going into a convolution
        main.add_module(f'initial:{nz}-{cngf}:convt',nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False))
        main.add_module(f'initial:{cngf}:batchnorm',nn.BatchNorm2d(cngf))
        main.add_module(f'initial:{cngf}:relu',nn.ReLU(True))

        csize_w, csize_h, cndf = 8, 3, cngf
        while csize_w < isize[0] // 2:   # Make sure we stop when width is close to isize
            main.add_module(f'pyramid:{cngf}-{cngf // 2}:convt', nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1, bias=False))
            main.add_module(f'pyramid:{cngf // 2}:batchnorm', nn.BatchNorm2d(cngf // 2))
            main.add_module(f'pyramid:{cngf // 2}:relu', nn.ReLU(True))
            cngf = cngf // 2
            csize_w = csize_w * 2

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module(f'extra-layers-{t}:{cngf}:conv', nn.Conv2d(cngf, cngf, 3, 1, 1, bias=False))
            main.add_module(f'extra-layers-{t}:{cngf}:batchnorm', nn.BatchNorm2d(cngf))
            main.add_module(f'extra-layers-{t}:{cngf}:relu', nn.ReLU(True))

        main.add_module(f'final:{cngf}-{nc}:convt', nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
        main.add_module(f'final:{nc}:tanh',nn.Tanh())
        self.main = main

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class DCGAN_G_nobn(nn.Module):
    def __init__(self, isize, nz, nc, ngf, ngpu, n_extra_layers=0):
        super(DCGAN_G_nobn, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf // 2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential()
        main.add_module('initial:{0}-{1}:convt'.format(nz, cngf),
                        nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False))
        main.add_module('initial:{0}:relu'.format(cngf),
                        nn.ReLU(True))

        csize, cndf = 4, cngf
        while csize < isize // 2:
            main.add_module('pyramid:{0}-{1}:convt'.format(cngf, cngf // 2),
                            nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1, bias=False))
            main.add_module('pyramid:{0}:relu'.format(cngf // 2),
                            nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}:{1}:conv'.format(t, cngf),
                            nn.Conv2d(cngf, cngf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}:{1}:relu'.format(t, cngf),
                            nn.ReLU(True))

        main.add_module('final:{0}-{1}:convt'.format(cngf, nc),
                        nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
        main.add_module('final:{0}:tanh'.format(nc),
                        nn.Tanh())
        self.main = main

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


if __name__ == '__main__':
    rowPatch_size_rect_2 = [48, 64]        # 输入尺寸顺序为(w, h)
    nc = 3
    ndf = 64
    ngf = 64
    n_extra_layers = 0

    nz = 100
    ngpu = 1
    # test_tensor_rect = torch.rand(8, 3, 64, 192)        # discriminator输入张量，尺寸顺序为(w, h)
    # D_rect = DCGAN_D_Rect(rowPatch_size_rect_2, nz, nc, ndf, ngpu, n_extra_layers)
    # out_dcgan_rect = D_rect(test_tensor_rect)
    # print('output_dcgan_rect:', out_dcgan_rect)

    noise = torch.FloatTensor(1, nz, 1, 1)
    G = DCGAN_G_Rect_2(rowPatch_size_rect_2, nz, nc, ngf, ngpu, n_extra_layers)
    output_G = G(noise)
    print('output origin G shape:', output_G.shape)     # generator输出张量为torch.Size([1, 3, 128, 64])，输出张量也是[w, h]的顺序
