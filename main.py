import numpy as np
import cv2
import config_file
import torch
import torchvision
from torch.nn.functional import normalize

from yuface import detect


def predict_face_yu(image):
    return detect(image)[1]


def open_img(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


resize = torchvision.transforms.Resize((128, 128))
grayscale = torchvision.transforms.Grayscale(num_output_channels=1)
to_tensor = torchvision.transforms.ToTensor()
to_pil = torchvision.transforms.ToPILImage()


# (3, 128, 128) -> (vector_len)
class DenseBlock(torch.nn.Module):
    def __init__(self, conv_params):
        """
        conv_params: list of torch.nn.Conv2d params representing a sequence for Conv2d modules
        """
        super().__init__()
        self.convs = torch.nn.ModuleList([torch.nn.Conv2d(*i) for i in conv_params])
        self.act = torch.nn.SiLU()

    def forward(self, X):
        X = self.convs[0](X)
        concat_maps = [X]
        for i in range(1, len(self.convs)):
            X = self.act(X)
            resize_t = torchvision.transforms.Resize((X.shape[-1], X.shape[-1]))
            X = torch.cat([torch.mean(torch.stack([resize_t(j) for j in concat_maps]), axis=0), X], axis=1)
            X = self.convs[i](X)
            concat_maps.append(X)
        return X


class ParallelBlock(torch.nn.Module):
    def __init__(self, conv_params, shapes_print=False):
        """
        conv_params: 3-d list of torch.nn.Conv2d params representing a sequence for Conv2d modules
        """
        super().__init__()
        self.convs = torch.nn.ModuleList([torch.nn.ModuleList([DenseBlock(j) for j in i]) for i in conv_params])
        self.act = torch.nn.SiLU()
        self.shapes_print = shapes_print

        # self.convs [path] [block]

    def forward(self, X):
        X_list = [self.convs[i][0](X) for i in range(len(self.convs))]  # first block of every path
        resize_shape = [j.shape[-1] for j in X_list]  # shapes of first block output of every path
        if self.shapes_print:
            print('BLOCK 1:')
            [print(list(i[0].shape), end=' ') for i in X_list]
            print()
        for j in range(1, len(self.convs[0])):
            X = torch.cat([torchvision.transforms.Resize((max(resize_shape), max(resize_shape)))(j) for j in X_list],
                          axis=1)
            X_list = [self.convs[i][j](torchvision.transforms.Resize((resize_shape[i], resize_shape[i]))(self.act(X)))
                      for i in range(len(self.convs))]
            resize_shape = [j.shape[-1] for j in X_list]
            if self.shapes_print:
                print(f'BLOCK {j + 1}:')
                [print(list(i[0].shape), end=' ') for i in X_list]
                print()
        X = torch.cat(
            [torchvision.transforms.Resize((int(np.median(resize_shape)), int(np.median(resize_shape))))(j) for j in
             X_list], axis=1)

        return X


# (3, 128, 128) -> (vector_len)

class FaceNet_v1(torch.nn.Module):
    def __init__(self, vector_len=128):
        super().__init__()

        self.backbone1 = ParallelBlock([
            # path 1
            [
                # block 1
                [[3, 2, 8, 2],
                 [4, 2, 6],
                 [4, 2, 6],
                 [4, 2, 5],
                 [4, 2, 5]],
                # block 2
                [[4, 2, 4],
                 [4, 2, 4],
                 [4, 2, 4],
                 [4, 2, 3],
                 [4, 2, 3]]
            ],
            # path 2
            [
                # block 1
                [[3, 2, 8, 2],
                 [4, 2, 8],
                 [4, 2, 8],
                 [4, 2, 8],
                 [4, 2, 8]],
                # block 2
                [[4, 2, 7],
                 [4, 2, 7],
                 [4, 2, 6],
                 [4, 2, 6],
                 [4, 2, 5]]
            ],
        ])

        self.backbone2 = DenseBlock([
            [4, 2, 5],
            [4, 2, 4],
        ])

        self.linear = torch.nn.Linear(242, vector_len)

        self.act = torch.nn.SiLU()
        self.flat = torch.nn.Flatten(1)

    def forward(self, X):
        X = self.act(self.backbone1(X))
        X = self.backbone2(X)
        X = self.flat(X)
        X = self.linear(self.act(X))
        return normalize(X)


vectorizer = FaceNet_v1(128)
vectorizer.load_state_dict(torch.load(rf'{config_file.weights_path}', map_location='cpu'))


vidcap = cv2.VideoCapture(rf'{config_file.in_video_path}')
imgs = []
success,image = vidcap.read()
count = 0
while success:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imgs.append(image)
    success,image = vidcap.read()
    count += 1

loss_f = torch.nn.CosineSimilarity(dim=1)

def predict_mask(image, coords):
    if not len(coords): return -1
    th = 0.98
    image = to_tensor(image)
    #x, y = max(0, x), max(0, y)
    images = [resize(image[:, max(0, y):max(0, y)+h, max(0, x):max(0, x)+w]) for x, y, w, h in coords]
    base_image = resize(to_tensor(open_img(rf'{config_file.in_base_image}')))
    vectors = vectorizer(torch.stack(images))
    base_vector_1 = vectorizer(base_image.unsqueeze(0))
    scores_1 = loss_f(vectors, base_vector_1)
    scores = scores_1
    if max(scores) < th: return -1
    return torch.argmax(scores)


height, width, layers = imgs[0].shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(rf'{config_file.out_video_name}', fourcc, 30, (width, height))

vectorizer.to(torch.device('cpu'));

for image in imgs:
    print('!', end='')
    image = image.copy()
    coords = predict_face_yu(image)
    a = predict_mask(image, coords)

    if a != -1:
        x, y, w, h = coords[a]
        image = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    video.write(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# cv2.destroyAllWindows()
video.release()

