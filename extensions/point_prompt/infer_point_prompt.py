import sys
sys.path.append('/home/zijianwu/projects/def-timsbc/zijianwu/codes/MedSAM/')
from segment_anything import sam_model_registry, SamPredictor
from os.path import join, isfile, basename
from os import getcwd
from matplotlib import pyplot as plt
from torch.nn import functional as F
import cv2
import torch
import numpy as np

print("--------------------test----------------------")

class PointPromptDemo:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.image = None
        self.image_embeddings = None
        self.img_size = None
        

    def show_mask(self, mask, ax, random_color=False, alpha=0.95):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([alpha])], axis=0)
        else:
            color = np.array([251/255, 252/255, 30/255, alpha])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    @torch.no_grad()
    def infer(self, x, y):
        coords_1024 = np.array([[[
            x * 1024 / self.img_size[1],
            y * 1024 / self.img_size[0]
        ]]])
        coords_torch = torch.tensor(coords_1024, dtype=torch.float32).to(self.model.device)
        labels_torch = torch.tensor([[1]], dtype=torch.long).to(self.model.device)
        point_prompt = (coords_torch, labels_torch)

        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points = point_prompt,
            boxes = None,
            masks = None,
        )
        low_res_logits, _ = self.model.mask_decoder(
            image_embeddings=self.image_embeddings, # (B, 256, 64, 64)
            image_pe=self.model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
        )

        low_res_probs = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)
        low_res_pred = F.interpolate(
            low_res_probs,
            size = self.img_size,
            mode = 'bilinear',
            align_corners = False
        )
        low_res_pred = low_res_pred.detach().cpu().numpy().squeeze()

        seg = np.uint8(low_res_pred > 0.5)

        return seg

    # def show(self, fig_size=5, alpha=0.95, scatter_size=10):

    #     assert self.image is not None, "Please set image first."
    #     seg = None
    #     fig, ax = plt.subplots(1, 1, figsize=(fig_size, fig_size))
    #     fig.canvas.header_visible = False
    #     fig.canvas.footer_visible = False
    #     fig.canvas.toolbar_visible = False
    #     fig.canvas.resizable = False

    #     plt.tight_layout()

    #     ax.imshow(self.image)
    #     ax.axis('off')

    #     def onclick(event):
    #         if event.inaxes == ax:
    #             x, y = float(event.xdata), float(event.ydata)
    #             with torch.no_grad():
    #                 ## rescale x, y from canvas size to 1024 x 1024
    #                 seg = self.infer(x, y)

    #             ax.clear()
    #             ax.imshow(self.image)
    #             ax.axis('off')
    #             ax.scatter(x, y, c='r', s=scatter_size)
    #             self.show_mask(seg, ax, random_color=False, alpha=alpha)

    #             gc.collect()

    #     fig.canvas.mpl_connect('button_press_event', onclick)
    #     plt.show()

    def set_image(self, image):
        self.img_size = image.shape[:2]
        if len(image.shape) == 2:
            image = np.repeat(image[:,:,None], 3, -1)
        self.image = image
        image_preprocess = self.preprocess_image(self.image)
        with torch.no_grad():
            self.image_embeddings = self.model.image_encoder(image_preprocess)
        
    def preprocess_image(self, image):
        img_resize = cv2.resize(
            image,
            (1024, 1024),
            interpolation=cv2.INTER_CUBIC
        )
        # Resizing
        img_resize = (img_resize - img_resize.min()) / np.clip(img_resize.max() - img_resize.min(), a_min=1e-8, a_max=None) # normalize to [0, 1], (H, W, 3
        # convert the shape to (3, H, W)
        assert np.max(img_resize)<=1.0 and np.min(img_resize)>=0.0, 'image should be normalized to [0, 1]'
        img_tensor = torch.tensor(img_resize).float().permute(2, 0, 1).unsqueeze(0).to(self.model.device)

        return img_tensor



def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    



medsam_ckpt_path = "/home/zijianwu/projects/def-timsbc/zijianwu/jobs/train_point_prompt_vit_h_endovis17_7e-5/medsam_point_prompt_best.pth"
device = "cuda:0"
medsam_model = sam_model_registry['vit_h'](checkpoint=medsam_ckpt_path)
medsam_model = medsam_model.to(device)

predictor = SamPredictor(medsam_model)

# load image
image = cv2.imread('/home/zijianwu/projects/def-timsbc/zijianwu/codes/MedSAM/extensions/point_prompt/frame0000.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Set prompt
predictor.set_image(image)

print("--------------------test----------------------")

# # input_point = np.array([[235, 650], [340, 612], [655, 513], [822, 488], [1170, 526], [1007, 512], [1253, 568]]) # [888, 490]
# input_point = np.array([[632, 520], [1211, 539]])

# # input_bbox = np.array([0, 560, 1366, 1300]) # [x, y, x, y]
# # input_label = np.array([1, 1, 1, 1, 1, 1, 1]) # 0
# input_label = np.array([1, 1])

# print("--------------------test----------------------")


# # Predict with prompt
# masks, scores, logits = predictor.predict(
#     point_coords=input_point,
#     point_labels=input_label,
#     multimask_output=True,
#     # point_coords=None,
#     # point_labels=None,
#     # box=input_bbox[None, :],
#     # multimask_output=False,
# )

# # Pick the highest score
# max_score = scores.max()
# max_idx = np.where(scores == max_score)
# mask = masks[max_idx]

# print("-----------------------Successful!!--------------------------")

# plt.imshow(image)
# show_mask(mask, plt.gca())
# show_points(input_point, input_label, plt.gca())

# plt.savefig('medsam_seg.png')

# # mask_img = mask.astype(np.uint8).squeeze()
# # mask_img =  mask_img * 255
# # ret, thresh = cv2.threshold(mask_img, 127, 255, 0)
# # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# # topleft_x, topleft_y, W, H = cv2.boundingRect(contours[0]) # (x, y, w, h)