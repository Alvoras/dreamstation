import os
import sys
from pathlib import Path

import imageio
import libxmp
import numpy as np
import torch
from PIL import Image
from discord_webhook import DiscordWebhook, DiscordEmbed
from imgtag import ImgTag
from omegaconf import OmegaConf
from torch import nn, optim
from torchvision import transforms
import kornia.augmentation as K
from stegano import lsb
import json

from lib.config import DISCORD_WEBHOOK

sys.path.append(os.path.join("lib", "vendor", "taming-transformers"))
from taming.models import cond_transformer, vqgan

from lib.vendor.CLIP import clip
from lib.const import (
    CLIP_MODEL,
    CUTN,
    CUT_POW,
    STEP_SIZE,
    NOISE_PROMPT_SEED,
    NOISE_PROMPT_WEIGHTS,
    INIT_WEIGHT,
    LOADING_TASK_STEPS,
)
from lib.grad import clamp_with_grad, replace_grad
from lib.image_utils import resize_image
from lib.process import vector_quantize, resample

from torchvision.transforms import functional as TF
from torch.nn import functional as F

from lib.prompt_utils import parse_prompt


class Trainer:
    def __init__(self, args, prompt, progress, loading_task, device):
        self.progress = progress
        self.prompt = prompt
        self.args = args
        self.discord_update = args.discord_update
        self.display_freq = args.display_freq
        self.discord_freq = args.discord_freq
        self.author = args.author
        self.no_stegano = args.no_stegano
        self.no_metadata = args.no_metadata
        self.model_name = args.model
        self.seed = args.seed
        self.width = args.width
        self.height = args.height
        self.max_iterations = args.max_iterations
        self.initial_image = args.initial_image
        self.target_images = args.target_images
        self.progress_dir = self.get_progress_dir()
        self.progress_img_path = os.path.join(self.args.out, "progress.png")

        self.vqgan_config = f"{self.model_name}.yaml"
        self.vqgan_checkpoint = f"{self.model_name}.ckpt"

        self.model = self.load_vqgan_model().to(device)
        progress.advance(loading_task)

        self.perceptor = (
            clip.load(CLIP_MODEL, jit=False)[0].eval().requires_grad_(False).to(device)
        )

        self.cut_size = self.perceptor.visual.input_resolution
        self.e_dim = self.model.quantize.e_dim
        self.f = 2 ** (self.model.decoder.num_resolutions - 1)
        self.make_cutouts = MakeCutouts(self.cut_size, CUTN, cut_pow=CUT_POW)
        self.n_toks = self.model.quantize.n_e
        self.toks_x, self.toks_y = args.width // self.f, args.height // self.f
        self.side_x, self.side_y = self.toks_x * self.f, self.toks_y * self.f
        self.z_min = self.model.quantize.embedding.weight.min(dim=0).values[
            None, :, None, None
        ]
        self.z_max = self.model.quantize.embedding.weight.max(dim=0).values[
            None, :, None, None
        ]

        progress.update(loading_task, completed=LOADING_TASK_STEPS)

        if args.initial_image:
            pil_image = Image.open(args.initial_image).convert("RGB")
            pil_image = pil_image.resize((self.side_x, self.side_y), Image.LANCZOS)
            self.z, *_ = self.model.encode(
                TF.to_tensor(pil_image).to(device).unsqueeze(0) * 2 - 1
            )
        else:
            one_hot = F.one_hot(
                torch.randint(self.n_toks, [self.toks_y * self.toks_x], device=device),
                self.n_toks,
            ).float()
            self.z = one_hot @ self.model.quantize.embedding.weight
            self.z = self.z.view([-1, self.toks_y, self.toks_x, self.e_dim]).permute(
                0, 3, 1, 2
            )
        self.z_orig = self.z.clone()
        self.z.requires_grad_(True)
        self.opt = optim.Adam([self.z], lr=STEP_SIZE)

        self.normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        )

        self.pMs = []

        for prompt_chunk in self.prompt:
            txt, weight, stop = parse_prompt(prompt_chunk)
            embed = self.perceptor.encode_text(clip.tokenize(txt).to(device)).float()
            self.pMs.append(Prompt(embed, weight, stop).to(device))

        for prompt_chunk in args.target_images:
            path, weight, stop = parse_prompt(prompt_chunk)
            img = resize_image(
                Image.open(path).convert("RGB"), (self.side_x, self.side_y)
            )
            batch = self.make_cutouts(TF.to_tensor(img).unsqueeze(0).to(device))
            embed = self.perceptor.encode_image(self.normalize(batch)).float()
            self.pMs.append(Prompt(embed, weight, stop).to(device))

        for seed, weight in zip(NOISE_PROMPT_SEED, NOISE_PROMPT_WEIGHTS):
            gen = torch.Generator().manual_seed(seed)
            embed = torch.empty([1, self.perceptor.visual.output_dim]).normal_(
                generator=gen
            )
            self.pMs.append(Prompt(embed, weight).to(device))

    def get_progress_dir(self):
        str_prompts = "_".join(self.prompt).replace(" ", "-")
        return f"{str_prompts}_{self.width}x{self.height}_{self.max_iterations}it"

    def preflight(self):
        progress_dir = self.get_progress_dir()
        Path(f"{self.args.out}/{progress_dir}").mkdir(parents=True, exist_ok=True)

    def load_vqgan_model(self):
        config = OmegaConf.load(self.vqgan_config)
        if config.model.target == "taming.models.vqgan.VQModel":
            model = vqgan.VQModel(**config.model.params)
            model.eval().requires_grad_(False)
            model.init_from_ckpt(self.vqgan_checkpoint)
        elif config.model.target == "taming.models.cond_transformer.Net2NetTransformer":
            parent_model = cond_transformer.Net2NetTransformer(**config.model.params)
            parent_model.eval().requires_grad_(False)
            parent_model.init_from_ckpt(self.vqgan_checkpoint)
            model = parent_model.first_stage_model
        else:
            raise ValueError(f"unknown model type: {config.model.target}")
        del model.loss
        return model

    def synth(self):
        z_q = vector_quantize(
            self.z.movedim(1, 3), self.model.quantize.embedding.weight
        ).movedim(3, 1)
        return clamp_with_grad(self.model.decode(z_q).add(1).div(2), 0, 1)

    def add_xmp_data(self, filename, iteration):
        image = ImgTag(filename=filename)
        image.xmp.append_array_item(
            libxmp.consts.XMP_NS_DC,
            "creator",
            self.author,
            {"prop_array_is_ordered": True, "prop_value_is_array": True},
        )
        image.xmp.append_array_item(
            libxmp.consts.XMP_NS_DC,
            "title",
            " | ".join(self.prompt),
            {"prop_array_is_ordered": True, "prop_value_is_array": True},
        )
        image.xmp.append_array_item(
            libxmp.consts.XMP_NS_DC,
            "iteration",
            str(iteration),
            {"prop_array_is_ordered": True, "prop_value_is_array": True},
        )
        image.xmp.append_array_item(
            libxmp.consts.XMP_NS_DC,
            "model",
            self.model_name,
            {"prop_array_is_ordered": True, "prop_value_is_array": True},
        )
        image.xmp.append_array_item(
            libxmp.consts.XMP_NS_DC,
            "seed",
            str(self.seed),
            {"prop_array_is_ordered": True, "prop_value_is_array": True},
        )
        image.xmp.append_array_item(
            libxmp.consts.XMP_NS_DC,
            "initial_image",
            str(self.initial_image),
            {"prop_array_is_ordered": True, "prop_value_is_array": True},
        )
        image.xmp.append_array_item(
            libxmp.consts.XMP_NS_DC,
            "target_images",
            str(self.target_images),
            {"prop_array_is_ordered": True, "prop_value_is_array": True},
        )
        image.close()

    def add_stegano_data(self, filename, iteration):
        data = {
            "title": " | ".join(self.prompt),
            "creator": self.author,
            "iteration": iteration,
            "model": self.model_name,
            "seed": str(self.seed),
            "initial_image": self.initial_image,
            "target_images": str(self.target_images),
        }
        lsb.hide(filename, json.dumps(data)).save(filename)

    @torch.no_grad()
    def save_progress(self, iteration):
        out = self.synth()
        TF.to_pil_image(out[0].cpu()).save(self.progress_img_path)
        if not self.no_stegano:
            self.add_stegano_data(self.progress_img_path, iteration)
        if not self.no_metadata:
            self.add_xmp_data(self.progress_img_path, iteration)

    def ascend_txt(self, iteration):
        out = self.synth()
        encoded_image = self.perceptor.encode_image(
            self.normalize(self.make_cutouts(out))
        ).float()

        result = []

        if INIT_WEIGHT:
            result.append(F.mse_loss(self.z, self.z_orig) * INIT_WEIGHT / 2)

        for prompt in self.pMs:
            result.append(prompt(encoded_image))
        img = np.array(
            out.mul(255).clamp(0, 255)[0].cpu().detach().numpy().astype(np.uint8)
        )[:, :, :]
        img = np.transpose(img, (1, 2, 0))
        filename = str(
            Path(self.args.out, self.progress_dir, f"{iteration:04}.png").absolute()
        )
        imageio.imwrite(filename, np.array(img))
        if not self.no_stegano:
            self.add_stegano_data(filename, iteration)

        if not self.no_metadata:
            self.add_xmp_data(filename, iteration)
        return result

    def train(self, iteration):
        self.opt.zero_grad()
        loss_all = self.ascend_txt(iteration)
        if iteration % self.display_freq == 0:
            self.save_progress(iteration)
        if self.discord_update:
            if iteration > 0 and iteration % self.discord_freq == 0:
                self.progress.log("Pushed progress to Discord")
                self.push_progress(
                    title=f"Checkpoint ({iteration}/{self.max_iterations})",
                    description=f"{str(self.prompt)} ({self.width}x{self.height}) - {self.max_iterations} iterations | {self.progress.tasks[1].elapsed:.2f}s",
                    color="84abcd",
                )

        loss = sum(loss_all)
        loss.backward()
        self.opt.step()
        with torch.no_grad():
            self.z.copy_(self.z.maximum(self.z_min).minimum(self.z_max))

    def push_progress(
        self, title="", description="", color="42ba96", username="Paprika"
    ):
        if DISCORD_WEBHOOK:
            webhook = DiscordWebhook(url=DISCORD_WEBHOOK, username=username)
            with open(self.progress_img_path, "rb") as f:
                webhook.add_file(file=f.read(), filename="progress.jpg")

            embed = DiscordEmbed(
                title=title,
                description=description,
                color=color,
            )

            webhook.add_embed(embed)

            # TODO: log failure
            # response = webhook.execute()
            webhook.execute()

    def push_finish(self):
        self.push_progress(
            title="Job done",
            description=f"{str(self.prompt)} ({self.width}x{self.height}) - {self.max_iterations} iterations | {self.progress.tasks[1].elapsed:.2f}s",
        )


class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.0):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow
        self.augs = nn.Sequential(
            K.RandomHorizontalFlip(p=0.5),
            # K.RandomSolarize(0.01, 0.01, p=0.7),
            K.RandomSharpness(0.3, p=0.4),
            K.RandomAffine(degrees=30, translate=0.1, p=0.8, padding_mode="border"),
            K.RandomPerspective(0.2, p=0.4),
            K.ColorJitter(hue=0.01, saturation=0.01, p=0.7),
        )
        self.noise_fac = 0.1

    def forward(self, input):
        side_y, side_x = input.shape[2:4]
        max_size = min(side_x, side_y)
        min_size = min(side_x, side_y, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(
                torch.rand([]) ** self.cut_pow * (max_size - min_size) + min_size
            )
            offset_x = torch.randint(0, side_x - size + 1, ())
            offset_y = torch.randint(0, side_y - size + 1, ())
            cutout = input[:, :, offset_y : offset_y + size, offset_x : offset_x + size]
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
        batch = self.augs(torch.cat(cutouts, dim=0))
        if self.noise_fac:
            facs = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_fac)
            batch = batch + facs * torch.randn_like(batch)
        return batch


class Prompt(nn.Module):
    def __init__(self, embed, weight=1.0, stop=float("-inf")):
        super().__init__()
        self.register_buffer("embed", embed)
        self.register_buffer("weight", torch.as_tensor(weight))
        self.register_buffer("stop", torch.as_tensor(stop))

    def forward(self, input):
        input_normed = F.normalize(input.unsqueeze(1), dim=2)
        embed_normed = F.normalize(self.embed.unsqueeze(0), dim=2)
        dists = input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)
        dists = dists * self.weight.sign()
        return (
            self.weight.abs()
            * replace_grad(dists, torch.maximum(dists, self.stop)).mean()
        )
