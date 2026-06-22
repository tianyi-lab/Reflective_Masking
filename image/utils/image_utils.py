import torch
import PIL
import random
from PIL import Image, ImageDraw
from diffusers import VQModel
from diffusers.image_processor import VaeImageProcessor
import torch.nn.functional as F

def decode_vq_to_image(
    vq_codes: torch.LongTensor, 
    save_path: str, 
    vae_ckpt: str, 
    image_height: int, 
    image_width: int,
    vqvae: VQModel = None
) -> Image.Image:
    device = vq_codes.device
    if vqvae is None:
        vqvae = VQModel.from_pretrained(vae_ckpt, subfolder="vqvae").to(device)
    
    scale = 2 ** (len(vqvae.config.block_out_channels) - 1)
    img_proc = VaeImageProcessor(vae_scale_factor=scale, do_normalize=False)

    latent_height = image_height // scale
    latent_width = image_width // scale

    if vq_codes.shape[1] != latent_height * latent_width:
        raise ValueError(
            f"VQ codes length mismatch: {vq_codes.shape[1]} != {latent_height * latent_width} "
            f"for image size ({image_height},{image_width}) with scale {scale}"
        )

    latents = (vq_codes.view(1, latent_height, latent_width) - 126356).long()

    recon = vqvae.decode(
        latents,
        force_not_quantize=True,
        shape=(1, latent_height, latent_width, vqvae.config.latent_channels),
    ).sample.clip(0, 1)

    img = img_proc.postprocess(recon.detach(), output_type="pil")[0]
    return img


def preprocess_image(image_path: str, target_size: tuple = (512, 512)):
    img = Image.open(image_path).convert("RGB")
    crop_size_list = generate_crop_size_list((target_size[0] // 32) ** 2, 32)
    processed_img = var_center_crop(img, crop_size_list=crop_size_list)
    return processed_img


def calculate_vq_params(image_height: int, image_width: int, vae_scale: int = 16):
    token_grid_height = image_height // vae_scale
    token_grid_width = image_width // vae_scale
    seq_len = token_grid_height * token_grid_width
    newline_every = token_grid_width
    return seq_len, newline_every, token_grid_height, token_grid_width

def center_crop(pil_image, crop_size):
    while pil_image.size[0] >= 2 * crop_size[0] and pil_image.size[1] >= 2 * crop_size[1]:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = max(crop_size[0] / pil_image.size[0], crop_size[1] / pil_image.size[1])
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    crop_left = random.randint(0, pil_image.size[0] - crop_size[0])
    crop_upper = random.randint(0, pil_image.size[1] - crop_size[1])
    crop_right = crop_left + crop_size[0]
    crop_lower = crop_upper + crop_size[1]
    return pil_image.crop(box=(crop_left, crop_upper, crop_right, crop_lower))


def var_center_crop(pil_image, crop_size_list, random_top_k=1):
    w, h = pil_image.size
    rem_percent = [min(cw / w, ch / h) / max(cw / w, ch / h) for cw, ch in crop_size_list]
    crop_size = random.choice(
        sorted(((x, y) for x, y in zip(rem_percent, crop_size_list)), reverse=True)[:random_top_k]
    )[1]
    return center_crop(pil_image, crop_size)


def generate_crop_size_list(num_patches, patch_size, max_ratio=4.0):
    assert max_ratio >= 1.0
    crop_size_list = []
    wp, hp = num_patches, 1
    while wp > 0:
        if max(wp, hp) / min(wp, hp) <= max_ratio:
            crop_size_list.append((wp * patch_size, hp * patch_size))
        if (hp + 1) * wp <= num_patches:
            hp += 1
        else:
            wp -= 1
    return crop_size_list

def add_break_line(sequence: list, H: int, W: int, new_number: int = 0) -> list:
    result = []
    for i in range(H):
        start = i * W
        end = start + W
        row = sequence[start:end]
        result.extend(row + [new_number])
    return result

def encode_img_with_breaks(img, vqvae, vae_scale_factor: int = 16):
    from diffusers.image_processor import VaeImageProcessor
    
    orig = img.convert("RGB")
    orig_resized = orig
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor, do_normalize=False)
    x = image_processor.preprocess(orig_resized).to(vqvae.device)
    latents = vqvae.encode(x).latents
    latents_bsz, channels, lat_h, lat_w = latents.shape
    quantized = vqvae.quantize(latents)[2][2] + 126356
    quantized = quantized.reshape(latents_bsz, lat_h, lat_w).flatten().tolist()
    img_token = add_break_line(quantized, lat_h, lat_w, new_number=126084)
    img_token = [126349] + img_token + [126350]
    return img_token

@torch.no_grad()
def encode_img_with_paint(
    img: Image.Image,
    vqvae: VQModel,
    *,
    mask_h_ratio: float = 1,
    mask_w_ratio: float = 0.2,
    gray_value: int = 127,
    downsample_mode: str = "area",
    dilate_latent_k: int = 0,
    mask_mode: str = "inpainting",
):
    MASK_TOKEN_ID = 126336
    NEWLINE_TOKEN_ID = 126084
    VQ_OFFSET = 126356

    assert mask_mode in ("inpainting", "outpainting"), "mask_mode must be 'inpainting' or 'outpainting'"

    img = img.convert("RGB")
    W, H = img.size
    mh = int(round(H * mask_h_ratio))
    mw = int(round(W * mask_w_ratio))
    top = (H - mh) // 2
    left = (W - mw) // 2
    bottom = top + mh
    right = left + mw

    if mask_mode == "inpainting":
        vis_img = img.copy()
        draw = ImageDraw.Draw(vis_img)
        draw.rectangle([left, top, right, bottom], fill=(gray_value, gray_value, gray_value))
    elif mask_mode == "outpainting":
        bg = Image.new("RGB", (W, H), (gray_value, gray_value, gray_value))
        crop = img.crop((left, top, right, bottom))
        bg.paste(crop, (left, top))
        vis_img = bg

    vae_scale_factor = 2 ** (len(vqvae.config.block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor, do_normalize=False)
    x = image_processor.preprocess(img).to(vqvae.device)
    latents = vqvae.encode(x).latents
    _, _, lat_h, lat_w = latents.shape

    quant_pack = vqvae.quantize(latents)
    indices = quant_pack[2][2].view(1, lat_h, lat_w)

    Hp, Wp = x.shape[-2:]
    mask_px = torch.zeros((1, 1, Hp, Wp), dtype=torch.float32, device=vqvae.device)
    top_p  = int(round(top  * Hp / H))
    left_p = int(round(left * Wp / W))
    bh_p   = int(round(mh   * Hp / H))
    bw_p   = int(round(mw   * Wp / W))
    mask_px[:, :, top_p:top_p+bh_p, left_p:left_p+bw_p] = 1.0

    if mask_mode == "outpainting":
        mask_px = 1.0 - mask_px

    if downsample_mode not in ("nearest", "area", "bilinear"):
        downsample_mode = "area"
    mask_lat = F.interpolate(mask_px, size=(lat_h, lat_w), mode=downsample_mode)
    mask_lat = (mask_lat > 0.5) if downsample_mode == "area" else (mask_lat >= 0.5)
    mask_lat = mask_lat[0, 0]

    if dilate_latent_k > 0:
        m = mask_lat.float().unsqueeze(0).unsqueeze(0)
        ker = 2 * dilate_latent_k + 1
        m = F.max_pool2d(m, kernel_size=ker, stride=1, padding=dilate_latent_k)
        mask_lat = (m[0, 0] > 0.5)

    idx_flat = indices.view(-1)
    mask_flat = mask_lat.view(-1)
    tokens = torch.empty_like(idx_flat)
    tokens[mask_flat] = MASK_TOKEN_ID
    tokens[~mask_flat] = idx_flat[~mask_flat] + VQ_OFFSET
    tokens_list = tokens.tolist()


    img_token = add_break_line(tokens_list, lat_h, lat_w, NEWLINE_TOKEN_ID)
    return img_token, vis_img