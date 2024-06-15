# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import logging

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like
from ldm.util import is_old_ms_version
from tqdm import tqdm

import mindspore as ms
from mindspore import ops, mint

_logger = logging.getLogger(__name__)


class PLMSSampler:
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0.0, verbose=True):
        if ddim_eta != 0:
            raise ValueError("ddim_eta must be 0 for PLMS")
        self.ddim_timesteps = make_ddim_timesteps(
            ddim_discr_method=ddim_discretize,
            num_ddim_timesteps=ddim_num_steps,
            num_ddpm_timesteps=self.ddpm_num_timesteps,
            verbose=verbose,
        )

        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, "alphas have to be defined for each timestep"

        self.betas = self.model.betas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = self.model.alphas_cumprod_prev

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = ops.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = ops.sqrt(1.0 - alphas_cumprod)
        self.log_one_minus_alphas_cumprod = ops.log(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = ops.sqrt(1.0 / alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = ops.sqrt(1.0 / alphas_cumprod - 1)

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(
            alphacums=alphas_cumprod, ddim_timesteps=self.ddim_timesteps, eta=ddim_eta, verbose=verbose
        )
        self.ddim_sigmas = ddim_sigmas
        self.ddim_alphas = ddim_alphas
        self.ddim_alphas_prev = ddim_alphas_prev
        self.ddim_sqrt_one_minus_alphas = ops.sqrt(1.0 - ddim_alphas)
        sigmas_for_original_sampling_steps = ddim_eta * ops.sqrt(
            (1 - self.alphas_cumprod_prev)
            / (1 - self.alphas_cumprod)
            * (1 - self.alphas_cumprod / self.alphas_cumprod_prev)
        )
        self.ddim_sigmas_for_original_num_steps = sigmas_for_original_sampling_steps

    def sample(
        self,
        S,
        batch_size,
        shape,
        conditioning=None,
        callback=None,
        normals_sequence=None,
        img_callback=None,
        quantize_x0=False,
        eta=0.0,
        mask=None,
        x0=None,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        verbose=True,
        x_T=None,
        log_every_t=100,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        T0=None,
        # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
        **kwargs,
    ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                ctmp = conditioning[list(conditioning.keys())[0]]
                while isinstance(ctmp, list):
                    ctmp = ctmp[0]
                cbs = ctmp.shape[0]
                if cbs != batch_size:
                    _logger.warning(f"Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    _logger.warning(f"Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")
        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        size = (batch_size, *shape)
        _logger.debug(f"Data shape for PLMS sampling is {size}")
        samples, intermediates = self.plms_sampling(
            conditioning,
            size,
            callback=callback,
            img_callback=img_callback,
            quantize_denoised=quantize_x0,
            mask=mask,
            x0=x0,
            ddim_use_original_steps=False,
            noise_dropout=noise_dropout,
            temperature=temperature,
            score_corrector=score_corrector,
            corrector_kwargs=corrector_kwargs,
            x_T=x_T,
            log_every_t=log_every_t,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=unconditional_conditioning,
            T0=T0,
        )
        return samples, intermediates

    def plms_sampling(
        self,
        cond,
        shape,
        x_T=None,
        ddim_use_original_steps=False,
        callback=None,
        timesteps=None,
        quantize_denoised=False,
        mask=None,
        x0=None,
        img_callback=None,
        log_every_t=100,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        T0=None,
    ):
        b = shape[0]
        if x_T is None:
            img = ops.standard_normal(shape)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {"x_inter": [img], "pred_x0": [img]}
        time_range = list(reversed(range(0, timesteps))) if ddim_use_original_steps else ms.numpy.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        _logger.debug(f"Running PLMS Sampling with {total_steps} timesteps")

        # iterator = tqdm(time_range, desc='PLMS Sampler', total=total_steps)
        iterator = time_range
        old_eps = []

        for i, step in tqdm(enumerate(iterator)):
            if T0 is not None:
                if i < T0:
                    continue

            index = total_steps - i - 1
            ts = ms.numpy.full((b,), step, dtype=ms.int64)
            ts_next = ms.numpy.full((b,), time_range[min(i + 1, len(time_range) - 1)], dtype=ms.int64)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts, ms.numpy.randn(x0.shape))
                img = img_orig * mask + (1.0 - mask) * img

            outs = self.p_sample_plms(
                img,
                cond,
                ts,
                index=index,
                use_original_steps=ddim_use_original_steps,
                quantize_denoised=quantize_denoised,
                temperature=temperature,
                noise_dropout=noise_dropout,
                score_corrector=score_corrector,
                corrector_kwargs=corrector_kwargs,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
                old_eps=old_eps,
                t_next=ts_next,
            )
            img, pred_x0, e_t = outs
            old_eps.append(e_t)
            if len(old_eps) >= 4:
                old_eps.pop(0)
            if callback:
                callback(i)
            if img_callback:
                img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates["x_inter"].append(img)
                intermediates["pred_x0"].append(pred_x0)

        return img, intermediates

    def p_sample_plms(
        self,
        x,
        c,
        t,
        index,
        repeat_noise=False,
        use_original_steps=False,
        quantize_denoised=False,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        old_eps=None,
        t_next=None,
    ):
        b = x.shape[0]

        def get_model_output(x, t):
            if unconditional_conditioning is None or unconditional_guidance_scale == 1.0:
                e_t = self.model.apply_model(x, t, c)
            else:
                x_in = ops.concat((x, x), axis=0)
                t_in = ops.concat((t, t), axis=0)
                if isinstance(c, dict):
                    assert isinstance(unconditional_conditioning, dict)
                    c_in = dict()
                    for k in c:
                        if isinstance(c[k], list):
                            c_in[k] = [
                                ops.concat([unconditional_conditioning[k][i], c[k][i]], axis=0)
                                for i in range(len(c[k]))
                            ]
                        else:
                            c_in[k] = ops.concat([unconditional_conditioning[k], c[k]], axis=0)
                    ldm_output = self.model.apply_model(x_in, t_in, c_in)  # hybrid condition
                else:
                    c_in = ops.concat((unconditional_conditioning, c), axis=0)
                    ldm_output = self.model.apply_model(x_in, t_in, c_in)

                if is_old_ms_version():
                    e_t_uncond, e_t = mint.split(ldm_output, axis=0, output_num=2)
                else:
                    e_t_uncond, e_t = mint.split(ldm_output, split_size_or_sections=ldm_output.shape[0] // 2, axis=0)

                e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

            if score_corrector is not None:
                assert self.model.parameterization == "eps"
                e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

            return e_t

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = (
            self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        )
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas

        def get_x_prev_and_pred_x0(e_t, index):
            # select parameters corresponding to the currently considered timestep
            a_t = ms.numpy.full((b, 1, 1, 1), alphas[index])
            a_prev = ms.numpy.full((b, 1, 1, 1), alphas_prev[index])
            sigma_t = ms.numpy.full((b, 1, 1, 1), sigmas[index])
            sqrt_one_minus_at = ms.numpy.full((b, 1, 1, 1), sqrt_one_minus_alphas[index])

            # current prediction for x_0
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
            if quantize_denoised:
                pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
            # direction pointing to x_t
            dir_xt = (1.0 - a_prev - sigma_t**2).sqrt() * e_t
            noise = sigma_t * noise_like(x.shape, repeat_noise) * temperature
            if noise_dropout > 0.0:
                noise, _ = ops.dropout(noise, p=noise_dropout)
            x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

            return x_prev, pred_x0

        e_t = get_model_output(x, t)
        if len(old_eps) == 0:
            # Pseudo Improved Euler (2nd order)
            x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t, index)
            e_t_next = get_model_output(x_prev, t_next)
            e_t_prime = (e_t + e_t_next) / 2
        elif len(old_eps) == 1:
            # 2nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (3 * e_t - old_eps[-1]) / 2
        elif len(old_eps) == 2:
            # 3nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (23 * e_t - 16 * old_eps[-1] + 5 * old_eps[-2]) / 12
        elif len(old_eps) >= 3:
            # 4nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (55 * e_t - 59 * old_eps[-1] + 37 * old_eps[-2] - 9 * old_eps[-3]) / 24

        x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t_prime, index)

        return x_prev, pred_x0, e_t
