import torch
import numpy as np
import torch.nn as nn

class ZOTrainer:

    def __init__(self, named_parameters_to_optim, lr, zo_eps, is_llama, qlinears):
        self.named_parameters_to_optim = named_parameters_to_optim
        self.loss_fct = nn.CrossEntropyLoss()
        self.lr = lr
        self.zo_eps = zo_eps
        self.is_llama = is_llama
        self.qlinears = qlinears

    def zo_perturb_parameters(self, random_seed=None, scaling_factor=1.0):
        torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)

        for name, param in self.named_parameters_to_optim:
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device,
                             dtype=param.data.dtype)
            param.data.add_(scaling_factor * self.zo_eps * z)

    def reverse_qlinear(self):

        for qlinear in self.qlinears:
            qlinear.scaling_factor = - qlinear.scaling_factor

    def perturb_qlinear(self, perturb):

        for qlinear in self.qlinears:
            qlinear.perturb_forward = perturb

    def zo_step(self, model, inputs):
        """
        Estimate gradient by MeZO. Return the loss from f(theta + z)
        """
        # Sample the random seed for sampling z
        self.zo_random_seed = np.random.randint(1000000000)
        self.perturb_qlinear(perturb=True)

        # First function evaluation: f(theta + z)
        torch.manual_seed(self.zo_random_seed)
        loss1 = self.zo_forward(model, inputs)

        self.reverse_qlinear()

        # Second function evaluation: f(theta - z)
        torch.manual_seed(self.zo_random_seed)
        loss2 = self.zo_forward(model, inputs)

        # Compute the projected gradient
        self.projected_grad = ((loss1 - loss2) / (2 * self.zo_eps)).item()

        return loss1

    def zo_update(self):
        """
        Update the parameters with the estimated gradients.
        """

        # Reset the random seed for sampling zs
        torch.manual_seed(self.zo_random_seed)

        # self.named_parameters_to_optim from the model
        for i, (name, param) in enumerate(self.named_parameters_to_optim):
            # Resample z
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            param.data.sub_(self.lr * self.projected_grad * z)


    def zo_forward(self, model, inputs):
        """
        Forward pass for the model with perturbed parameters.
        """
        with torch.no_grad():
            # inputs["use_cache"] = False
            outputs = model.model.decoder(inputs['input_ids']) if not self.is_llama else model.model(
                inputs['input_ids'])
            hidden_states = outputs[0]
            logits = model.lm_head(hidden_states)
            shift_logits = logits[:, :-1, :]
            shift_labels = inputs['input_ids'][:, 1:]
            loss = self.loss_fct(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
            )

        return loss.detach()
