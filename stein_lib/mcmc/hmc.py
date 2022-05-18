import torch
import hamiltorch

class HMC(object):
    def __init__(
            self,
            step_size=0.3,
            num_steps_per_sample=1,
            num_restarts=1,
    ):

        self.step_size = step_size
        self.num_steps = num_steps_per_sample
        self.num_restarts = num_restarts

    def sample(self, x_start, model, num_samples):
        return hamiltorch.sample(
            log_prob_func=model.log_prob,
            params_init=x_start,
            num_samples=num_samples,
            step_size=self.step_size,
            num_steps_per_sample=self.num_steps,
        )

    def apply(self, x_inits, model, num_samples):

        assert x_inits.size(0) >= self.num_restarts

        N = num_samples
        d = self.num_restarts
        num_samples_list = [
            N // d + (1 if k < N % d else 0) for k in range(d)
        ]

        hist = []
        for i, ns in enumerate(num_samples_list):
            print('\nNum. samples: ', ns)
            samples = self.sample(x_inits[i], model, ns)
            hist += samples

        particles = torch.stack(hist)
        p_hist = [particles[:i].cpu().numpy() for i in range(N * self.num_restarts)]
        return (particles, p_hist)


class NUTS(HMC):

    def __init__(
            self,
            burn_in_steps=100,
            **kwargs,
    ):

        super().__init__(**kwargs)

        self.burn_in_steps = burn_in_steps

    def sample(self, x_start, model, num_samples):
        return hamiltorch.sample(
            log_prob_func=model.log_prob,
            params_init=x_start,
            num_samples=num_samples + self.burn_in_steps,
            step_size=self.step_size,
            num_steps_per_sample=self.num_steps,
            sampler=hamiltorch.Sampler.HMC_NUTS,
            burn=self.burn_in_steps,
            desired_accept_rate=0.8,
        )