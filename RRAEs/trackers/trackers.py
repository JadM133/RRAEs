from RRAEs.utilities import get_diff_func
import jax.numpy as jnp


class Null_Tracker:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, current_loss, prev_avg_loss, *args, **kwargs):
        return {}

    def init(self):
        return {}


class RRAE_gen_Tracker:
    def __init__(
        self,
        k_init,
        patience=10,
        ideal_loss=10,
    ):
        self.patience = patience
        self.ideal_loss = ideal_loss
        self.patience_c = 0
        self.k_now = k_init

        self.change_prot = False
        self.loss_prev_mode = jnp.inf
        self.wait_counter = 0
        self.k_now = k_init

    def __call__(self, current_loss, *args, **kwargs):

        if current_loss < self.ideal_loss:
            self.patience_c += 1
            if self.patience_c == self.patience:
                self.patience_c = 0
                self.k_now -= 1
        else:
            self.patience_c = 0

        return {"k_max": self.k_now}

    def init(self):
        return {"k_max": self.k_now}


class RRAE_pars_Tracker:
    def __init__(
        self,
        k_init=None,
        patience=10,
        max_wait=500,
        eps_0=1,
        wac_0=1,
        diff_func_params_eps=None,
        diff_func_params_wac=None,
    ):
        k_init = 1 if k_init is None else k_init

        if diff_func_params_eps is None:
            diff_func_params_eps = {}
            diff_func_params_eps["x_1"] = 10
            diff_func_params_eps["x_2"] = 40
            diff_func_params_eps["V_1"] = 0
            diff_func_params_eps["V_2"] = eps_0
            diff_func_params_eps["type"] = "line"

        if diff_func_params_wac is None:
            diff_func_params_wac = {}
            diff_func_params_wac["x_1"] = 10
            diff_func_params_wac["x_2"] = 40
            diff_func_params_wac["V_1"] = 0
            diff_func_params_wac["V_2"] = wac_0
            diff_func_params_wac["type"] = "line"

        self.patience = patience
        self.diff_func_eps = get_diff_func(**diff_func_params_eps)
        self.diff_func_wac = get_diff_func(**diff_func_params_wac)
        self.patience_c = 0
        self.change_prot = False
        self.loss_prev_mode = jnp.inf
        self.wait_counter = 0
        self.max_wait = max_wait
        self.k_now = k_init

    def __call__(self, current_loss, prev_avg_loss, *args, **kwargs):
        if not self.change_prot:
            if jnp.abs(current_loss - prev_avg_loss) < self.diff_func_eps(current_loss):
                self.patience_c += 1
                if self.patience_c == self.patience:
                    self.patience_c = 0
                    self.change_prot = True
                    self.k_now += 1
                    self.loss_prev_mode = prev_avg_loss
            else:
                self.patience_c = 0
        else:
            self.wait_counter += 1
            new_wac = self.diff_func_wac(current_loss)

            if (self.wait_counter >= self.max_wait) or (
                self.loss_prev_mode - current_loss
            ) > new_wac:
                self.change_prot = False
                self.wait_counter = 0
        return {"k_max": self.k_now}

    def init(self):
        return {"k_max": self.k_now}


class RRAE_fixed_Tracker:
    def __init__(self, k_init):
        self.k_now = k_init

    def __call__(self, *args, **kwargs):
        return {"k_max": self.k_now}
    
    def init(self):
        return {"k_max": self.k_now}
