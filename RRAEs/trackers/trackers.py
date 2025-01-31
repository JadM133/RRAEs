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
        patience_conv=1,
        patience_not_right=500,
        max_steps=1000,
        perf_loss=0,
        eps_0=5,
        eps_perc=1,
    ):

        self.patience_c_conv = 0
        self.patience_c = 0
        self.steps_c = 0
        self.k_now = k_init

        self.change_prot = False
        self.loss_prev_mode = jnp.inf
        self.wait_counter = 0
        self.k_now = k_init
        self.max_steps = max_steps
        self.converged = False
        self.total_steps = 0

        self.patience_conv = patience_conv
        self.patience = patience_not_right
        self.init_phase = True
        self.ideal_loss = jnp.nan
        self.eps_0 = eps_0
        self.perf_loss = perf_loss
        self.eps_perc = eps_perc
        self.k_steps = 0
        self.prev_k_steps = 0
        self.max_patience = jnp.inf

    def __call__(self, current_loss, prev_avg_loss, *args, **kwargs):
        save = False
        break_ = False
        self.prev_k_steps += 1

        if self.init_phase:
            if (
                jnp.abs(current_loss - prev_avg_loss) / jnp.abs(prev_avg_loss) * 100
                < self.eps_perc
            ):
                self.patience_c += 1
                if self.patience_c == self.patience:
                    self.patience_c = 0
                    self.init_phase = False
                    self.ideal_loss = prev_avg_loss
                    print(f"Ideal loss is {self.ideal_loss}")
            else:
                self.patience_c = 0

            if current_loss < self.perf_loss:
                self.ideal_loss = self.perf_loss
                self.init_phase = False
                self.patience_c = 0
                print(f"Ideal loss is {self.ideal_loss}")

            if not self.init_phase:
                diff_func_params_eps = {}
                diff_func_params_eps["x_1"] = self.ideal_loss
                diff_func_params_eps["x_2"] = 40
                diff_func_params_eps["V_1"] = 0
                diff_func_params_eps["V_2"] = self.eps_0
                diff_func_params_eps["type"] = "line"
                self.diff_func_eps = get_diff_func(**diff_func_params_eps)

            return {"k_max": self.k_now, "save": save, "break_": break_}

        self.total_steps += 1
        if not self.converged:
            if current_loss < self.ideal_loss:
                self.patience_c = 0
                self.k_steps = 0
                self.patience_c_conv += 1
                if self.patience_c_conv == self.patience_conv:
                    self.patience_c_conv = 0
                    self.k_now -= 1
                    self.prev_k_steps = 0
                    if self.total_steps == self.patience_conv:
                        save = False
                    else:
                        save = True
                    self.total_steps = 0
            else:
                self.patience_c_conv = 0
                self.k_steps += 1
                if jnp.abs(current_loss - prev_avg_loss) < self.diff_func_eps(
                    current_loss
                ):
                    self.k_steps = 0
                    self.patience_c += 1
                    if self.patience_c == self.patience:
                        self.patience_c = 0
                        self.k_now += 1
                        self.prev_k_steps = 0
                        save = True
                        self.converged = True
                        break_ = True
                else:
                    if self.k_steps == self.prev_k_steps * 5:
                        self.k_now += 1
                        save = True
                        self.converged = True
                        break_ = True
                        print(f"Reached max steps for k={self.k_now}")

                    self.patience_c = 0

        return {"k_max": self.k_now, "save": save, "break_": break_}

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
        perf_loss=10,
        diff_func_params_eps=None,
        diff_func_params_wac=None,
    ):
        k_init = 1 if k_init is None else k_init

        if diff_func_params_eps is None:
            diff_func_params_eps = {}
            diff_func_params_eps["x_1"] = perf_loss
            diff_func_params_eps["x_2"] = 40
            diff_func_params_eps["V_1"] = 0
            diff_func_params_eps["V_2"] = eps_0
            diff_func_params_eps["type"] = "line"

        if diff_func_params_wac is None:
            diff_func_params_wac = {}
            diff_func_params_wac["x_1"] = perf_loss
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
        save = False
        if not self.change_prot:
            if jnp.abs(current_loss - prev_avg_loss) < self.diff_func_eps(current_loss):
                self.patience_c += 1
                if self.patience_c == self.patience:
                    self.patience_c = 0
                    self.change_prot = True
                    self.k_now += 1
                    save = True
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
        return {"k_max": self.k_now, "save": save}

    def init(self):
        return {"k_max": self.k_now}


class RRAE_fixed_Tracker:
    def __init__(self, k_init):
        self.k_now = k_init

    def __call__(self, *args, **kwargs):
        return {"k_max": self.k_now}

    def init(self):
        return {"k_max": self.k_now}
