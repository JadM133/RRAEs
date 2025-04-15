from RRAEs.utilities import get_diff_func
import jax.numpy as jnp


class Null_Tracker:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, current_loss, prev_avg_loss, *args, **kwargs):
        return {}

    def init(self):
        return {}

class RRAE_Null_Tracker:
    def __init__(self, k_max, *args, **kwargs):
        self.k_max = k_max

    def __call__(self, current_loss, prev_avg_loss, *args, **kwargs):
        return {"k_max": self.k_max}

    def init(self):
        return {"k_max": self.k_max}

class VRRAE_Null_Tracker:
    def __init__(self, k_max, sigma, *args, **kwargs):
        self.sigma = sigma
        self.k_max = k_max

    def __call__(self, current_loss, prev_avg_loss, *args, **kwargs):
        return {"sigma": self.sigma, "k_max": self.k_max}

    def init(self):
        return {"sigma": self.sigma, "k_max": self.k_max}


class VRRAE_sigma_Tracker:
    def __init__(self, k_max, sigma0=10, sigmaf=3, steps=200, steps_last=1400, jump=3, *args, **kwargs):
        self.sigma = sigma0
        self.sigmaf = sigmaf
        self.jump = jump
        self.steps = steps
        self.steps_last = steps_last
        self.step_c = 0
        self.k_max = k_max
        self.st = False

    def __call__(self, current_loss, prev_avg_loss, *args, **kwargs):
        self.step_c += 1
        if (self.step_c == self.steps) and (self.sigma > self.sigmaf):
            self.step_c = 0
            self.sigma = max(self.sigma - self.jump, self.sigmaf)
        elif (self.step_c == self.steps_last) and (self.sigma == self.sigmaf):
            self.st = True
        
        return {"sigma": self.sigma, "k_max": self.k_max, "stop_train": self.st}

    def init(self):
        return {"sigma": self.sigma, "k_max": self.k_max}


class RRAE_gen_Tracker:
    def __init__(
        self,
        k_init,
        patience_conv=1,
        patience_init=None,
        patience_not_right=500,
        max_steps=1000,
        perf_loss=0,
        eps_0=1,
        eps_perc=5,
        save_steps=20
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
        self.patience_init = patience_init
        self.init_phase = True
        self.ideal_loss = jnp.nan
        self.eps_0 = eps_0
        self.perf_loss = perf_loss
        self.eps_perc = eps_perc
        self.k_steps = 0
        self.prev_k_steps = 0
        self.max_patience = jnp.inf
        self.save_steps = save_steps
        self.stop_train = False

    def __call__(self, current_loss, prev_avg_loss, *args, **kwargs):
        save = False
        break_ = False
        self.prev_k_steps += 1
        if self.init_phase:
            if self.patience_init is not None:
                if (
                    jnp.abs(current_loss - prev_avg_loss) / jnp.abs(prev_avg_loss) * 100
                    < self.eps_perc
                ):
                    self.patience_c += 1
                    if self.patience_c == self.patience_init:
                        self.patience_c = 0
                        self.init_phase = False
                        self.ideal_loss = prev_avg_loss
                        print(f"Ideal loss is {self.ideal_loss}")
                        print("Stagnated")
            
            if current_loss < self.perf_loss:
                self.ideal_loss = self.perf_loss
                self.init_phase = False
                self.patience_c = 0
                print(f"Ideal loss is {self.ideal_loss}")

            return {"k_max": self.k_now, "save": save, "break_": break_, "stop_train": self.stop_train}

        self.total_steps += 1
        if not self.converged:
            if current_loss < self.ideal_loss:
                self.patience_c = 0
                self.k_steps = 0
                self.patience_c_conv += 1
                if self.patience_c_conv == self.patience_conv:
                    self.patience_c_conv = 0
                    self.k_now -= 1
                    if (self.total_steps == self.patience_conv):
                        save = False
                    else:
                        if self.prev_k_steps >= self.save_steps:
                            save = True
                    self.prev_k_steps = 0
                    self.total_steps = 0
            else:
                self.patience_c_conv = 0
                self.k_steps += 1
                if jnp.abs(current_loss - prev_avg_loss)/jnp.abs(prev_avg_loss)*100 < self.eps_0:
                    self.k_steps = 0
                    self.patience_c += 1
                    if self.patience_c == self.patience:
                        self.patience_c = 0
                        self.k_now += 1
                        self.prev_k_steps = 0
                        save = True
                        self.converged = True
                        break_ = True
                        print("adding one and shit")
                        
        else:
            if jnp.abs(current_loss - prev_avg_loss)/jnp.abs(prev_avg_loss)*100 < self.eps_perc:
                self.patience_c += 1
                if self.patience_c == self.patience:
                    self.patience_c = 0
                    self.prev_k_steps = 0
                    save = True
                    self.stop_train = True
                    print("Stopping training")

        return {"k_max": self.k_now, "save": save, "break_": break_, "stop_train": self.stop_train}

    def init(self):
        return {"k_max": self.k_now}


class RRAE_pars_Tracker:
    def __init__(
        self,
        k_init=None,
        patience=5000,
        max_wait=500,
        eps_perc=1,
        perf_loss=10,
        diff_func_params_eps=None,
        diff_func_params_wac=None,
    ):
        k_init = 1 if k_init is None else k_init

        self.patience = patience
        self.eps = eps_perc
        self.patience_c = 0
        self.change_prot = False
        self.loss_prev_mode = jnp.inf
        self.wait_counter = 0
        self.max_wait = max_wait
        self.k_now = k_init
        self.converged = False
        self.stop_train = False

    def __call__(self, current_loss, prev_avg_loss, *args, **kwargs):
        save = False
        break_ = False
        print(self.patience_c)
        if not self.converged:
            if jnp.abs(current_loss - prev_avg_loss) < self.eps:
                self.patience_c += 1
                if self.patience_c == self.patience:
                    self.patience_c = 0
                    save = True
                
                    if jnp.abs(prev_avg_loss - self.loss_prev_mode)/jnp.abs(self.loss_prev_mode)*100 <  self.eps:
                        import pdb; pdb.set_trace()
                        self.k_now -= 1
                        break_ = True
                        self.converged = True
                        self.patience_c = 0
                    else:
                        self.k_now += 1
                        self.loss_prev_mode = prev_avg_loss
        else:
             if jnp.abs(current_loss - prev_avg_loss)/jnp.abs(prev_avg_loss)*100 < self.eps:
                self.patience_c += 1
                if self.patience_c == self.patience:
                    self.patience_c = 0
                    self.prev_k_steps = 0
                    save = True
                    self.stop_train = True
                    print("Stopping training")    
        
        return {"k_max": self.k_now, "save": save, "break_": break_, "stop_train": self.stop_train}

    def init(self):
        return {"k_max": self.k_now}


class RRAE_fixed_Tracker:
    def __init__(self, k_init):
        self.k_now = k_init

    def __call__(self, *args, **kwargs):
        return {"k_max": self.k_now}

    def init(self):
        return {"k_max": self.k_now}
