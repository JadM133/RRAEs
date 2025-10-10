import jax.numpy as jnp

class Null_Tracker:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, current_loss, prev_avg_loss, *args, **kwargs):
        return {}

    def init(self):
        return {}

class RRAE_fixed_Tracker:
    """ Default Tracker for RRAEs that provides fixed value of k_max """
    def __init__(self, k_init):
        self.k_now = k_init

    def __call__(self, *args, **kwargs):
        return {"k_max": self.k_now}

    def init(self):
        return {"k_max": self.k_now}
    

class RRAE_gen_Tracker:
    """ Tracker that performs the generic adaptive algorithm.
    
    The algorithm begins with a large value of k_max provided as k_init, it starts
    training until convergence, and saves the loss value as 'optimal loss'. Then 
    k_max is decreased one by one and training is continued until convergence to 
    the optimal loss each time. Finally, when a value of k_max is too low that
    the loss never converges again to the optimal value, k_max is re-increased
    by one and training is continued.
    
    Parameters
    ----------
    k_init: start value of k_max (choose to be big)
    patience_conv: patience for convergence (assuming that loss stagnated)
    patience_init: how many forward steps to do as 'initialization', so
                   before the adaptive algorithm starts.
    patience_not_right: patience for assuming that the value of k_max
                        is too small
    perf_loss: the optimal loss, if known a priori
    eps_0: the error in percent, below which we assume stagnation
           for the initialmization phase
    eps_perc: the error in percent, below which we assume stagnation
           after the initialization part is done
    save_steps: if set to True, model is saved when changing k value
    k_min: the minimum value of k, if known in advance
    converged_steps: number of steps after convergence to the right k
                    if this parameter is not set, the stagnation 
                    criteria will determine when training is stopped
    """
    def __init__(
        self,
        k_init,
        patience_conv=1,
        patience_init=None,
        patience_not_right=500,
        perf_loss=0,
        eps_0=1,
        eps_perc=5,
        save_each_k=False,
        k_min=0,
        converged_steps=jnp.inf,
    ):

        self.patience_c_conv = 0
        self.patience_c = 0
        self.steps_c = 0
        self.k_now = k_init

        self.change_prot = False
        self.loss_prev_mode = jnp.inf
        self.wait_counter = 0
        self.k_now = k_init
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
        self.max_patience = jnp.inf
        self.save_each_k = save_each_k
        self.stop_train = False
        self.k_min = k_min
        self.converged_steps = converged_steps
        self.converged_steps_c = 0

    def __call__(self, current_loss, prev_avg_loss, *args, **kwargs):
        save = False
        break_ = False
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

            return {"k_max": self.k_now, "save": save, "break_": break_, "stop_train": self.stop_train, "load": False}

        self.total_steps += 1

        if (self.k_now <= self.k_min) and (not self.converged):
          print("Converged to minimum value")
          self.converged = True

        load = False

        if not self.converged:
            if current_loss < self.ideal_loss:
                self.patience_c = 0
                self.k_steps = 0
                self.patience_c_conv += 1
                if self.patience_c_conv == self.patience_conv:
                    self.patience_c_conv = 0
                    self.k_now -= 1
                    save = True
                    self.total_steps = 0
            else:
                self.patience_c_conv = 0
                self.k_steps += 1
                stg = jnp.abs(current_loss - prev_avg_loss)/jnp.abs(prev_avg_loss)*100 < self.eps_0
                worse = current_loss > prev_avg_loss
                if stg or worse:
                    self.k_steps = 0
                    self.patience_c += 1
                    if self.patience_c == self.patience:
                        self.patience_c = 0
                        self.k_now += 1
                        save = False
                        load = True
                        self.converged = True
                        break_ = True
                        print("Reached a k_max that's too low, adding 1 to k_max")
                        
        else:
            self.converged_steps_c += 1

            if self.converged_steps_c >= self.converged_steps:
                self.stop_train = True
                save = True
                self.patience_c = 0
                
            else:
                if jnp.abs(current_loss - prev_avg_loss)/jnp.abs(prev_avg_loss)*100 < self.eps_perc:
                    self.patience_c += 1
                    if self.patience_c == self.patience:
                        self.patience_c = 0
                        save = True
                        self.stop_train = True
                        print("Stopping training")

        return {"k_max": self.k_now, "save": save, "break_": break_, "stop_train": self.stop_train, "load": load}

    def init(self):
        return {"k_max": self.k_now}


class RRAE_pars_Tracker:
    """ Tracker that performs the parsimonious adaptive algorithm.
    
    The algorithm begins with the smallest value of k_max (1 if not given)
    then the value of k_max is increased by 1 everytime convergence is reached.

    NOTE: It is not recommended to use this algorithm, it usually leads to less
    explainable, and worse, results. But it is here if someone wants to experiment.
    
    Parameters
    ----------
    k_init: start value of k_max (choose to be small)
    patience: how much to wait to assume stagnation
    eps_perc: error in percent under which we assume stagnation
    """
    def __init__(
        self,
        k_init=None,
        patience=5000,
        eps_perc=1
    ):
        k_init = 1 if k_init is None else k_init

        self.patience = patience
        self.eps = eps_perc
        self.patience_c = 0
        self.loss_prev_mode = jnp.inf
        self.k_now = k_init
        self.converged = False
        self.stop_train = False

    def __call__(self, current_loss, prev_avg_loss, *args, **kwargs):
        save = False
        break_ = False
        if not self.converged:
            if jnp.abs(current_loss - prev_avg_loss) < self.eps:
                self.patience_c += 1
                if self.patience_c == self.patience:
                    self.patience_c = 0
                    save = True
                
                    if jnp.abs(prev_avg_loss - self.loss_prev_mode)/jnp.abs(self.loss_prev_mode)*100 <  self.eps:
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
