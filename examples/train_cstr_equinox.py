#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tqdm import tqdm
import time
import jax
import jax.nn as jnn
import jax.numpy as jnp
from sklearn.preprocessing import StandardScaler
import diffrax
from flax import linen as nn
from typing import Sequence, Dict, Any
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax  # https://github.com/deepmind/optax
from interpolation import ZOHInterpolation
import nonlinear_benchmarks
import nonlinear_benchmarks.error_metrics as metrics
import equinox as eqx  # https://github.com/patrick-kidger/equinox


# In[2]:


#get_ipython().run_line_magic('matplotlib', 'widget')


# In[3]:


train_val, test = nonlinear_benchmarks.Cascaded_Tanks(atleast_2d=True)
sampling_time = train_val.sampling_time
u_train, y_train = train_val


# In[4]:


# Rescale data
scaler_u = StandardScaler()
u = scaler_u.fit_transform(u_train).astype(jnp.float32)

scaler_y = StandardScaler()
y = scaler_y.fit_transform(y_train).astype(jnp.float32)

ts = jnp.arange(0.0, u.shape[0]) * sampling_time


# In[5]:


nx = 2
nu = u.shape[-1]
ny = y.shape[-1]


# In[6]:


class StateUpdateMLP(eqx.Module):
    mlp: eqx.nn.MLP
    scale: float = 1e-3

    def __init__(self, nx, nu, width, depth, *, scale, key, **kwargs):
        super().__init__(**kwargs)
        self.mlp = eqx.nn.MLP(nx + nu, nx, width, depth, jnn.softplus, key=key)
        self.scale = scale

    def __call__(self, x, u, args):
        dx = self.scale * self.mlp(jnp.r_[x, u])
        return dx  
    

class NeuralODE(eqx.Module):
    f_xu: StateUpdateMLP
    g_x: eqx.nn.MLP

    def __init__(self, f_xu, g_x):
        super().__init__()
        self.f_xu = f_xu
        self.g_x = g_x

    def __call__(self, ts, us, x0):

        u_fun = ZOHInterpolation(ts=ts, ys=us.ravel()) # input interpolation function

        def vector_field(t, x, args):
            ut = u_fun.evaluate(t)[..., None] # u interpolated at time t
            dx = self.f_xu(x, ut, args)
            return dx
    
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(vector_field),
            #diffrax.Tsit5(),
            diffrax.Euler(),
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=x0,
            #stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            saveat=diffrax.SaveAt(ts=ts),
            max_steps=100_000
        )
        x_sim = solution.ys
        return jax.vmap(self.g_x)(x_sim)


# In[7]:


f_xu = StateUpdateMLP(nx, nu, 32, 2, scale=1e-3, key=jax.random.PRNGKey(0))
g_x = eqx.nn.MLP(nx, ny, 16, 1, jnn.softplus, key=jax.random.PRNGKey(1))
simulator = NeuralODE(f_xu, g_x)
#simulator(ts, u, jnp.zeros(nx))


# In[8]:


x0 = jnp.zeros(nx)
opt_variables = (simulator, x0)

@eqx.filter_value_and_grad
def loss_grad_fn(opt_variables, t, u, y):
    simulator, x0 = opt_variables
    y_pred = simulator(t, u, x0)
    return jnp.mean((y - y_pred) ** 2)

#loss_grad_fn(opt_variables, ts, u, x0)


# In[9]:


# Setup optimizer
optimizer = optax.adam(learning_rate=1e-4)
#opt_state = optimizer.init(eqx.filter(opt_variables, eqx.is_inexact_array))
opt_state = optimizer.init(eqx.filter(opt_variables, eqx.is_inexact_array))


# In[10]:


#[eqx.filter(simulator, eqx.is_inexact_array), x0]
#eqx.filter(opt_variables, eqx.is_inexact_array)[1]
#type(updates[0])


# In[12]:


# Training loop
time_start = time.time()
LOSS = []
epochs = 10_000
for epoch in (pbar := tqdm(range(epochs))):
    loss_val, grads = loss_grad_fn(opt_variables, ts, u, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    opt_variables = optax.apply_updates(opt_variables, updates)
    LOSS.append(loss_val)
    if epoch % 100 == 0:
        pbar.set_postfix_str(f"Loss step {epoch}: {loss_val}")
    #print()

train_time = time.time() - time_start
print(f"Training time: {train_time:.2f}")


# In[ ]:


#optax.apply_updates(simulator, updates)

