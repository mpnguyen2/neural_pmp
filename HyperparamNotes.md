# Notes

When testing, I try to aim for an initial loss between $[250,350]$ and compare the elbows. Having difficulty getting accurate starting loss (often too high or too low).

## Shape Opt'

Where is phase 2? 

### Adjoint Net Dim
* Original: shape_opt,16,16,"[32, 64]","[64, 8]","[64, 8]",4,[64],"[16, 8]","[16, 8]","[8, 16, 64]"
* shape_opt,16,16,"[32, 64, **32**]","[64, 8]","[64, 8]",4,[64],"[16, 8]","[16, 8]","[8, 16, 64]"
    * Levels out the same but loss is higher (also higher starting loss)

# Notes from last meeting
## Control $u$
$\hat{u}(q_t,p_t)$ is calculated in the `sample_step()` function.
$$
u=\text{einsum}(f_u,-p_i)
$$ 
Where $f_u \in \mathbb{R}^3$ is the partial derivative of dynamics $f$ wrt control $u$ (assuming linear control) and $p_i \in \mathbb{R}^2$ is the momentum given by the Adjoint network (P-net).

## Dynamics $\hat{f}(q_t,\hat{u}_t)$
$\hat{f}(q_t,\hat{u}_t)$ is calculated in the `sample_step()` function. It is returned from the environment as a result of $p_i$ and the control/action given by the agent $\hat{u}$
