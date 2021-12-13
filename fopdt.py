import os
import sys
import numpy

i32,f32 = numpy.int32,numpy.float32
f32_one = f32(1)
i32_one = i32(1)
i32_two = i32_one+i32_one
i32_negtwo = -i32_two
i32_max = (1<<31)-1

def mean32(base, delta):
  """Obsolete: mean of numpy.int32 values base and (base+delta)"""
  assert isinstance(base,i32) and isinstance(delta,i32)
  halfdelta = numpy.bitwise_and(delta,i32_negtwo)//i32_two
  return (base + halfdelta)

a=i32

class FOPDT32:
  """
First-Order Plus Dead Time (FOPDT) model

Cf. https://www.plctalk.net/qanda/showthread.php?t=131429

Model slow-moving flow of water through ditch with long residence time

- Residence time, i.e. ratio of [Ditch Volume:Flowrate through ditch] is
  a dozen minutes or more
- Use twos-complement, 32-bit, fixed-point data to store model, as it
  might need to be on a Programmable Logic Controller emulating such
  a process

Model parameters (arbitray units)

- Independent and variable
  - Flow into ditch, l/s (l=>litre)
  - Oxygen added via aerators, mg/s
  - Oxygen concentration (O2) in influent, mg/l

- Dependent output
  - Oxygen concentration of effluent, mg/l

- Design constants
  - Model size:  number of cells (slices) along direction of flow
  - Volume of water in ditch, l
  - Factor from oxygen concentration (mg/l) to oxygen per-cell*
    - arbitrary units
      - e.g. pg/(mg/l) if per-cell volume is 1l
        - pg => picogram
  - Per-cell volume, calculate from total volume and number of cells

- Internal

Model assumptions

- Cells are well-mixed i.e. homogeneous oxygen concentration in any cell
- Constant flow over entire cross-sectional area of ditch normal to flow
- Volume of water in ditch does not change, level is controlled

"""
  ######################################################################
  def __init__(self
              ,model_size=128
              ,system_volume=None
              ,pv_factor=1e9
              ):
    """
Define model using input arguments as design parameters
-    model_size:  number of cells to model ditch
- system_volume:  volume of water in ditch
-     pv_factor:  Conversion from mg/l to a per-cell quantity of oxygen

Attributes
- .model_size
- .system_volume
- .pv_factor:  scale factor from O2 mg/l to 32-bit per-cell quantity
- .cell_volume:  volume of one cell
- .cells_pg32:  (model_size+1) cells, array of numpy.int32
  - 32-bit integer quantity of O2 dissolved in each cell
  - one cell for influent
  - (model_size) cells for aerated water
- .pv_to_pg32 - scale from O2 mg/l to 32-bit quantity using .pv_factor
- .amount_to_pg32 - Function, convert from per-cell mg O2 to 32-bit quantity

"""
    self.model_size = i32(model_size)
    self.system_volume = f32(None is system_volume
                             and self.model_size
                             or system_volume
                            )
    self.pv_factor = f32(pv_factor)

    self.cell_volume = f32(self.system_volume/model_size)
    self.cells_pg32 = numpy.zeros(model_size+1,dtype=i32)

    self.pv_to_pg32 = lambda specific: i32(numpy.round(specific*self.pv_factor,0))
    self.amount_to_pg32 = lambda amount: self.pv_to_pg32(amount/self.cell_volume)

  ######################################################################
  def model_step(self,deltat,aeration,inflow,influent_specific=None):
    """
Model one timestep of the model using the Implicit Euler method, given
- delta_s:  timestep size, s
- aeration:  total oxygen aeration rate, mg/s
- inflow:  flowrate, l/s
- influent_specific:  influent  oxygen concentration, mg/l

  Model is mass balance:  Per-cell Accumulation = (O2 In) - (O2 out)
  - F = Flowrate
  - V = Per-cell volume
  - O[i] = O2 content of Cell [i]
  - O[i-1] = O2 content of next cell upstream from Cell [i]
  - O[0] = O2 content of shadow "Cell 0" i.e. influent
  - Discrete form, Implicit Euler Method i.e. use O[i](t+dt)
  - O2 In:
    - Per-cell aeration = dt * dAi/dt
    - Flow from upstream cell = dt * F * Concentration[i-1](t+dt)
                              = dt * F * O[i-1](t+dt) / V
                              = (dt * F / V) * O[i-1](t+dt) / V
  - O2 out:
    - Flow to downstream cell = (dt * F / V) * O[i](t+dt)
  - Full equation
    - Per-cell, per-timestep accumulation is dO[I]
    - No inter-cell diffusion

dO[i] = dt (dAi/dt + (F/V) (O[i-1](t+dt) - O[i](t+dt)       )
dO[i] = dt (dAi/dt + (F/V) (O[i-1](t+dt) - (O[i](t) + dO[i]))

  - O[i-1](t+dt) is known from previous cells solution or influent
  - O[i](t) is known from previous timestep
  - Linear in dO[i], solve for dO[i]:
  - dtF = dt * F
  - dtFoV = dtF / V

dO[i] = dt dAi/dt + dtFoV O[i-1](t+dt) - dtFoV(O[i](t) - dtFov dO[i]
dO[i] + dtFov dO[i] = dt dAi/dt + dtFoV O[i-1](t+dt) - dtFoV O[i](t)
dO[i] (1 + dtFoV)   = dt dAi/dt + dtFoV (O[i-1](t+dt) - O[i](t))

  - Multiply through by V

dO[i] (V+dtF) = dt V dAi/dt + dtF (O[i-1](t+dt) - O[i](t))

  - dA/dt is total system aeration
  - N is number of cells in model
  - dAi/dt = dA/dt / N
  - Solution as coded:

dO[i] (V+dtF) = (  V / (V+dtF)) (dt dA/dt / N)
              + (dtF / (V+dtF)) (O[i-1](t+dt) - O[i](t))

"""
    dt32 = f32(deltat)
    F32 = f32(inflow)
    dtF32 = dt32 * F32

    denom = self.cell_volume + dtF32
    ratioO = dtF32 / denom

    ### Per-cell, per-timestep Aeration, mg = dt (dAdt,mg/s) / N
    dAi = dt32 * f32(aeration) / f32(self.model_size)

    ### dAi32 = scaled per-cell, per timestep 32-bit O2 from Aeration
    ###       = (V / (V + dt*F)) * dAi
    dAi32 = self.amount_to_pg32(dAi * self.cell_volume / denom)

    ### Loop over cells
    for i in range(self.model_size+1):

      ### Special case:  "Cell 0" influent O2 is independent variable
      if 0==i:
        if not (None is influent_specific):
          self.cells_pg32[i] = self.pv_to_pg32(influent_specific)
        continue

      ### 32-bit per-cell, per timestep O2 from flow in and out
      ### dO32 = ((dt*F) / (V + dt*F)) * (O[i-1](t+dt) - O[i](t))
      dO32 = i32(ratioO * (self.cells_pg32[i-1] - self.cells_pg32[i]))

      ### Clamp delta to not exceed 32-bit max value
      delta32 = min([i32_max-self.cells_pg32[i]  ### delta to 32-bit max
                    ,dO32+dAi32                  ### calculated delta
                    ])

      self.cells_pg32[i] += delta32          ### Update O2 content array

    ### End of per-cell loop
    ####################################################################


  ######################################################################
  def init_steady_state(self
                       ,influent_specific=.32
                       ,effluent_specific=1.6
                       ,inflow=None
                       ):
    """
Initialize cell O2 quantities for steady-state condition,
calculate O2 aeration and total water flow

Returns tuple:
- aeration_rate, mg/s
- total_water flow, l/s
- self
  - E.g. aeration,flowrate,model = fopdt.FOPDT32().init_steady_state()


Arguments
- influent_specific:  steady-state O2 mg/l in influent of ditch
- effluent_specific:  steady-state O2 mg/l in effluent of ditch
- inflow:  steady-state flowate of influent, l/s; see below

If inflow is None, set flowrate for 8s residence time in each cell

"""
    eff_vec = numpy.arange(self.model_size+1) / self.model_size
    inf_vec = 1.0 - eff_vec
    self.cells_pg32[:] = self.pv_to_pg32((influent_specific*inf_vec)+(effluent_specific*eff_vec))
    total_water_flow = None is inflow and (self.cell_volume/8.0) or float(inflow)
    aeration_rate = (effluent_specific - influent_specific) * total_water_flow
    print(self.cells_pg32)
    print(dict(residence_time_minutes=(self.cell_volume*self.model_size/total_water_flow)/60.
              ,aeration_rate_mg_per_s=aeration_rate
              ,**{'total_water_flow_l/s':total_water_flow}
              ))
    return aeration_rate,total_water_flow,self


########################################################################
if "__main__" == __name__:
  import fopdt
  print(fopdt.FOPDT32().init_steady_state())
  aeration,inflow,model = all3 = fopdt.FOPDT32(system_volume=3000*3000*4096/1e6).init_steady_state()
  print(all3)
  if '--plot' in sys.argv[1:]:
    data_to_plot = list()
    inflow_to_plot = list()
    air_to_plot = list()
    times_to_plot = list()
    for t in range(4097):
      if 256==t: inflow *= 1.5
      if 1024==t: model.cells_pg32[0] *= .75
      if 2048==t: aeration *= 1.5
      if 0==(t&7):
        data_to_plot.append(model.cells_pg32[::8]/model.pv_factor)
        inflow_to_plot.append(inflow)
        air_to_plot.append(aeration)
        times_to_plot.append(t)
      model.model_step(1,aeration,inflow)
      if 0==(t&127):
        sys.stdout.write('.')
        sys.stdout.flush()
    sys.stdout.write('\n')
    sys.stdout.flush()
    import matplotlib.pyplot as plt
    fig,(axflow,axair,axdo,) = plt.subplots(3,1,sharex=True)

    axflow.plot(times_to_plot,inflow_to_plot)
    axflow.set_ylabel('Influent flowrate, l/s')

    axair.plot(times_to_plot,air_to_plot)
    axair.set_ylabel('Aeration, mg/s')

    axdo.set_ylabel('Cell DO, mg/l')
    axdo.set_xlabel('Time, s')
    offset = 0
    for dorow in zip(*data_to_plot):
      axdo.plot(times_to_plot,numpy.array(dorow),lw=.5)
      offset += 1

    plt.show()
