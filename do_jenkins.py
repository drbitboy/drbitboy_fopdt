"""
Implement floating control of model instantiated from fopdt.FOPDT32 class

Usage:

  python do_jenkins.py [--plot] [--scale-control] [--dolog] [--stop-at=1024]
"""
import sys
import fopdt
import numpy

### Process command-line options

### - True/False options:
###   --log:  verbose output; also affects fopdt.FOPDT32 class
###   --plot:  plot data after the run
###   --scale-control:  floating control deltaCV increments are scaled
###                     with flowrate
do_log = '--log' in sys.argv[1:]
do_plot = '--plot' in sys.argv[1:]
do_scale_control = '--scale-control' in sys.argv[1:]

### - Numeric options:
###   --stop-at=T>:  stop simulation after [T] modeled seconds
stop_at = ([False]+[float(s[10:]) for s in sys.argv[1:] if s.startswith('--stop-at=')]).pop()

if "__main__" == __name__:

  ### Instantiate model in steady-state condition
  aeration,inflow,model = fopdt.FOPDT32(system_volume=3000*3000*4096/1e6).init_steady_state()

  if '--plot' in sys.argv[1:]:

    ### Initialize data to plot
    do_data_to_plot = list()
    inflow_to_plot = list()
    air_to_plot = list()
    times_to_plot = list()
    t_disturbances = list()

    ### Use PV as initial setpoint, and assign dead band limits
    sp = model.cells_pg32[-1]/model.pv_factor
    deadband_lo,deadband_hi = sp-0.08,sp+0.08

    ### Parameter to model flow disturbances
    bump_flow = 1.3125

    ### Floating control tuning parameters
    ### - daeration_lo when PV below dead band
    ### - daeration_hi when PV above dead band
    ### - Allow for scaling these tuning parameters with flow
    daeration_lo = aeration * 0.024 / (do_scale_control and (inflow * bump_flow) or 1.0)
    #daeration_hi = -0.7 * daeration_lo
    daeration_hi = -aeration * 0.015 / (do_scale_control and (inflow * bump_flow) or 1.0)

    ### Run model for ~41ks (1<<12 * 10)
    for t in range(40961):

      if stop_at and t>stop_at: break

      ### Disturbances at 256s and every 5120s (~ 1.4h) after start
      ### - Flowrate-only changes
      t_disturbances.append(t)                ### Figure it out ;-)
      if 256==t: inflow *= bump_flow
      elif 5120==t: inflow /= bump_flow
      ### - Influent DO-only changes by +/- 0.2mg/l
      elif 10240==t: model.cells_pg32[0] += model.pv_to_pg32(0.2)
      elif 15360==t: model.cells_pg32[0] -= model.pv_to_pg32(0.2)
      ### - Concurrent flowrate + DO changes in the same direction
      elif 20480==t:
        model.cells_pg32[0] += model.pv_to_pg32(0.2)
        inflow /= bump_flow
      elif 25600==t:
        model.cells_pg32[0] -= model.pv_to_pg32(0.2)
        inflow *= bump_flow
      ### - Concurrent flowrate + DO changes in the opposite direction
      elif 30720==t:
        model.cells_pg32[0] -= model.pv_to_pg32(0.2)
        inflow /= bump_flow
      elif 35840==t:
        model.cells_pg32[0] += model.pv_to_pg32(0.2)
        inflow *= bump_flow
      else: t_disturbances.pop()              ### Figure it out ;-)

      ### Select every eighth point to plot: 128+1 cells yields 17 data
      if 0==(t&7):
        do_data_to_plot.append(model.cells_pg32[::8]/model.pv_factor)
        inflow_to_plot.append(inflow)
        air_to_plot.append(aeration)
        times_to_plot.append(t)

      ### Run model by one time step
      model.model_step(1,aeration,inflow)

      ### Send progress dots to STDOUT if requested (--log)
      if do_log and 0==(t&1023):
        sys.stdout.write('.')
        sys.stdout.flush()

      ### Do not run control algorithm for 63 of every 64 steps
      if (t&63): continue

      ### Run control algorithm for one out of every 64 steps
      ### - Calculate PV from O2
      pv = model.cells_pg32[-1]/model.pv_factor

      ### - Use floating control with dead band
      ### - Set daeration to zero to do nothing if PV in is dead band
      if pv < deadband_lo  : daeration = daeration_lo
      elif pv > deadband_hi: daeration = daeration_hi  ### Negative valu3
      else: daeration = 0.0

      ### - Scale delta-CV if configured to do so, then increment aeration
      if do_scale_control: daeration *= inflow
      aeration += daeration

    ### End of loop
    ####################################################################

    ### Clean up STDOUT if progress dots were output (--log)
    if do_log:
      sys.stdout.write('\n')
      sys.stdout.flush()

    ### Plot data
    import matplotlib.pyplot as plt

    fig,(axflow,axair,axdo,) = plt.subplots(3,1,sharex=True)

    ### Flowrate
    axflow.set_title(r'Aeration $\varpropto$ CV; $\Delta$CV = K$_i$ $\times${0} $\Delta$t '.format(do_scale_control and ' Inflow $\\times$' or ''))
    axflow.plot(times_to_plot,inflow_to_plot)
    axflow.set_ylabel('Influent flowrate, l/s')

    ### Aeration rate
    axair.plot(times_to_plot,air_to_plot)
    axair.set_ylabel('Aeration, mg/s')

    ### DO plots
    ### - Annotation
    axdo.set_ylabel('Cell DO, mg/l')
    axdo.set_xlabel('Time, s')
    ### - Dead band limits
    axdo.axhline(deadband_lo,color='k',ls='dotted',lw=.8)
    axdo.axhline(deadband_hi,color='k',ls='dotted',lw=.8)
    ### - Times of disturbancesl  figure it out ;-)
    for t in t_disturbances:
      axdo.axvline(t,color='k',ls='dotted',lw=.5)

    ### - Plot DO data
    offset = 0
    for dorow in zip(*do_data_to_plot):
      axdo.plot(times_to_plot,numpy.array(dorow),lw=offset and .5 or .8)
      offset += 1

    plt.show()
