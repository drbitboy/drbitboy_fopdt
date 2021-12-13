# drbitboy_fopdt
Crude model of sludge water aeration cf. https://www.plctalk.net/qanda/showthread.php?t=131415
# Usage
### As shell command
    python fopdt.py [--plot]
### As Python model
    import matplotlib.pyplot as plt
    import fopdt
    aeration,inflow,model = fopdt.FOPDT32().init_steady_state()
    daeration = aeration * 1e-4
    dinflow = inflow * 1e-4
    o2_data = list()
    for i in range(1024):
      model.model_step(1,aeration,inflow)
      o2_data.append(model.cells_pg32[-1]/model.pv_factor)
      aeration += daeration
      inflow -= dinflow
    
    plt.plot(o2_data)
    plt.show()
# Manifest
### fopdt.py
* Python module fopdt with class FOPDT32
* Requires module numpy
* Require module matplotlib to plot
# Result of [python fopdt.py --plot]
![](https://github.com/drbitboy/drbitboy_fopdt/raw/master/do_ditch.png)
