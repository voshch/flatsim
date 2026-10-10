Axis linear        # forward axis, m/s
Axis lateral       # sideways axis, m/s, low = high = 0 for a non-holonomic base
Axis angular       # turn axis, rad/s
float64[3] gain    # body twist per unit setpoint: linear, lateral, angular
float64 lag        # first-order response time, s, 0 = immediate
float64 traction   # planar acceleration limit, m/s^2, inf = unbounded
float64 pivot      # forward offset of the turn center from the base, m
float64 timeout    # sim seconds a command stays in force
