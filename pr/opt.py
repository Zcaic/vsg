import aerosandbox as asb

naca0012=asb.Airfoil(name='naca0012')
aero=naca0012.get_aero_from_neuralfoil()