import plate_modal_analysis
import animate_mode_shape
import plot_mode_shape

# Calculate plate resonance frequencies
freqs, eps, V = plate_modal_analysis.plate_modal_shell_rm(l1=5*0.3, l2=15*0.3, t=1, E=1.5e9, nu=0.3, rho=500.0, num_modes=20, nx=30, ny=90, bc_type="simply_supported_free")
print(freqs)


# Plot mode shape
mode = [5]
for i in mode:
    freq_mode = freqs[i - 1]
    plot_mode_shape.plot_mode_shape(eps, V, i, freq_mode, 0.2, "simply_supported")


# Animate mode shape
mode = [5]

for i in mode:
    freq_mode = freqs[i - 1]
    animate_mode_shape.animate_mode_shape_interactive_pyvista046(eps, V, i, freq_mode, warp_scale=0.5, fps=120, save_video=False)
