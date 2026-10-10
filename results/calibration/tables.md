| Quantity | Test NLL (nats), mean ± sd over 3 seeds |
|---|---|
| true (entropy) | 0.768 ± 0.021 |
| vae_iwae_k128 | 1.578 ± 0.000 |
| ddpm_vlb | 3.777 ± 0.042 |

| Score | Coverage at nominal 0.90 | 5th-95th pct over calibration draws | Coverage ECE |
|---|---|---|---|
| true_nll (oracle) | 0.898 ± 0.008 | 0.874 – 0.918 | 0.0105 ± 0.0057 |
| vae_recon_error | 0.902 ± 0.006 | 0.880 – 0.922 | 0.0038 ± 0.0016 |
| vae_nll | 0.901 ± 0.005 | 0.878 – 0.921 | 0.0029 ± 0.0010 |
| vae_latent_mahalanobis | 0.897 ± 0.008 | 0.875 – 0.918 | 0.0052 ± 0.0008 |
| ddpm_denoising_error | 0.901 ± 0.005 | 0.879 – 0.920 | 0.0050 ± 0.0035 |
| ddpm_nll | 0.895 ± 0.004 | 0.872 – 0.916 | 0.0056 ± 0.0019 |
| raw_mahalanobis | 0.900 ± 0.006 | 0.878 – 0.921 | 0.0038 ± 0.0012 |

| Score | shift 0.0 | shift 0.05 | shift 0.1 | shift 0.2 | shift 0.3 | shift 0.5 | shift 1.0 |
|---|---|---|---|---|---|---|---|
| true_nll (oracle) | 0.898 | 0.859 | 0.748 | 0.476 | 0.326 | 0.198 | 0.287 |
| vae_recon_error | 0.902 | 0.901 | 0.903 | 0.901 | 0.901 | 0.884 | 0.640 |
| vae_nll | 0.901 | 0.898 | 0.895 | 0.870 | 0.818 | 0.696 | 0.446 |
| vae_latent_mahalanobis | 0.899 | 0.895 | 0.886 | 0.869 | 0.841 | 0.783 | 0.668 |
| ddpm_denoising_error | 0.901 | 0.886 | 0.843 | 0.695 | 0.543 | 0.398 | 0.414 |
| ddpm_nll | 0.896 | 0.884 | 0.860 | 0.744 | 0.596 | 0.452 | 0.447 |
| raw_mahalanobis | 0.900 | 0.899 | 0.896 | 0.887 | 0.866 | 0.786 | 0.493 |

| Score | shifted_by_1.0 | uniform_box | wide_gaussian |
|---|---|---|---|
| true_nll (oracle) | 0.842 ± 0.008 | 0.920 ± 0.002 | 0.916 ± 0.005 |
| vae_recon_error | 0.693 ± 0.004 | 0.690 ± 0.013 | 0.615 ± 0.013 |
| vae_nll | 0.764 ± 0.003 | 0.817 ± 0.004 | 0.788 ± 0.006 |
| vae_latent_mahalanobis | 0.577 ± 0.029 | 0.695 ± 0.015 | 0.695 ± 0.012 |
| ddpm_denoising_error | 0.786 ± 0.006 | 0.870 ± 0.002 | 0.848 ± 0.003 |
| ddpm_nll | 0.781 ± 0.005 | 0.849 ± 0.003 | 0.847 ± 0.003 |
| raw_mahalanobis | 0.669 ± 0.005 | 0.760 ± 0.002 | 0.716 ± 0.003 |
