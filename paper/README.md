# Hopper figures for the paper

The PDFs in [`figures/`](figures/) are vector graphics sized for a two-column paper. The PNGs are 300 dpi previews. Generate them from the current `runs_2m_sphere/mo_hopper` data with:

```bash
python paper/plot_hopper.py
```

[`figures.tex`](figures.tex) contains a ready-to-adapt LaTeX block for the four-column figure and its separate legend.

| Use | PDF | Preview |
|---|---|---|
| Main four-column figure | [hopper_5eval_four_panel.pdf](figures/hopper_5eval_four_panel.pdf) | [PNG](figures/hopper_5eval_four_panel.png) |
| Separate legend | [hopper_5eval_legend.pdf](figures/hopper_5eval_legend.pdf) | [PNG](figures/hopper_5eval_legend.png) |
| Alternative: performance only | [hopper_5eval_performance.pdf](figures/hopper_5eval_performance.pdf) | [PNG](figures/hopper_5eval_performance.png) |
| Alternative: directional reward only | [hopper_5eval_directions.pdf](figures/hopper_5eval_directions.pdf) | [PNG](figures/hopper_5eval_directions.png) |
| Appendix: mean return by seed | [hopper_5eval_seeds.pdf](figures/hopper_5eval_seeds.pdf) | [PNG](figures/hopper_5eval_seeds.png) |
| Appendix: forward reward by seed | [hopper_5eval_forward_seeds.pdf](figures/hopper_5eval_forward_seeds.pdf) | [PNG](figures/hopper_5eval_forward_seeds.png) |
| Appendix: vertical reward by seed | [hopper_5eval_vertical_seeds.pdf](figures/hopper_5eval_vertical_seeds.pdf) | [PNG](figures/hopper_5eval_vertical_seeds.png) |

## Suggested captions

**Four-column figure.** Hopper-v5 results for Uniform and D-LEVER. Panels show mean task return, CVaR10, discounted forward-velocity reward, and discounted vertical-distance reward, respectively. The vertical reward coordinate is `10 × z_distance_from_origin`. Each plotted point averages five consecutive evaluations, ending at the indicated environment step; evaluations occur every 50,000 steps. Lines show the mean over seeds with both methods available at that step, and shaded regions show one standard error across those seeds. Each task return is an interquartile mean across evaluation episodes; with ten fixed evaluation tasks, CVaR10 is the lowest task return. Four paired seeds contribute through 1.70 million steps; three contribute from 1.75 to 2.00 million steps. The legend is saved separately for placement in the paper.

**Directional rewards.** Mean discounted forward-velocity and vertical-distance reward components on the fixed Hopper evaluation tasks. The plotted reward coordinates are `x_velocity` and `10 × z_distance_from_origin`, respectively. Each point first averages five consecutive evaluations for each seed and method, then averages paired seeds. Shading shows one standard error across paired seeds. The paired-seed count is four through 1.70 million steps and three thereafter.

**Seed panels.** Five-evaluation moving averages for each Hopper training seed. These panels show the seed-level variation underlying the aggregate figures. Seed 4 ends earlier than seeds 0, 1, and 2.

The first moving-average point is at 250,000 environment steps. Adjacent points share four of five evaluations, so they should be read as a smoothed trajectory rather than independent measurements. The source data has no seed 3 run.
