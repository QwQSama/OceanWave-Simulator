# Ocean Wave Simulation

## Project Overview

This project simulates the motion of ocean waves using a simple mathematical model based on the Gerstner wave model. Implemented in Python, the simulation generates visually realistic ocean wave patterns over time using the Matplotlib library. The animated simulation is saved as a GIF.

## Table of Contents

- [Introduction](#introduction)
- [Methodology](#methodology)
- [Implementation](#implementation)
- [Results](#results)
- [Future Work](#future-work)

## Introduction

Ocean waves are complex natural phenomena influenced by various factors such as wind and gravity. This project simulates ocean waves using a parametric surface model to represent wave behavior over time. The primary objective is to create a computationally efficient and realistic visualization of ocean waves.

## Methodology

We employed the **Lagrangian method** to model the surface of the ocean. In this approach, the motion of individual points on the ocean surface is calculated based on sinusoidal wave equations. The surface is treated as a graphical primitive, and particles are set in circular motion around their rest positions, creating the wave-like movement.

### Key Equations:

- For the X and Z coordinates of each point:
  \[
  x = x_0 + r * \sin(\kappa x_0 - \omega t)
  \]
  \[
  z = z_0 - r * \cos(\kappa x_0 - \omega t)
  \]
  
Where:
- \( r \) is the wave amplitude
- \( \kappa \) is the wavenumber
- \( \omega \) is the angular frequency
- \( t \) is time

These equations generate the trochoidal shape of waves over time, creating a visually appealing simulation of ocean surface dynamics.

## Implementation

- **Language**: Python
- **Libraries**: Numpy, Matplotlib, Matplotlib Animation

The ocean surface is represented by a grid, and the height of the waves is computed at each point using wave equations. The animation is created using the `matplotlib.animation` module, which updates the wave positions over time.

The animated wave simulation is stored as a GIF file.

## Results

Here is a GIF of the generated ocean wave simulation:

![Ocean Wave Simulation](fluid.gif)

The generated waves realistically depict the motion of the ocean surface, with varying wave heights and movements that closely resemble natural ocean behavior.


## Future Work

- **Wave Interaction**: Introduce wave interference and interaction to simulate more complex wave patterns.
- **Foam and Surf**: Add a particle system to simulate foam and surf near the shorelines.
- **Wind Influence**: Model wind direction and speed to dynamically alter wave patterns.
