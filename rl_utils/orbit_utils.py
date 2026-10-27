import jax
import jax.numpy as jnp

from utils.planet_params import PlanetParams, Earth


def orbital_elements_to_pci(elements: jax.Array, planet=Earth):
    """
    Vectorized conversion for multiple sets of orbital elements.
    
    Parameters
    ----------
    elements : jax.Array
        Array of shape (N, 6) with columns [a, e, i, raan, omega, nu]
        a : float
        Semi-major axis (km)
        e : float
            Eccentricity (dimensionless)
        i : float
            Inclination (radians)
        raan : float
            Right Ascension of Ascending Node (radians)
        omega : float
            Argument of Periapsis (radians)
        nu : float
            True Anomaly (radians)
        
    Returns
    -------
    pos_and_vel_eci : jax.Array
        Position and velocity vectors in PCI frame (km) [N, 6]
    """
    mu = planet.mu 
    
    # Use ellipsis to support (6,) or (N, 6)
    a     = elements[..., 0]
    e     = elements[..., 1]
    i     = elements[..., 2]
    raan  = elements[..., 3]
    omega = elements[..., 4]
    nu    = elements[..., 5]
  
    # Semi-latus rectum and magnitude
    p = a * (1 - e**2)
    r_mag = p / (1 + e * jnp.cos(nu))
    
    # Position in perifocal (PQW) frame (components are shape (N,) or scalar)
    r_p = r_mag * jnp.cos(nu)
    r_q = r_mag * jnp.sin(nu)
    # r_w is 0
    
    # Velocity in perifocal (PQW) frame
    v_factor = jnp.sqrt(mu / p)
    v_p = v_factor * -jnp.sin(nu)
    v_q = v_factor * (e + jnp.cos(nu))
    # v_w is 0
    
    # Pre-calculate trig terms
    c_raan, s_raan = jnp.cos(raan), jnp.sin(raan)
    c_i, s_i       = jnp.cos(i), jnp.sin(i)
    c_om, s_om     = jnp.cos(omega), jnp.sin(omega)

    # Manual rotation PQW -> PCI coordinates   
    # Position X, Y, Z
    x_eci = (c_raan * c_om - s_raan * s_om * c_i) * r_p + \
            (-c_raan * s_om - s_raan * c_om * c_i) * r_q
            
    y_eci = (s_raan * c_om + c_raan * s_om * c_i) * r_p + \
            (-s_raan * s_om + c_raan * c_om * c_i) * r_q
            
    z_eci = (s_om * s_i) * r_p + (c_om * s_i) * r_q
    
    # Velocity X, Y, Z
    vx_eci = (c_raan * c_om - s_raan * s_om * c_i) * v_p + \
             (-c_raan * s_om - s_raan * c_om * c_i) * v_q
             
    vy_eci = (s_raan * c_om + c_raan * s_om * c_i) * v_p + \
             (-s_raan * s_om + c_raan * c_om * c_i) * v_q
             
    vz_eci = (s_om * s_i) * v_p + (c_om * s_i) * v_q
    
    # Stack along the last dimension to get (N, 6) or (6,)
    return jnp.stack([x_eci, y_eci, z_eci, vx_eci, vy_eci, vz_eci], axis=-1)